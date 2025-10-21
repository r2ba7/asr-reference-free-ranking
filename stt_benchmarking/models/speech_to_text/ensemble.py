from difflib import SequenceMatcher
from collections import defaultdict
from itertools import islice
from collections import Counter, OrderedDict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import time
import hashlib
import math
import difflib

import numpy as np
from tqdm import tqdm
from Levenshtein import distance
from rapidfuzz import fuzz
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

from stt_benchmarking.utils import text_processing, helpers, metrics
from stt_benchmarking.models.llms import reinforcer
from . import LOGGER

class HybridEnsemble:
    def __init__(self, use_llm, mode="conservative", perfection_rule_length=5, dynamic_match_percentage=0.8):
        self.use_llm = use_llm
        if self.use_llm:
            self._initialize_llm()

        self.perfection_rule_length = perfection_rule_length
        self.dynamic_match_percentage = dynamic_match_percentage
        self.priority_order = self._voting_priority(mode)
        self._input_to_fusion = {}
        self._samples_info = {}
        self._processed_results = []
        self._overall_metrics = None
        self.model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"Using the following parameters, perfection_rule_length: {self.perfection_rule_length}, dynamic_match_percentage: {self.dynamic_match_percentage}, priority_order: {self.priority_order}")

    def _voting_priority(self, mode):
        if mode.lower() == "conservative":
            priority_order = ["<KEEP>", "<REPLACE>", "<DELETE>", "<SKIP>", "<INSERT>"]
        elif mode.lower() == "aggressive":
            priority_order = ["<REPLACE>", "<KEEP>", "<INSERT>", "<DELETE>", "<SKIP>"]
        # reference
        elif mode.lower() == "reference":
            priority_order = ["<KEEP>", "<SKIP>", "<REPLACE>", "<INSERT>", "<DELETE>"]
        else:
            raise ValueError("mode should be one of the following: [conservative, aggressive, reference]")
        return priority_order

    def _initialize_llm(self):
        self.REINFORCER = reinforcer.FusionReinforcer()
    
    @staticmethod
    def compute_weights(accuracies):
        """
        Compute model weights based on inverse error scores.
        
        Args:
            errors (list): List of error scores for each model
            
        Returns:
            np.array: Normalized weights (higher weight for lower error)
        """
        accuracies = np.array(accuracies)
        weights = np.maximum(accuracies, 1e-10)
        weights = weights / weights.sum()
        return weights
        
    def combine_models_transcriptions(self, *samples_dicts, missing_value=""):
        """
        Align multiple samples_info dicts into an audio-centric structure.

        Args:
            *samples_dicts: Each is a dict like {audio_path: {normalized_prediction: str}}
            missing_value (str): Placeholder when a dict has no prediction for a sample.

        Returns:
            dict: {audio_path: [transcript_from_model1, transcript_from_model2, ...]}
        """
        if not samples_dicts:
            return {}

        # Collect all unique audio paths
        all_audio_paths = sorted({path for d in samples_dicts for path in d.keys()})
        # Build combined dict
        combined = {}
        for audio_path in all_audio_paths:
            combined[audio_path] = []
            for d in samples_dicts:
                if audio_path in d:
                    transcript = d[audio_path].get("normalized_prediction", missing_value)
                else:
                    transcript = missing_value
                combined[audio_path].append(transcript)

        self._input_to_fusion = combined

    def filter_transcriptions_hdbscan(self, transcriptions):
        """
        Filters transcriptions using HDBSCAN density-based clustering.
        No tuning required - automatically detects outliers.
        """
        import hdbscan
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        num_transcriptions = len(transcriptions)
        
        if num_transcriptions <= 2:
            metadata = {
                "status": "Skipped",
                "reason": "Not enough items to cluster (<= 2)",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
            return transcriptions, metadata
        
        # Embed transcriptions using multilingual model (handles Arabic)
        embeddings = self.model.encode(transcriptions)
        
        # HDBSCAN with auto-tuned min_cluster_size
        min_cluster_size = max(2, num_transcriptions // 3)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)
        
        # If all marked as noise, return all
        if np.all(labels == -1):
            metadata = {
                "status": "Skipped",
                "reason": "All transcriptions marked as noise",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
            return transcriptions, metadata
        
        # Keep largest cluster
        unique_labels = np.unique(labels[labels != -1])
        if len(unique_labels) == 0:
            metadata = {
                "status": "Skipped",
                "reason": "No valid clusters found",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
            return transcriptions, metadata
        
        label_counts = [(label, np.sum(labels == label)) for label in unique_labels]
        largest_cluster_label = max(label_counts, key=lambda x: x[1])[0]
        
        kept_indices = np.where(labels == largest_cluster_label)[0]
        filtered_indices = np.where(labels != largest_cluster_label)[0]
        
        filtered_transcriptions = [transcriptions[i] for i in kept_indices]
        
        metadata = {
            "status": "Success",
            "cluster_labels": labels,
            "consensus_cluster_label": largest_cluster_label,
            "kept_indices": kept_indices,
            "filtered_indices": filtered_indices,
            "outlier_probabilities": clusterer.outlier_scores_,
            "min_cluster_size_used": min_cluster_size
        }
        
        return filtered_transcriptions, metadata

    def filter_transcriptions_embeddings_0(self, transcriptions: List[str], 
                                     variance_threshold: float = 1e-5):
        """
        Filters redundant or dissimilar transcriptions using embedding similarity and hierarchical clustering.
        
        Steps:
        1. Encode using SentenceTransformer (GPU if available)
        2. Remove near-constant embeddings (variance filter)
        3. Compute cosine similarity + convert to distance
        4. Perform hierarchical clustering with dynamic threshold
        5. Keep largest cluster
        """

        if len(transcriptions) <= 2:
            return transcriptions, {
                "status": "Skipped",
                "reason": "Not enough items to cluster (<= 2)",
                "kept_indices": list(range(len(transcriptions))),
                "filtered_indices": []
            }

        # --- Embedding ---
        embeddings = self.model.encode(transcriptions, convert_to_tensor=False, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # --- Variance Filter ---
        var_per_dim = np.var(embeddings, axis=0)
        low_var_mask = var_per_dim > variance_threshold
        if np.sum(low_var_mask) == 0:
            return transcriptions, {
                "status": "Skipped",
                "reason": "All embedding dimensions below variance threshold",
                "kept_indices": list(range(len(transcriptions))),
                "filtered_indices": []
            }
        embeddings = embeddings[:, low_var_mask]

        # --- Similarity and Distance ---
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1 - sim_matrix
        dist_matrix = (dist_matrix + dist_matrix.T) / 2        # enforce symmetry
        np.fill_diagonal(dist_matrix, 0)                       # enforce zero self-distance
        dist_matrix = np.where(dist_matrix < 0, 0, dist_matrix) 
        # dist_matrix = 1 - sim_matrix
        if np.all(dist_matrix == 0):
            return transcriptions, {
                "status": "Skipped",
                "reason": "All embeddings identical",
                "kept_indices": list(range(len(transcriptions))),
                "filtered_indices": []
            }

        # --- Hierarchical Clustering ---
        dist_condensed = squareform(dist_matrix, checks=False)
        Z = linkage(dist_condensed, method='average', metric='cosine')

        merge_distances = Z[:, 2]
        jumps = np.diff(merge_distances)
        if len(jumps) == 0:
            return transcriptions, {
                "status": "Skipped",
                "reason": "Single merge, no jump",
                "kept_indices": list(range(len(transcriptions))),
                "filtered_indices": []
            }

        jump_index = np.argmax(jumps)
        t = (merge_distances[jump_index] + merge_distances[jump_index + 1]) / 2
        labels = fcluster(Z, t=t, criterion='distance')

        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        kept_indices = np.where(labels == largest_label)[0]
        filtered_indices = np.where(labels != largest_label)[0]

        filtered_transcriptions = [transcriptions[i] for i in kept_indices]

        return filtered_transcriptions, {
            "status": "Success",
            "embedding_dim": embeddings.shape[1],
            # "variance_filtered_dims": np.sum(low_var_mask),
            "dynamic_threshold_t": t,
            "kept_indices": kept_indices.tolist(),
            "filtered_indices": filtered_indices.tolist()
        }    

    def filter_transcriptions_1(self, transcriptions):
        def _calculate_similarity_matrix(transcriptions: List[str]) -> pd.DataFrame:
            """
            Creates an N*N pairwise similarity matrix for all transcriptions.
            
            Args:
                transcriptions (List[str]): ['transcription_text_1', 'transcription_text_2', ...]
                
            Returns:
                pd.DataFrame: An N*N matrix where cell (i, j) is the Levenshtein
                            ratio (0-100) between transcription i and transcription j.
            """
            def word_sequence_ratio(s1: str, s2: str) -> float:
                """
                Calculates the similarity ratio (0-100) based on matching *words*,
                not characters.
                """
                words1 = s1.split()
                words2 = s2.split()
                if not words1 and not words2: return 100.0
                if not words1 or not words2: return 0.0
                matcher = difflib.SequenceMatcher(None, words1, words2)
                return matcher.ratio() * 100.0
            
            num_systems = len(transcriptions)
            df_similarity = pd.DataFrame(index=range(num_systems), columns=range(num_systems), dtype=float)
            for i in range(num_systems):
                for j in range(i, num_systems):
                    if i == j:
                        similarity = 100.0
                    else:
                        hyp_i = transcriptions[i]
                        hyp_j = transcriptions[j]
                        similarity = word_sequence_ratio(hyp_i, hyp_j)
                    
                    df_similarity.loc[i, j] = similarity
                    df_similarity.loc[j, i] = similarity
                    
            return df_similarity
        
        num_transcriptions = len(transcriptions)
        if num_transcriptions <= 2:
            metadata = {
                "status": "Skipped",
                "reason": "Not enough items to cluster (<= 2)",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
            return transcriptions, metadata
        
        similarity_df = _calculate_similarity_matrix(transcriptions)
        distance_matrix = 100 - similarity_df.values
        if np.all(distance_matrix == 0):
            metadata = {
                "status": "Skipped",
                "reason": "All items are identical",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
            return transcriptions, metadata
        
        distance_condensed = squareform(distance_matrix, checks=False)
        Z = linkage(distance_condensed, method='average')
        merge_distances = Z[:, 2]
        jumps = np.diff(merge_distances)
        
        # Find the index of the largest jump
        # (We add 1 because np.diff returns an array 1 shorter)
        if len(jumps) == 0:
            # Only one merge, cannot find a jump, return original
            metadata = {
                "status": "Skipped",
                "reason": "Only one merge, cannot find a jump",
                "kept_indices": list(range(num_transcriptions)),
                "filtered_indices": []
            }
            return transcriptions, metadata
            
        jump_index = np.argmax(jumps)
        last_good_distance = merge_distances[jump_index]
        first_bad_distance = merge_distances[jump_index + 1]
        dynamic_t = (last_good_distance + first_bad_distance) / 2
        labels = fcluster(Z, t=dynamic_t, criterion='distance')
        unique_labels, counts = np.unique(labels, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        kept_indices = np.where(labels == largest_cluster_label)[0]
        filtered_indices = np.where(labels != largest_cluster_label)[0]
        cluster_indices = np.where(labels == largest_cluster_label)[0]
        filtered_transcriptions = [transcriptions[i] for i in cluster_indices]
        filtration_metadata = {
            "status": "Success",
            "similarity_matrix_df": similarity_df.to_dict(orient="records"),
            "linkage_matrix_Z": Z,
            "merge_distances": merge_distances,
            "distance_jumps": jumps,
            "dynamic_threshold_t": dynamic_t,
            "cluster_labels": labels,
            "consensus_cluster_label": largest_cluster_label,
            "kept_indices": kept_indices,
            "filtered_indices": filtered_indices
        }
        return filtered_transcriptions, filtration_metadata

    def _calculate_intracluster_distance(self, cluster_indices, distance_matrix):
        """
        Helper function to find the average distance *within* a cluster.
        A low score means the cluster is very "tight".
        """
        # If cluster has < 2 members, its internal distance is 0
        if len(cluster_indices) < 2:
            return 0.0
            
        # Extract the sub-matrix for just this cluster's members
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        
        # Get the upper triangle of the matrix (to avoid self-distances and duplicates)
        # k=1 starts above the main diagonal
        distances_list = cluster_distances[np.triu_indices(len(cluster_indices), k=1)]
        
        # Return the average distance
        return np.mean(distances_list)

    def filter_transcriptions_3(self, transcriptions: List[str]) -> List[str]:
        """
        Filters out outlier transcriptions using K-Means (k=2) after
        converting the distance matrix to 2D coordinates using MDS.
        
        It then selects the "tightest" cluster (lowest avg. internal distance).
        """
        def _calculate_similarity_matrix(transcriptions: List[str]) -> pd.DataFrame:
            """
            Creates an N*N pairwise similarity matrix for all transcriptions.
            
            Args:
                transcriptions (List[str]): ['transcription_text_1', 'transcription_text_2', ...]
                
            Returns:
                pd.DataFrame: An N*N matrix where cell (i, j) is the Levenshtein
                            ratio (0-100) between transcription i and transcription j.
            """
            def word_sequence_ratio(s1: str, s2: str) -> float:
                """
                Calculates the similarity ratio (0-100) based on matching *words*,
                not characters.
                """
                words1 = s1.split()
                words2 = s2.split()
                if not words1 and not words2: return 100.0
                if not words1 or not words2: return 0.0
                matcher = difflib.SequenceMatcher(None, words1, words2)
                return matcher.ratio() * 100.0
            
            num_systems = len(transcriptions)
            df_similarity = pd.DataFrame(index=range(num_systems), columns=range(num_systems), dtype=float)
            for i in range(num_systems):
                for j in range(i, num_systems):
                    if i == j:
                        similarity = 100.0
                    else:
                        hyp_i = transcriptions[i]
                        hyp_j = transcriptions[j]
                        similarity = word_sequence_ratio(hyp_i, hyp_j)
                    
                    df_similarity.loc[i, j] = similarity
                    df_similarity.loc[j, i] = similarity
                    
            return df_similarity
    
        num_transcriptions = len(transcriptions)
        
        # K-Means with k=2 is meaningless for 2 or fewer items.
        if num_transcriptions <= 2:
            return transcriptions
            
        # 1. Get the Word-Distance Matrix (Same as your other method)
        df_similarity = _calculate_similarity_matrix(transcriptions)
        distance_matrix = 100 - df_similarity.to_numpy()

        # 2. CONVERT DISTANCE MATRIX TO COORDINATES (The MDS step)
        # We convert the N*N matrix into N points in 2D space
        mds = MDS(
            n_components=2,          # Convert to 2D points
            dissimilarity='precomputed',
            random_state=42,         # For reproducible results
            n_init=4,                # Run 4 times, pick best
            normalized_stress=False
        )
        print(distance_matrix)
        # coordinates is now an [N, 2] array
        coordinates = mds.fit_transform(distance_matrix)
        print(coordinates)
        # 3. RUN K-MEANS (k=2)
        kmeans = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=10 # Run 10 times, pick best
        )
        labels = kmeans.fit_predict(coordinates) # e.g., [0, 0, 0, 0, 1, 1]
        print(labels)
        # 4. IDENTIFY THE BEST CLUSTER (Your logic)
        cluster_0_indices = np.where(labels == 0)[0]
        cluster_1_indices = np.where(labels == 1)[0]
        
        # Handle edge case where a cluster is empty
        if len(cluster_0_indices) == 0:
            return [transcriptions[i] for i in cluster_1_indices]
        if len(cluster_1_indices) == 0:
            return [transcriptions[i] for i in cluster_0_indices]

        # Calculate average distance *within* each cluster
        dist_0 = self._calculate_intracluster_distance(cluster_0_indices, distance_matrix)
        dist_1 = self._calculate_intracluster_distance(cluster_1_indices, distance_matrix)
        print(dist_0)
        print(dist_1)
        # 5. Pick the cluster with the LOWEST internal distance (the "tightest" one)
        if dist_0 <= dist_1:
            best_indices = cluster_0_indices
        else:
            best_indices = cluster_1_indices
            
        # 6. Return the sentences from the best cluster
        filtered_transcriptions = [
            transcriptions[i] for i in best_indices
        ]

        return filtered_transcriptions

    def filter_transcriptions_2(self, transcriptions: List[str]):
        """
        Filters transcriptions using Hierarchical Clustering with 'ward'
        after converting text to TF-IDF vectors.
        """
        
        num_transcriptions = len(transcriptions)
        
        if num_transcriptions <= 1:
            return transcriptions

        # 1. Convert Sentences to Numerical Vectors (TF-IDF)
        # This replaces _calculate_similarity_matrix
        vectorizer = TfidfVectorizer()
        
        # .fit_transform() learns the vocabulary and converts the list
        # of strings into a sparse matrix of vectors.
        tfidf_vectors = vectorizer.fit_transform(transcriptions)
        # 2. Run linkage directly on the vectors
        # We replace distance_condensed with the vector matrix.
        # We specify method='ward' and metric='euclidean'.
        # .toarray() is needed because linkage doesn't like sparse matrices.
        Z = linkage(tfidf_vectors.toarray(), method='ward', metric='euclidean')
        print(Z)
        # 3. Find clusters (This part is the same as your code)
        # You may need to adjust 't' (1.15) as it's now
        # acting on different 'inconsistency' values.
        labels = fcluster(Z, t=70.0, criterion='distance')
        print(labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Handle case where all are outliers
        if len(counts) == 0:
            return []
            
        largest_cluster_label = unique_labels[np.argmax(counts)]
        print(largest_cluster_label)
        # 4. Filter and return (This is the same)
        cluster_indices = np.where(labels == largest_cluster_label)[0]
        print(cluster_indices)
        filtered_transcriptions = [
            transcriptions[i] for i in cluster_indices
        ]

        return filtered_transcriptions

    def get_reference_from_transcriptions(self, transcriptions):
        def get_longest_reference(transcriptions):
            def validate_anchor_quality(reference, transcription):
                """
                Validate that anchors represent meaningful common structure using SequenceMatcher
                
                Returns:
                    tuple: (is_valid, anchor_info)
                """
                def find_first_anchors(reference_sentence, compared_sentence):
                    ref_words = reference_sentence.split()
                    comp_words = compared_sentence.split()
                    for i, ref_word in enumerate(ref_words):
                        for j, comp_word in enumerate(comp_words):
                            if ref_word == comp_word:
                                return (i, j)
                    return None
                
                anchors = find_first_anchors(reference, transcription)
                if not anchors: return False, None
                    
                ref_pos, trans_pos = anchors
                ref_words = reference.split()
                trans_words = transcription.split()
                ref_remaining = ref_words[ref_pos:]
                trans_remaining = trans_words[trans_pos:]
                # Use SequenceMatcher on the remaining sequences for better validation
                matcher = SequenceMatcher(None, ref_remaining, trans_remaining)
                # Get the longest matching block starting from position 0 (right after anchor)
                longest_match = matcher.find_longest_match(0, len(ref_remaining), 0, len(trans_remaining))
                additional_matches = longest_match.size
                shorter_sequence_length = min(len(ref_remaining), len(trans_remaining))
                is_valid = False
                min_required = 0
                if shorter_sequence_length <= self.perfection_rule_length:
                    # Check if the number of matching words is exactly equal to the length of the shorter sequence.
                    min_required = shorter_sequence_length
                    if additional_matches == shorter_sequence_length:
                        is_valid = True
                else:
                    min_required = int(shorter_sequence_length * self.dynamic_match_percentage)
                    # Check if we found at least our minimum required number of matches.
                    if additional_matches >= min_required:
                        is_valid = True

                return is_valid, {
                    "positions": anchors, 
                    "additional_matches": additional_matches,
                    "min_required": min_required
                }
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: len(x[1]))
            successful_alignments = 0
            total_comparisons = len(valid_transcriptions) - 1
            for original_index, transcription in valid_transcriptions:
                if original_index != longest_original_index:
                    is_valid, _ = validate_anchor_quality(longest_reference, transcription)
                    if is_valid:
                        successful_alignments += 1
            
            success_ratio = successful_alignments / total_comparisons if total_comparisons > 0 else 1.0
            success = success_ratio >= 0.5
            metadata = {
                'strategy_metric': 'longest_match_validation',
                'score': success_ratio,
                'successful_alignments': successful_alignments,
                'total_comparisons': total_comparisons,
                'reference_index': longest_original_index
            }
            return success, longest_reference, metadata
        
        def get_common_words_reference(transcriptions):
            """
            Reference selection using SequenceMatcher similarity ratios
            """
            def calculate_similarity(sentence1, sentence2):
                """Calculate similarity ratio between two sentences using SequenceMatcher"""
                words1, words2 = sentence1.split(), sentence2.split()
                matcher = SequenceMatcher(None, words1, words2)
                return matcher.ratio()
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None

            best_total_similarity = -1
            best_reference, best_index = None, None
            for original_i, transcription_i in valid_transcriptions:
                total_similarity = 0.0
                for original_j, transcription_j in valid_transcriptions:
                    if original_i != original_j:
                        similarity = calculate_similarity(transcription_i, transcription_j)
                        total_similarity += similarity
                
                if total_similarity > best_total_similarity:
                    best_total_similarity = total_similarity
                    best_reference = transcription_i
                    best_index = original_i
            
            if best_reference and best_total_similarity >= 0:
                num_comparisons = len(valid_transcriptions) - 1 if len(valid_transcriptions) > 1 else 1
                avg_similarity = best_total_similarity / num_comparisons if num_comparisons > 0 else 0.0
                metadata = {
                    'strategy_metric': 'most_common_words',
                    'score': avg_similarity,
                    'total_similarity_score': best_total_similarity,
                    'reference_index': best_index
                }
                return True, best_reference, metadata
            
            return False, None, None

        def get_longest_reference_fallback(transcriptions):
            """
            Fallback when anchor validation fails but we still need a reference.
            Uses average pairwise similarity to find most representative transcription.
            """
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            # Single transcription case
            if len(valid_transcriptions) == 1:
                return True, valid_transcriptions[0][1], valid_transcriptions[0][0]
            
            # Calculate average word-level edit distance for each transcription
            best_avg_similarity = -1
            best_reference = None
            best_index = None
            
            for original_i, transcription_i in valid_transcriptions:
                words_i = transcription_i.split()
                total_similarity = 0.0
                comparisons = 0
                
                for original_j, transcription_j in valid_transcriptions:
                    if original_i != original_j:
                        words_j = transcription_j.split()
                        # Use SequenceMatcher for word-level similarity
                        matcher = SequenceMatcher(None, words_i, words_j)
                        total_similarity += matcher.ratio()
                        comparisons += 1
                
                avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
                
                # Tiebreaker: prefer longer transcription when similarity is equal
                if avg_similarity > best_avg_similarity or \
                (avg_similarity == best_avg_similarity and len(transcription_i) > len(best_reference or "")):
                    best_avg_similarity = avg_similarity
                    best_reference = transcription_i
                    best_index = original_i
            
            # Success if we found reasonable consensus (>30% similarity)
            success = best_avg_similarity >= 0.2
            metadata = {
                'strategy_metric': 'pairwise_similarity',
                'score': best_avg_similarity,
                'reference_index': best_index
            }
            return success, best_reference, metadata
        
        # Always ensure we have at least one valid transcription to return
        valid_transcriptions = [t for t in transcriptions if t is not None]
        if not valid_transcriptions:
            return None, None, None
        
        longest_success, longest_reference, reference_metadata = get_longest_reference(transcriptions)
        if longest_success: return "longest", longest_reference, reference_metadata

        common_words_success, common_words_reference, reference_metadata = get_common_words_reference(transcriptions)
        if common_words_success: return "common_words", common_words_reference, reference_metadata

        fallback_success, fallback_reference, reference_metadata = get_longest_reference_fallback(transcriptions)
        if fallback_success: return "longest_fallback", fallback_reference, reference_metadata
        else: return "failed", fallback_reference , reference_metadata

    def align_transcriptions_to_reference(self, reference, reference_type, reference_index, transcriptions):
        """
        Simple alignment function that extracts operations and candidate values.
        
        Args:
            reference (str): Reference transcription
            transcription (str): Candidate transcription to align
            
        Returns:
            tuple: (operations, candidate_values)
        """
        # Wont change
        def align_with_longest_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns a list of transcriptions against a single authoritative reference.

            This function uses the provided 'reference' sentence as a structural backbone.
            It iterates through each transcription in the list and compares it to the
            reference using Python's SequenceMatcher. For each position in the
            reference, it determines an operation ('<KEEP>', '<REPLACE>', '<DELETE>')
            and the corresponding token from the transcription.

            The key characteristic of this strategy is that the output structure is
            always dictated by the length of the reference. Insertions in the
            candidate transcriptions are ignored in the final alignment to preserve this
            structure, ensuring all output lists have the same length.

            Args:
                reference (str): The authoritative transcription to align against.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence within the
                    transcriptions list.
                reference_type (str): A string descriptor for the strategy used
                    (e.g., "longest").

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription. Each dictionary contains:
                            - 'model_index' (int): The original index of the model.
                            - 'reference_type' (str): The strategy name.
                            - 'is_reference' (bool): A flag indicating if this was
                            the reference transcription.
                            - 'operations' (list[str]): A list of operations
                            relative to the reference.
                            - 'tokens' (list[str or None]): A list of words (or None)
                            aligned to the reference positions.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # --- PHASE 1: SURVEY AND PAD THE REFERENCE ---
            # Find all unique insertion points from other transcriptions
            insertion_map = {}
            for model_idx, transcription in enumerate(transcriptions):
                if model_idx == reference_index or not transcription: 
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, _, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])

            # Build the new, flexible "padded" reference blueprint
            padded_ref_words = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    # Add placeholders for unique words other models inserted
                    padded_ref_words.extend([None] * len(set(insertion_map[i])))
                padded_ref_words.append(word)
            
            # Handle insertions that occur after the last word of the reference
            if len(ref_words) in insertion_map:
                padded_ref_words.extend([None] * len(set(insertion_map[len(ref_words)])))
                
            padded_ref_length = len(padded_ref_words)

            # --- PHASE 2: ALIGN ALL TRANSCRIPTIONS TO THE PADDED REFERENCE ---
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                # Align the original reference model to the new padded structure
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words[:]
                
                # Align all other models to the new padded structure
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            # 'insert' is now handled implicitly by the padded structure,
                            # so a 'pass' is safe here as a fallback.
                            elif op == 'insert':
                                pass
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata

        def align_with_common_words_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns transcriptions using a flexible, consensus-based reference.

            This strategy is designed for cases where no single transcription is a clear
            structural authority. It works in two main phases:

            1.  **Reference Padding:** It first surveys all other transcriptions to find
                words they contain that are missing from the reference ('insertions').
                It then creates a new "padded" reference by inserting `None`
                placeholders at the appropriate positions to create slots for these
                potential new words.

            2.  **Final Alignment:** It then aligns all transcriptions against this new,
                longer, padded reference. This allows words that would have been
                insertions to be properly mapped to a slot, enabling a vote on whether
                they should be included in the final output.

            The purpose is to create a flexible alignment structure that accommodates
            structural differences between models, rather than strictly enforcing the
            structure of one reference.

            Args:
                reference (str): The consensus-based transcription to use as a starting point.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence.
                reference_type (str): A string descriptor for the strategy.

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription, all aligned to the padded reference length.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # Pad reference by finding all unique insertion points from other transcriptions
            padded_ref_words = ref_words.copy()
            insertion_map = {}  # Maps ref position -> list of words inserted before it
            for model_index, transcription in enumerate(transcriptions):
                if model_index == reference_index or transcription is None:
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        # Words in transcription that don't exist in reference
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])
            
            # Build padded reference with None placeholders for insertions
            final_ref = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    final_ref.extend([None] * len(set(insertion_map[i])))  # Unique insertions only
                final_ref.append(word)
            if len(ref_words) in insertion_map:  # Trailing insertions
                final_ref.extend([None] * len(set(insertion_map[len(ref_words)])))
            
            padded_ref_words = final_ref
            padded_ref_length = len(padded_ref_words)
            
            # Align each transcription to padded reference
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words.copy()
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            elif op == 'insert':
                                pass  # Insertions don't map to reference positions
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata
        
        def align_with_longest_fallback_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Fallback alignment when anchor validation fails.
            
            Flow:
            1. Compute pairwise similarity matrix for all valid transcriptions
            2. Calculate reliability weight for each model (average similarity to others)
            3. Perform standard word-level alignment (same as longest strategy)
            4. Attach weight metadata to each alignment result for downstream use
            
            Difference from longest strategy:
            - Longest: High anchor confidence, reference is structural authority
            - Fallback: Low anchor confidence, reference is "best guess", weights signal reliability
            
            Output: Same structure as longest strategy + 'weight' field per model
            Purpose: Enable weighted voting in consensus step when reference quality is uncertain
            """
            alignment_results = []
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t and t.strip()]
            
            if len(valid_transcriptions) <= 1:
                # Degenerate case: use simple alignment without weights
                return align_with_longest_strategy(reference, transcriptions, reference_index, reference_type)
            
            # Calculate pairwise similarities
            model_weights = {}
            for i, trans_i in valid_transcriptions:
                words_i = trans_i.split()
                total_sim = 0.0
                comparisons = 0
                for j, trans_j in valid_transcriptions:
                    if i != j:
                        words_j = trans_j.split()
                        total_sim += SequenceMatcher(None, words_i, words_j).ratio()
                        comparisons += 1
                model_weights[i] = total_sim / comparisons if comparisons > 0 else 0.0
            
            # Normalize weights to [0, 1]
            max_weight = max(model_weights.values()) if model_weights else 1.0
            if max_weight > 0:
                model_weights = {k: v / max_weight for k, v in model_weights.items()}
            
            # Perform standard alignment
            ref_words = reference.split()
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index),
                    'weight': model_weights.get(model_index, 0.0)  # Reliability score
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>"] * len(ref_words)
                    alignment_result['tokens'] = ref_words.copy()
                else:
                    if not transcription or not transcription.strip():
                        alignment_result['operations'] = ["<DELETE>"] * len(ref_words)
                        alignment_result['tokens'] = [None] * len(ref_words)
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * len(ref_words)
                        candidate_values = [None] * len(ref_words)
                        
                        matcher = SequenceMatcher(None, ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    candidate_values[i] = ref_words[i]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        candidate_values[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = candidate_values
                
                alignment_results.append(alignment_result)
            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': ref_words.copy(), 
                'model_weights': model_weights # This is the key metadata for this strategy
            }
            return alignment_results, metadata
        
        if reference is None or not transcriptions:
            return []
        if reference_type == "longest":
            alignment_results, alignment_metadata = align_with_longest_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "common_words":
            alignment_results, alignment_metadata = align_with_common_words_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "longest_fallback":
            alignment_results, alignment_metadata = align_with_longest_fallback_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "failed":
            alignment_results, alignment_metadata = align_with_longest_fallback_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        return alignment_results, alignment_metadata
    
    def voting_scheme(self, alignment_results):
        """
        Implement majority voting scheme for operations and random voting for tokens.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with operations, tokens, and metadata
        """
        def collect_position_votes(alignment_results, position):
            """Collect all votes for a specific position"""
            position_votes = {
                'operations': [],
                'tokens': [],
                'model_indices': [],
                'is_reference_flags': []
            }
            
            for result in alignment_results:
                position_votes['operations'].append(result['operations'][position])
                position_votes['tokens'].append(result['tokens'][position])
                position_votes['model_indices'].append(result['model_index'])
                position_votes['is_reference_flags'].append(result['is_reference'])
            
            return position_votes
        
        def vote_for_operation(position_votes):
            """Determine majority operation with tie-breaking"""
            operation_counts = Counter(position_votes['operations'])
            majority_operation = operation_counts.most_common(1)[0][0]
            
            # Handle ties with priority order
            max_count = operation_counts.most_common(1)[0][1]
            tied_operations = [op for op, count in operation_counts.items() if count == max_count]
            if len(tied_operations) > 1:
                for preferred_op in self.priority_order:
                    if preferred_op in tied_operations:
                        majority_operation = preferred_op
                        break
            
            return majority_operation, operation_counts
        
        def vote_for_token(position_votes, majority_operation):
            """Determine final token based on operation with random selection"""
            def token_similarity_tiebreaker(tied_tokens, all_candidate_tokens):
                def stable_hash(s):
                    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
                
                """Select token with highest character overlap across all candidates"""
                if len(tied_tokens) == 1:
                    return tied_tokens[0]
                
                scores = {}
                for candidate in tied_tokens:
                    candidate_chars = set(candidate)
                    total_overlap = sum(
                        len(candidate_chars & set(other)) 
                        for other in all_candidate_tokens if other is not None and other != candidate
                    )
                    scores[candidate] = total_overlap
                
                max_score = max(scores.values())
                best_tokens = [t for t, s in scores.items() if s == max_score]
                
                # return random.choice(best_tokens)  # Random choice
                return min(best_tokens, key=lambda t: (len(t), stable_hash(t)))  # Length-first hybrid

            def vote_by_edit_distance(tied_tokens, all_tokens):
                scores = {}
                for candidate in tied_tokens:
                    total_distance = sum(
                        distance(candidate, other) 
                        for other in all_tokens if other and other != candidate
                    )
                    scores[candidate] = total_distance
                
                min_distance = min(scores.values())
                best_tokens = [t for t, d in scores.items() if d == min_distance]
                
                if len(best_tokens) > 1:
                    return token_similarity_tiebreaker(best_tokens, all_tokens)
                return best_tokens[0]
                        
            final_token = None
            if majority_operation == "<KEEP>":
                for i, is_ref in enumerate(position_votes['is_reference_flags']):
                    if is_ref:
                        final_token = position_votes['tokens'][i]
                        break
                if final_token is None:
                    final_token = position_votes['tokens'][0]
                                
            elif majority_operation in ["<REPLACE>", "<INSERT>"]:
                operation_tokens = [
                    position_votes['tokens'][i]
                    for i, op in enumerate(position_votes['operations'])
                    if op == majority_operation and position_votes['tokens'][i] is not None
                ]
                
                if operation_tokens:
                    counts = Counter(operation_tokens)
                    max_count = counts.most_common(1)[0][1]
                    tied_tokens = [t for t, c in counts.items() if c == max_count]
                    if len(tied_tokens) > 1:
                        all_tokens = position_votes['tokens']
                        final_token = vote_by_edit_distance(tied_tokens, all_tokens)
                    else:
                        final_token = tied_tokens[0]
                    
            elif majority_operation in ["<DELETE>", "<SKIP>"]:
                final_token = None
            
            return final_token
        
        def create_voting_detail(position, operation_counts, majority_operation, final_token, position_votes):
            """
            Create detailed voting information for a position
            """
            token_weights = {}
            if majority_operation in ["<REPLACE>", "<INSERT>"]:
                operation_tokens = [
                    position_votes['tokens'][i]
                    for i, op in enumerate(position_votes['operations'])
                    if op == majority_operation and position_votes['tokens'][i] is not None
                ]
                token_weights = {token: operation_tokens.count(token) for token in set(operation_tokens)}
            
            return {
                'position': position,
                'operation_votes': dict(operation_counts),
                'majority_operation': majority_operation,
                'final_token': final_token,
                'models_voted': len(position_votes['operations']),
                'token_weights': token_weights  # Now shows counts instead of weights
            }
        
        def construct_final_transcription(fusion_operations, fusion_tokens):
            """Build the final transcription from operations and tokens"""
            final_transcription_words = []
            for operation, token in zip(fusion_operations, fusion_tokens):
                if operation in ["<KEEP>", "<REPLACE>", "<INSERT>"] and token is not None:
                    final_transcription_words.append(token)
            
            return " ".join(final_transcription_words)
        
        def calculate_confidence_score(alignment_results, fusion_operations, fusion_tokens):
            total_positions = len(fusion_operations)
            if total_positions == 0:
                return 0
            
            position_confidences = []
            for i in range(total_positions):
                ops = [r['operations'][i] for r in alignment_results]
                tokens = [r['tokens'][i] for r in alignment_results]
                
                op_agreement = ops.count(fusion_operations[i]) / len(ops)
                
                if fusion_operations[i] in ["<REPLACE>", "<INSERT>"] and fusion_tokens[i] is not None:
                    token_agreement = tokens.count(fusion_tokens[i]) / len(tokens)
                    position_confidences.append((op_agreement + token_agreement) / 2)
                else:
                    position_confidences.append(op_agreement)
            
            return sum(position_confidences) / total_positions
        
        
        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_operations": [], "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
                
        # Main voting logic
        operations_length = len(alignment_results[0]['operations'])
        fusion_operations = []
        fusion_tokens = []
        voting_details = []
        # Process each position
        for position in range(operations_length):
            position_votes = collect_position_votes(alignment_results, position)
            majority_operation, operation_counts = vote_for_operation(position_votes)
            final_token = vote_for_token(position_votes, majority_operation)
            
            fusion_operations.append(majority_operation)
            fusion_tokens.append(final_token)
            
            voting_detail = create_voting_detail(position, operation_counts, majority_operation, final_token, position_votes)
            voting_details.append(voting_detail)
        
        # Construct final results
        fusion_transcript = construct_final_transcription(fusion_operations, fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_operations, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        voting_metadata = {
            'reference_type': alignment_results[0]['reference_type'],
            'confidence_score': confidence_score,
            'total_models': len(alignment_results),
            'operations_length': operations_length,
            'operation_counts': {
                'total_keep': fusion_operations.count('<KEEP>'),
                'total_replace': fusion_operations.count('<REPLACE>'),
                'total_insert': fusion_operations.count('<INSERT>'),
                'total_delete': fusion_operations.count('<DELETE>'),
                'total_skip': fusion_operations.count('<SKIP>')
            },
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_operations": fusion_operations,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }
    
    def llm_reinforcer(self, fusion_tokens, candidates_tokens, max_tokens, chunk_size, overlap):
        def postprocess_reinforced_output(response: reinforcer.GeneratedResponse) -> tuple:
            chunks = response.reinforced_results or []
            modifications = sum(1 for chunk in chunks if chunk.get("is_modified"))
            transcript_pieces = []
            for chunk in chunks:
                if chunk.get("is_modified") and chunk.get("sentence"):
                    transcript_pieces.append(chunk["sentence"])
                else:
                    original_sentence = chunk.get("metadata", {}).get("fusion_sentence", "")
                    transcript_pieces.append(original_sentence)

            unwanted_values = [None, "None", "Null", "null", ""]
            filtered_pieces = [piece for piece in transcript_pieces if piece not in unwanted_values]
            llm_transcript = " ".join(filtered_pieces).strip()
            llm_metadata = {
                "chunks": chunks,
                "num_chunks": len(chunks),
                "modifications": modifications,
                "modification_ratio": modifications / len(chunks) if chunks else 0.0,
                "is_chunked": response.is_chunked,
            }
            
            return llm_transcript, llm_metadata

        if not self.use_llm:
            # Return an empty transcript and a default metadata object for consistency
            llm_metadata = {
                "llm_time": 0.0,
                "status": "LLM reinforcement was disabled.",
                "num_chunks": 0, "modifications": 0, "modification_ratio": 0.0
            }
            return "", llm_metadata
        
        llm_start = time.time()
        response = self.REINFORCER.main(
            fusion_tokens=fusion_tokens,
            candidate_tokens=candidates_tokens,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        llm_time = time.time() - llm_start
        llm_transcript, llm_metadata = postprocess_reinforced_output(response)
        llm_metadata["llm_time"] = llm_time
        return llm_transcript, llm_metadata    

    def token_voting_scheme(self, alignment_results):
        """
        Implement a token-only majority voting scheme.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """
        
        def collect_position_tokens(alignment_results, position):
            """Collect all tokens for a specific position, including None."""
            tokens = []
            for result in alignment_results:
                tokens.append(result['tokens'][position])
            return tokens

        def vote_for_token(position_tokens):
            """
            Determine the final token by majority vote, with complex tie-breaking.
            
            Args:
                position_tokens: A list of all token candidates at this position
                                 (e.g., ['wordA', 'wordB', 'wordA', None])
                                 
            Returns:
                (final_token, token_counts_dict)
            """
            def token_similarity_tiebreaker(tied_tokens, all_candidate_tokens):
                """Select token with highest character overlap across all candidates."""
                def stable_hash(s):
                    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)
                
                if len(tied_tokens) == 1:
                    return tied_tokens[0]
                
                scores = {}
                for candidate in tied_tokens:
                    if candidate is None: 
                        scores[candidate] = -1 
                        continue
                        
                    candidate_chars = set(candidate)
                    total_overlap = sum(
                        len(candidate_chars & set(other)) 
                        for other in all_candidate_tokens if other is not None and other != candidate
                    )
                    scores[candidate] = total_overlap
                
                max_score = max(scores.values())
                best_tokens = [t for t, s in scores.items() if s == max_score]
                
                return min(best_tokens, key=lambda t: (len(t) if t is not None else float('inf'), stable_hash(t) if t is not None else float('inf')))

            def vote_by_edit_distance(tied_tokens, all_tokens):
                """
                Tie-breaker using Levenshtein distance.
                Selects the token with the minimum total edit distance to all other tokens.
                """
                scores = {}
                for candidate in tied_tokens:
                    if candidate is None:
                        scores[candidate] = float('inf') 
                        continue

                    total_distance = sum(
                        distance(candidate, other) 
                        for other in all_tokens if other and other != candidate
                    )
                    scores[candidate] = total_distance
                
                min_distance = min(scores.values())
                best_tokens = [t for t, d in scores.items() if d == min_distance]
                
                if len(best_tokens) > 1:
                    return token_similarity_tiebreaker(best_tokens, all_tokens)
                return best_tokens[0]

            token_counts = Counter(position_tokens)
            if not token_counts: return None, {} 

            max_count = token_counts.most_common(1)[0][1]
            tied_tokens = [t for t, c in token_counts.items() if c == max_count]
            final_token = None
            if len(tied_tokens) == 1:
                final_token = tied_tokens[0]
            else:
                word_tokens = [t for t in tied_tokens if t is not None]
                if len(word_tokens) == 0:
                    final_token = None
                elif len(word_tokens) == 1:
                    final_token = word_tokens[0]
                else:
                    all_non_none_tokens = [t for t in position_tokens if t is not None]
                    final_token = vote_by_edit_distance(word_tokens, all_non_none_tokens)
            
            return final_token, dict(token_counts)

        def create_voting_detail(position, final_token, token_counts, total_votes):
            """Create detailed voting information for a position."""
            return {
                'position': position,
                'token_votes': token_counts,
                'final_token': final_token,
                'models_voted': total_votes,
            }

        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)

        def calculate_confidence_score(alignment_results, fusion_tokens):
            """Calculate average agreement on the chosen token."""
            total_positions = len(fusion_tokens)
            if total_positions == 0:
                return 0
            
            position_confidences = []
            num_models = len(alignment_results)
            if num_models == 0:
                return 0
                
            for i in range(total_positions):
                tokens_at_pos = [r['tokens'][i] for r in alignment_results]
                agreement_count = tokens_at_pos.count(fusion_tokens[i])
                position_confidences.append(agreement_count / num_models)
            
            return sum(position_confidences) / total_positions

        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
                
        sequence_length = len(alignment_results[0]['tokens'])
        fusion_tokens = []
        voting_details = []
        total_models = len(alignment_results)

        for position in range(sequence_length):
            position_tokens = collect_position_tokens(alignment_results, position)
            final_token, token_counts = vote_for_token(position_tokens)
            
            fusion_tokens.append(final_token)
            
            voting_detail = create_voting_detail(position, final_token, token_counts, total_models)
            voting_details.append(voting_detail)
        
        fusion_transcript = construct_final_transcription(fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'confidence_score': confidence_score,
            'total_models': total_models,
            'sequence_length': sequence_length,
            'selection_method': 'token_majority_vote',
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }
    
    def rover_scheme(self, alignment_results):
        """
        Implement the ROVER (Recognizer Output Voting Error Reduction) voting scheme.
        This method first builds a Word Transition Network (WTN) and then finds the
        best-scoring path through it to generate the fused transcript.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """

        def build_word_transition_network(alignment_results):
            """Create a Word Transition Network from aligned hypotheses."""
            if not alignment_results:
                return []
            
            sequence_length = len(alignment_results[0]['tokens'])
            wtn = []
            for i in range(sequence_length):
                position_tokens = [result['tokens'][i] for result in alignment_results]
                wtn.append(Counter(position_tokens))
            return wtn

        def find_best_path(wtn):
            """Find the best path through the WTN using dynamic programming (Viterbi-like)."""
            if not wtn:
                return []

            scores = [{}]
            backpointers = [{}]

            # Initialize first column
            for word, count in wtn[0].items():
                scores[0][word] = count
                backpointers[0][word] = None

            # Iterate through the rest of the network
            for i in range(1, len(wtn)):
                scores.append({})
                backpointers.append({})
                for word, count in wtn[i].items():
                    max_score = -1
                    best_prev_word = None
                    
                    # Find the previous word that maximizes the path score
                    for prev_word, prev_score in scores[i-1].items():
                        current_score = prev_score + count
                        if current_score > max_score:
                            max_score = current_score
                            best_prev_word = prev_word
                    
                    scores[i][word] = max_score
                    backpointers[i][word] = best_prev_word

            # Trace back from the best final word
            best_path = []
            last_word = max(scores[-1], key=scores[-1].get)
            
            for i in range(len(wtn) - 1, -1, -1):
                best_path.insert(0, last_word)
                last_word = backpointers[i][last_word]
                
            return best_path

        def create_voting_detail(position, final_token, wtn_level):
            """Create detailed voting information for a position."""
            return {
                'position': position,
                'token_votes': dict(wtn_level),
                'final_token': final_token,
                'models_voted': sum(wtn_level.values()),
            }

        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)

        def calculate_confidence_score(alignment_results, fusion_tokens):
            """Calculate average agreement on the chosen token."""
            total_positions = len(fusion_tokens)
            if total_positions == 0: return 0
            
            position_confidences = []
            num_models = len(alignment_results)
            if num_models == 0: return 0
                
            for i in range(total_positions):
                tokens_at_pos = [r['tokens'][i] for r in alignment_results]
                agreement_count = tokens_at_pos.count(fusion_tokens[i])
                position_confidences.append(agreement_count / num_models)
            
            return sum(position_confidences) / total_positions

        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "", "fusion_tokens": [],
                "candidates_tokens": [], "voting_metadata": {"status": "No alignment results"}
            }
                
        wtn = build_word_transition_network(alignment_results)
        fusion_tokens = find_best_path(wtn)

        voting_details = []
        for i, token in enumerate(fusion_tokens):
            voting_details.append(create_voting_detail(i, token, wtn[i]))

        fusion_transcript = construct_final_transcription(fusion_tokens)
        confidence_score = calculate_confidence_score(alignment_results, fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'confidence_score': confidence_score,
            'total_models': len(alignment_results),
            'sequence_length': len(fusion_tokens),
            'selection_method': 'rover',
            'voting_details': voting_details
        }

        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }
    
    def main(self, **kwargs):
        def fuse_sample_transcriptions(audio_path, transcriptions):
            voting_start = time.time()
            filtered_transcriptions, filtration_metadata = self.filter_transcriptions_1(transcriptions)
            reference_type, reference, reference_metadata = self.get_reference_from_transcriptions(filtered_transcriptions)
            alignment_results, alignment_metadata = self.align_transcriptions_to_reference(reference=reference, reference_type=reference_type, 
                                                                                           reference_index=reference_metadata["reference_index"], 
                                                                                           transcriptions=filtered_transcriptions)
            voting_output = self.token_voting_scheme(alignment_results)
            llm_transcript, llm_metadata = self.llm_reinforcer(
                fusion_tokens=voting_output["fusion_tokens"],
                candidates_tokens=voting_output["candidates_tokens"],
                max_tokens=kwargs.get("max_tokens", 25),
                chunk_size=kwargs.get("chunk_size", 15),
                overlap=kwargs.get("overlap", 3),
            )
            data = {
                # --- Primary Outputs ---
                "fusion_transcript": voting_output["fusion_transcript"],
                "llm_transcript": llm_transcript,
                
                # --- Diagnostic/Internal Data ---
                # "fusion_operations": voting_output["fusion_operations"],
                "fusion_tokens": voting_output["fusion_tokens"],
                "candidates_tokens": voting_output["candidates_tokens"],
                "voting_time": time.time() - voting_start,
                
                # --- Aggregated Metadata ---
                "metadata": {
                    "reference_selection": reference_metadata,
                    "alignment": alignment_metadata,
                    "voting": voting_output["voting_metadata"],
                    "llm_reinforcement": llm_metadata  # <-- NEW metadata section
                }
            }
            
            return {audio_path: data}

        def process_item(item):
            key, value = item
            transcriptions_lists = [t for t in value]
            voting_result = fuse_sample_transcriptions(key, transcriptions_lists)
            return voting_result
        
        if not self.input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_item, item): item for item in self.input_to_fusion.items()}
            samples_info = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing Inputs..."):
                result = future.result()  # this is {audio_path: {...}}
                samples_info.update(result)

        self._samples_info = dict(sorted(samples_info.items()))

    def eval(self, audios_chunk):
        def _eval_common(data):
            return data["fusion_transcript"]

        def _eval_llm(data):
            return data["llm_transcript"]

        refs_lookup = {sample["audio_path"]: sample["normalized_transcription"] for sample in audios_chunk}
        all_audio_paths = list(self._samples_info.keys())
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    ref = refs_lookup.get(audio_path)
                    hyp = _eval_llm(self._samples_info[audio_path]) if self.use_llm else _eval_common(self._samples_info[audio_path])
                    self._samples_info[audio_path]["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(hyp, substitute=True)
                    self._samples_info[audio_path]["normalized_transcription"] = ref
                    if ref is not None:
                        norm_hyp = self._samples_info[audio_path]["normalized_prediction"]
                        sample_metrics = metrics.BasicSTTMetrics.evaluate(refs=ref, hyps=norm_hyp)
                        self._samples_info[audio_path]["metrics"] = sample_metrics
                    else:
                        self._samples_info[audio_path]["metrics"] = helpers._empty_metrics()

                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {
                        "normalized_prediction": None,
                        "metrics": helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }

        # Compute overall metrics
        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def summary_of_evaluation(self):
        """
        Display a simple summary of the overall evaluation metrics.
        """
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return

        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")

    def reset(self):
        """
        Reset the EnsembleInferenceRefactored instance.
        Clears input_to_fusion and fusion_results, but keeps the model loaded.
        """
        self._input_to_fusion = {}
        self._fusion_results = []
        self._overall_metrics = None
        LOGGER.info("Ensemble instance has been reset.")

    @property
    def input_to_fusion(self):
        """
        Get the input to fusion dictionary.
        """
        if not self._input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        return self._input_to_fusion
    
    @input_to_fusion.setter
    def input_to_fusion(self, value):
        """
        Set the input to fusion dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Input to fusion must be a dictionary.")
        self._input_to_fusion = value

    @property
    def samples_info(self):
        """
        Get the fusion results.
        """
        if not self._samples_info:
            raise ValueError("Run EnsembleInference.fusion first.")
        
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics
    

class LLMEnsemble:
    def __init__(self, perfection_rule_length=5, dynamic_match_percentage=0.8):
        self._initialize_llm()

        self.perfection_rule_length = perfection_rule_length
        self.dynamic_match_percentage = dynamic_match_percentage
        self._input_to_fusion = {}
        self._samples_info = {}
        self._processed_results = []
        self._overall_metrics = None
        LOGGER.info(f"Using the following parameters, perfection_rule_length: {self.perfection_rule_length}, dynamic_match_percentage: {self.dynamic_match_percentage}")

    def _initialize_llm(self):
        self.REINFORCER = reinforcer.TokenReinforcer()
        
    def combine_models_transcriptions(self, *samples_dicts, missing_value=""):
        """
        Align multiple samples_info dicts into an audio-centric structure.

        Args:
            *samples_dicts: Each is a dict like {audio_path: {normalized_prediction: str}}
            missing_value (str): Placeholder when a dict has no prediction for a sample.

        Returns:
            dict: {audio_path: [transcript_from_model1, transcript_from_model2, ...]}
        """
        if not samples_dicts:
            return {}

        # Collect all unique audio paths
        all_audio_paths = sorted({path for d in samples_dicts for path in d.keys()})
        # Build combined dict
        combined = {}
        for audio_path in all_audio_paths:
            combined[audio_path] = []
            for d in samples_dicts:
                if audio_path in d:
                    transcript = d[audio_path].get("normalized_prediction", missing_value)
                else:
                    transcript = missing_value
                combined[audio_path].append(transcript)

        self._input_to_fusion = combined
    
    @staticmethod
    def find_first_anchors(reference_sentence, compared_sentence):
        ref_words = reference_sentence.split()
        comp_words = compared_sentence.split()
        for i, ref_word in enumerate(ref_words):
            for j, comp_word in enumerate(comp_words):
                if ref_word == comp_word:
                    return (i, j)
        return None
    
    def validate_anchor_quality(self, reference, transcription):
        """
        Validate that anchors represent meaningful common structure using SequenceMatcher
        
        Returns:
            tuple: (is_valid, anchor_info)
        """
        anchors = self.find_first_anchors(reference, transcription)
        if not anchors: return False, None
            
        ref_pos, trans_pos = anchors
        ref_words = reference.split()
        trans_words = transcription.split()
        ref_remaining = ref_words[ref_pos:]
        trans_remaining = trans_words[trans_pos:]
        # Use SequenceMatcher on the remaining sequences for better validation
        matcher = SequenceMatcher(None, ref_remaining, trans_remaining)
        # Get the longest matching block starting from position 0 (right after anchor)
        longest_match = matcher.find_longest_match(0, len(ref_remaining), 0, len(trans_remaining))
        additional_matches = longest_match.size
        shorter_sequence_length = min(len(ref_remaining), len(trans_remaining))
        is_valid = False
        min_required = 0
        if shorter_sequence_length <= self.perfection_rule_length:
            # Check if the number of matching words is exactly equal to the length of the shorter sequence.
            min_required = shorter_sequence_length
            if additional_matches == shorter_sequence_length:
                is_valid = True
        else:
            min_required = int(shorter_sequence_length * self.dynamic_match_percentage)
            # Check if we found at least our minimum required number of matches.
            if additional_matches >= min_required:
                is_valid = True

        return is_valid, {
            "positions": anchors, 
            "additional_matches": additional_matches,
            "min_required": min_required
        }

    def get_reference_from_transcriptions(self, transcriptions):
        def get_longest_reference(transcriptions):
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None, None
            
            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: len(x[1]))
            successful_alignments = 0
            total_comparisons = len(valid_transcriptions) - 1
            for original_index, transcription in valid_transcriptions:
                if original_index != longest_original_index:
                    is_valid, _ = self.validate_anchor_quality(longest_reference, transcription)
                    if is_valid:
                        successful_alignments += 1
            
            success_ratio = successful_alignments / total_comparisons if total_comparisons > 0 else 1.0
            success = success_ratio >= 0.5
            metadata = {
                'strategy_metric': 'validation_success_ratio',
                'score': success_ratio,
                'successful_alignments': successful_alignments,
                'total_comparisons': total_comparisons,
                'reference_index': longest_original_index
            }
            return success, longest_reference, longest_original_index, metadata
        
        def get_common_words_reference(transcriptions):
            """
            Reference selection using SequenceMatcher similarity ratios
            """
            def calculate_similarity(sentence1, sentence2):
                """Calculate similarity ratio between two sentences using SequenceMatcher"""
                words1, words2 = sentence1.split(), sentence2.split()
                matcher = SequenceMatcher(None, words1, words2)
                return matcher.ratio()
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None, None

            best_total_similarity = -1
            best_reference, best_index = None, None
            for original_i, transcription_i in valid_transcriptions:
                total_similarity = 0.0
                for original_j, transcription_j in valid_transcriptions:
                    if original_i != original_j:
                        similarity = calculate_similarity(transcription_i, transcription_j)
                        total_similarity += similarity
                
                if total_similarity > best_total_similarity:
                    best_total_similarity = total_similarity
                    best_reference = transcription_i
                    best_index = original_i
            
            if best_reference and best_total_similarity >= 0:
                num_comparisons = len(valid_transcriptions) - 1 if len(valid_transcriptions) > 1 else 1
                avg_similarity = best_total_similarity / num_comparisons if num_comparisons > 0 else 0.0
                metadata = {
                    'strategy_metric': 'average_similarity_to_best',
                    'score': avg_similarity,
                    'total_similarity_score': best_total_similarity,
                    'reference_index': best_index
                }
                return True, best_reference, best_index, metadata
            
            return False, None, None, None

        def get_longest_reference_fallback(transcriptions):
            """
            Fallback when anchor validation fails but we still need a reference.
            Uses average pairwise similarity to find most representative transcription.
            """
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            # Single transcription case
            if len(valid_transcriptions) == 1:
                return True, valid_transcriptions[0][1], valid_transcriptions[0][0]
            
            # Calculate average word-level edit distance for each transcription
            best_avg_similarity = -1
            best_reference = None
            best_index = None
            
            for original_i, transcription_i in valid_transcriptions:
                words_i = transcription_i.split()
                total_similarity = 0.0
                comparisons = 0
                
                for original_j, transcription_j in valid_transcriptions:
                    if original_i != original_j:
                        words_j = transcription_j.split()
                        # Use SequenceMatcher for word-level similarity
                        matcher = SequenceMatcher(None, words_i, words_j)
                        total_similarity += matcher.ratio()
                        comparisons += 1
                
                avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
                
                # Tiebreaker: prefer longer transcription when similarity is equal
                if avg_similarity > best_avg_similarity or \
                (avg_similarity == best_avg_similarity and len(transcription_i) > len(best_reference or "")):
                    best_avg_similarity = avg_similarity
                    best_reference = transcription_i
                    best_index = original_i
            
            # Success if we found reasonable consensus (>30% similarity)
            success = best_avg_similarity >= 0.2
            metadata = {
                'strategy_metric': 'average_pairwise_similarity',
                'score': best_avg_similarity,
                'reference_index': best_index
            }
            return success, best_reference, best_index, metadata
        
        # Always ensure we have at least one valid transcription to return
        valid_transcriptions = [t for t in transcriptions if t is not None]
        if not valid_transcriptions:
            return None, None, None
        
        longest_success, longest_reference, longest_index, reference_metadata = get_longest_reference(transcriptions)
        if longest_success: return "longest", longest_reference, longest_index, reference_metadata

        common_words_success, common_words_reference, common_words_index, reference_metadata = get_common_words_reference(transcriptions)
        if common_words_success: return "common_words", common_words_reference, common_words_index, reference_metadata

        fallback_success, fallback_reference, fallback_index, reference_metadata = get_longest_reference_fallback(transcriptions)
        if fallback_success: return "longest_fallback", fallback_reference, fallback_index, reference_metadata
        else: return "failed", fallback_reference, fallback_index , reference_metadata

    def align_transcriptions_to_reference(self, reference, reference_type, reference_index, transcriptions):
        """
        Simple alignment function that extracts operations and candidate values.
        
        Args:
            reference (str): Reference transcription
            transcription (str): Candidate transcription to align
            
        Returns:
            tuple: (operations, candidate_values)
        """
        # Wont change
        def align_with_longest_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns a list of transcriptions against a single authoritative reference.

            This function uses the provided 'reference' sentence as a structural backbone.
            It iterates through each transcription in the list and compares it to the
            reference using Python's SequenceMatcher. For each position in the
            reference, it determines an operation ('<KEEP>', '<REPLACE>', '<DELETE>')
            and the corresponding token from the transcription.

            The key characteristic of this strategy is that the output structure is
            always dictated by the length of the reference. Insertions in the
            candidate transcriptions are ignored in the final alignment to preserve this
            structure, ensuring all output lists have the same length.

            Args:
                reference (str): The authoritative transcription to align against.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence within the
                    transcriptions list.
                reference_type (str): A string descriptor for the strategy used
                    (e.g., "longest").

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription. Each dictionary contains:
                            - 'model_index' (int): The original index of the model.
                            - 'reference_type' (str): The strategy name.
                            - 'is_reference' (bool): A flag indicating if this was
                            the reference transcription.
                            - 'operations' (list[str]): A list of operations
                            relative to the reference.
                            - 'tokens' (list[str or None]): A list of words (or None)
                            aligned to the reference positions.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # --- PHASE 1: SURVEY AND PAD THE REFERENCE ---
            # Find all unique insertion points from other transcriptions
            insertion_map = {}
            for model_idx, transcription in enumerate(transcriptions):
                if model_idx == reference_index or not transcription: 
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, _, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])

            # Build the new, flexible "padded" reference blueprint
            padded_ref_words = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    # Add placeholders for unique words other models inserted
                    padded_ref_words.extend([None] * len(set(insertion_map[i])))
                padded_ref_words.append(word)
            
            # Handle insertions that occur after the last word of the reference
            if len(ref_words) in insertion_map:
                padded_ref_words.extend([None] * len(set(insertion_map[len(ref_words)])))
                
            padded_ref_length = len(padded_ref_words)

            # --- PHASE 2: ALIGN ALL TRANSCRIPTIONS TO THE PADDED REFERENCE ---
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                # Align the original reference model to the new padded structure
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words[:]
                
                # Align all other models to the new padded structure
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            # 'insert' is now handled implicitly by the padded structure,
                            # so a 'pass' is safe here as a fallback.
                            elif op == 'insert':
                                pass
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata

        def align_with_common_words_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Aligns transcriptions using a flexible, consensus-based reference.

            This strategy is designed for cases where no single transcription is a clear
            structural authority. It works in two main phases:

            1.  **Reference Padding:** It first surveys all other transcriptions to find
                words they contain that are missing from the reference ('insertions').
                It then creates a new "padded" reference by inserting `None`
                placeholders at the appropriate positions to create slots for these
                potential new words.

            2.  **Final Alignment:** It then aligns all transcriptions against this new,
                longer, padded reference. This allows words that would have been
                insertions to be properly mapped to a slot, enabling a vote on whether
                they should be included in the final output.

            The purpose is to create a flexible alignment structure that accommodates
            structural differences between models, rather than strictly enforcing the
            structure of one reference.

            Args:
                reference (str): The consensus-based transcription to use as a starting point.
                transcriptions (list[str]): A list of all candidate transcriptions.
                reference_index (int): The index of the reference sentence.
                reference_type (str): A string descriptor for the strategy.

            Returns:
                list[dict]: A list of alignment result dictionaries, one for each
                            transcription, all aligned to the padded reference length.
            """
            alignment_results = []
            ref_words = reference.split()
            
            # Pad reference by finding all unique insertion points from other transcriptions
            padded_ref_words = ref_words.copy()
            insertion_map = {}  # Maps ref position -> list of words inserted before it
            for model_index, transcription in enumerate(transcriptions):
                if model_index == reference_index or transcription is None:
                    continue
                
                trans_words = transcription.split()
                matcher = SequenceMatcher(None, ref_words, trans_words)
                for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                    if op == 'insert':
                        # Words in transcription that don't exist in reference
                        if ref_start not in insertion_map:
                            insertion_map[ref_start] = []
                        insertion_map[ref_start].extend(trans_words[trans_start:trans_end])
            
            # Build padded reference with None placeholders for insertions
            final_ref = []
            for i, word in enumerate(ref_words):
                if i in insertion_map:
                    final_ref.extend([None] * len(set(insertion_map[i])))  # Unique insertions only
                final_ref.append(word)
            if len(ref_words) in insertion_map:  # Trailing insertions
                final_ref.extend([None] * len(set(insertion_map[len(ref_words)])))
            
            padded_ref_words = final_ref
            padded_ref_length = len(padded_ref_words)
            
            # Align each transcription to padded reference
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>" if w is not None else "<DELETE>" for w in padded_ref_words]
                    alignment_result['tokens'] = padded_ref_words.copy()
                else:
                    if not transcription:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    tokens[i] = trans_words[trans_start + (i - ref_start)]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        tokens[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                            elif op == 'insert':
                                pass  # Insertions don't map to reference positions
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                alignment_results.append(alignment_result)

            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': padded_ref_words,
                'insertion_map': insertion_map
            }
            return alignment_results, metadata
        
        def align_with_longest_fallback_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            """
            Fallback alignment when anchor validation fails.
            
            Flow:
            1. Compute pairwise similarity matrix for all valid transcriptions
            2. Calculate reliability weight for each model (average similarity to others)
            3. Perform standard word-level alignment (same as longest strategy)
            4. Attach weight metadata to each alignment result for downstream use
            
            Difference from longest strategy:
            - Longest: High anchor confidence, reference is structural authority
            - Fallback: Low anchor confidence, reference is "best guess", weights signal reliability
            
            Output: Same structure as longest strategy + 'weight' field per model
            Purpose: Enable weighted voting in consensus step when reference quality is uncertain
            """
            alignment_results = []
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t and t.strip()]
            
            if len(valid_transcriptions) <= 1:
                # Degenerate case: use simple alignment without weights
                return align_with_longest_strategy(reference, transcriptions, reference_index, reference_type)
            
            # Calculate pairwise similarities
            model_weights = {}
            for i, trans_i in valid_transcriptions:
                words_i = trans_i.split()
                total_sim = 0.0
                comparisons = 0
                for j, trans_j in valid_transcriptions:
                    if i != j:
                        words_j = trans_j.split()
                        total_sim += SequenceMatcher(None, words_i, words_j).ratio()
                        comparisons += 1
                model_weights[i] = total_sim / comparisons if comparisons > 0 else 0.0
            
            # Normalize weights to [0, 1]
            max_weight = max(model_weights.values()) if model_weights else 1.0
            if max_weight > 0:
                model_weights = {k: v / max_weight for k, v in model_weights.items()}
            
            # Perform standard alignment
            ref_words = reference.split()
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index),
                    'weight': model_weights.get(model_index, 0.0)  # Reliability score
                }
                
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>"] * len(ref_words)
                    alignment_result['tokens'] = ref_words.copy()
                else:
                    if not transcription or not transcription.strip():
                        alignment_result['operations'] = ["<DELETE>"] * len(ref_words)
                        alignment_result['tokens'] = [None] * len(ref_words)
                    else:
                        trans_words = transcription.split()
                        operations = ["<DELETE>"] * len(ref_words)
                        candidate_values = [None] * len(ref_words)
                        
                        matcher = SequenceMatcher(None, ref_words, trans_words)
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            if op == 'equal':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<KEEP>"
                                    candidate_values[i] = ref_words[i]
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        candidate_values[i] = trans_words[trans_idx]
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                        
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = candidate_values
                
                alignment_results.append(alignment_result)
            metadata = {
                'original_reference_tokens': ref_words,
                'adjusted_reference_tokens': ref_words.copy(), 
                'model_weights': model_weights # This is the key metadata for this strategy
            }
            return alignment_results, metadata
        
        if reference is None or not transcriptions:
            return []
        if reference_type == "longest":
            alignment_results, alignment_metadata = align_with_longest_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "common_words":
            alignment_results, alignment_metadata = align_with_common_words_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "longest_fallback":
            alignment_results, alignment_metadata = align_with_longest_fallback_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "failed":
            alignment_results, alignment_metadata = align_with_longest_fallback_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        return alignment_results, alignment_metadata
    
    def llm_voting_scheme(self, alignment_results):
        """
        Use LLM-only voting for token selection.
        
        Args:
            alignment_results: List of dictionaries containing alignment data for each model
            
        Returns:
            dict: Final voting result with tokens and metadata
        """
        def construct_final_transcription(fusion_tokens):
            """Build the final transcription from tokens"""
            return " ".join(token for token in fusion_tokens if token is not None)
        
        if not alignment_results or len(alignment_results) == 0:
            return {
                "fusion_transcript": "",
                "fusion_tokens": [],
                "candidates_tokens": [],
                "voting_metadata": {"status": "No alignment results"}
            }
        
        result = self.REINFORCER.main(alignment_results)
        fusion_tokens = result["selected_tokens"]
        selection_metadata = result["selection_metadata"]
        fusion_transcript = construct_final_transcription(fusion_tokens)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        voting_metadata = {
            'reference_type': alignment_results[0].get('reference_type'),
            'total_models': len(alignment_results),
            'operations_length': len(fusion_tokens),
            'selection_method': 'llm_only',
            'selection_details': selection_metadata
        }
        return {
            "fusion_transcript": fusion_transcript,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_metadata": voting_metadata
        }

    def main(self, **kwargs):
        def fuse_sample_transcriptions(audio_path, transcriptions):
            voting_start = time.time()
            reference_type, reference, reference_index, reference_metadata = self.get_reference_from_transcriptions(transcriptions)
            alignment_results, alignment_metadata = self.align_transcriptions_to_reference(reference=reference, reference_type=reference_type, 
                                                                                           reference_index=reference_index, transcriptions=transcriptions)
            voting_output = self.llm_voting_scheme(alignment_results)
            data = {
                # --- Primary Outputs ---
                "fusion_transcript": voting_output["fusion_transcript"],
                
                # --- Diagnostic/Internal Data ---
                "fusion_tokens": voting_output["fusion_tokens"],
                "candidates_tokens": voting_output["candidates_tokens"],
                "voting_time": time.time() - voting_start,
                
                # --- Aggregated Metadata ---
                "metadata": {
                    "reference_selection": reference_metadata,
                    "alignment": alignment_metadata,
                    "voting": voting_output["voting_metadata"],
                }
            }
            
            return {audio_path: data}

        def process_item(item):
            key, value = item
            transcriptions_lists = [t for t in value]
            voting_result = fuse_sample_transcriptions(key, transcriptions_lists)
            return voting_result
        
        if not self.input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_item, item): item for item in self.input_to_fusion.items()}
            samples_info = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing Inputs..."):
                result = future.result()  # this is {audio_path: {...}}
                samples_info.update(result)

        self._samples_info = dict(sorted(samples_info.items()))

    def eval(self, audios_chunk):
        refs_lookup = {sample["audio_path"]: sample["normalized_transcription"] for sample in audios_chunk}
        all_audio_paths = list(self._samples_info.keys())
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    ref = refs_lookup.get(audio_path)
                    hyp = self._samples_info[audio_path]["fusion_transcript"]
                    self._samples_info[audio_path]["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(hyp, substitute=True)
                    self._samples_info[audio_path]["normalized_transcription"] = ref
                    if ref is not None:
                        norm_hyp = self._samples_info[audio_path]["normalized_prediction"]
                        sample_metrics = metrics.BasicSTTMetrics.evaluate(refs=ref, hyps=norm_hyp)
                        self._samples_info[audio_path]["metrics"] = sample_metrics
                    else:
                        self._samples_info[audio_path]["metrics"] = helpers._empty_metrics()

                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {
                        "normalized_prediction": None,
                        "metrics": helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }

        # Compute overall metrics
        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def summary_of_evaluation(self):
        """
        Display a simple summary of the overall evaluation metrics.
        """
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return

        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")

    def reset(self):
        """
        Reset the EnsembleInferenceRefactored instance.
        Clears input_to_fusion and fusion_results, but keeps the model loaded.
        """
        self._input_to_fusion = {}
        self._fusion_results = []
        self._overall_metrics = None
        LOGGER.info("Ensemble instance has been reset.")

    @property
    def input_to_fusion(self):
        """
        Get the input to fusion dictionary.
        """
        if not self._input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")
        
        return self._input_to_fusion
    
    @input_to_fusion.setter
    def input_to_fusion(self, value):
        """
        Set the input to fusion dictionary.
        """
        if not isinstance(value, dict):
            raise ValueError("Input to fusion must be a dictionary.")
        self._input_to_fusion = value

    @property
    def samples_info(self):
        """
        Get the fusion results.
        """
        if not self._samples_info:
            raise ValueError("Run EnsembleInference.fusion first.")
        
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics