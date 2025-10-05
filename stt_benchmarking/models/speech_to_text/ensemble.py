from difflib import SequenceMatcher
from collections import defaultdict
from itertools import islice
from collections import Counter, OrderedDict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

from stt_benchmarking.utils import text_processing, helpers, metrics
from stt_benchmarking.models.llms import reinforcer
from . import LOGGER

class HybridEnsemble:
    def __init__(self, use_llm):
        self.use_llm = use_llm
        if self.use_llm:
            self.initialize_llm()
        self._input_to_fusion = {}
        self._fusion_results = []
        self._processed_results = []
        self._overall_metrics = None

    def initialize_llm(self):
        self.REINFORCER = reinforcer.TokenReinforcer()
    
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
    
    @staticmethod
    def find_first_anchors(reference_sentence, compared_sentence):
        ref_words = reference_sentence.split()
        comp_words = compared_sentence.split()
        for i, ref_word in enumerate(ref_words):
            for j, comp_word in enumerate(comp_words):
                if ref_word == comp_word:
                    return (i, j)
        return None
    
    @staticmethod
    def validate_anchor_quality(reference, transcription):
        """
        Validate that anchors represent meaningful common structure using SequenceMatcher
        
        Returns:
            tuple: (is_valid, anchor_info)
        """
        anchors = HybridEnsemble.find_first_anchors(reference, transcription)
        if not anchors:
            return False, None
            
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
        min_len = min(len(ref_remaining), len(trans_remaining))
        if min_len <= 2:
            is_valid = additional_matches == min_len
        else:
            is_valid = additional_matches >= 2
            
        return is_valid, {
            "positions": anchors, 
            "additional_matches": additional_matches,
            "min_required": 2 if min_len > 2 else min_len
        }

    # Done reference
    def get_reference_from_transcriptions(self, transcriptions):
        def get_longest_reference(transcriptions):
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: len(x[1]))
            successful_alignments = 0
            total_comparisons = len(valid_transcriptions) - 1
            for original_index, transcription in valid_transcriptions:
                if original_index != longest_original_index:
                    is_valid, anchor_info = self.validate_anchor_quality(longest_reference, transcription)
                    if is_valid:
                        successful_alignments += 1
            
            success_ratio = successful_alignments / total_comparisons if total_comparisons > 0 else 1.0
            success = success_ratio >= 0.5
            return success, longest_reference, longest_original_index

        def get_longest_reference_fallback(transcriptions):
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None
            
            longest_original_index, longest_reference = max(valid_transcriptions, key=lambda x: len(x[1]))
            return True, longest_reference, longest_original_index
        
        def get_common_words_reference(transcriptions):
            """
            Reference selection using SequenceMatcher similarity ratios
            """
            def calculate_similarity(sentence1, sentence2):
                """Calculate similarity ratio between two sentences using SequenceMatcher"""
                words1 = sentence1.split()
                words2 = sentence2.split()
                matcher = SequenceMatcher(None, words1, words2)
                return matcher.ratio()
            
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            if not valid_transcriptions:
                return False, None, None

            best_total_similarity = -1
            best_reference = None
            best_index = None
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
            
            if best_reference and best_total_similarity > 0:
                return True, best_reference, best_index
            
            return False, None, None

        # Always ensure we have at least one valid transcription to return
        valid_transcriptions = [t for t in transcriptions if t is not None]
        if not valid_transcriptions:
            return None, None, None
        
        longest_success, longest_reference, longest_index = get_longest_reference(transcriptions)
        if longest_success:
            return "longest", longest_reference, longest_index
        else:
            common_words_success, common_words_reference, common_words_index = get_common_words_reference(transcriptions)
            if common_words_success:
                return "common_words", common_words_reference, common_words_index
            else:
                _, longest_reference, longest_original_index = get_longest_reference(transcriptions)
                return "longest_fallback", longest_reference, longest_original_index

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
            alignment_results = []
            ref_words = reference.split()
            for model_index, transcription in enumerate( transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>"] * len(ref_words)
                    alignment_result['tokens'] = ref_words.copy()
                else:
                    if not transcription:
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
                                    candidate_values[i] = ref_words[i]  # Same as trans_words[trans_start + (i - ref_start)]
                                    
                            elif op == 'replace':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<REPLACE>"
                                    trans_idx = trans_start + (i - ref_start)
                                    if trans_idx < trans_end:
                                        candidate_values[i] = trans_words[trans_idx]
                                    else:
                                        candidate_values[i] = None
                                        
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                                    candidate_values[i] = None
                                    
                            elif op == 'insert':
                                # Words exist in transcription but not in reference
                                # These don't affect our reference-based operations array
                                # but could be logged for debugging
                                pass

                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = candidate_values
                alignment_results.append(alignment_result)
            return alignment_results        

        # Maybe needs small tweaking
        def align_with_common_words_strategy(reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            alignment_results = []
            valid_transcriptions = [(i, t) for i, t in enumerate(transcriptions) if t is not None]
            ref_words = reference.split()
            max_length = max(len(t.split()) for _, t in valid_transcriptions)
            longest_idx, longest_trans = max(valid_transcriptions, key=lambda x: len(x[1].split()))
            longest_words = longest_trans.split()
            padded_transcriptions_words = {}
            padded_ref_words = ref_words.copy()
            for model_index, transcription in enumerate(transcriptions):
                if model_index == longest_idx:
                    padded_transcriptions_words[model_index] = longest_words
                    continue

                if model_index == reference_index:
                    matcher = SequenceMatcher(None, longest_words, padded_ref_words)
                    insert_positions = []
                    for op, long_start, long_end, ref_start, ref_end in matcher.get_opcodes():
                        if op == 'delete':
                            for i in range(long_start, long_end):
                                insert_positions.append((ref_start, longest_words[i]))
                    for pos, _ in sorted(insert_positions, key=lambda x: x[0]):
                        padded_ref_words.insert(pos, None)
                    padded_ref_words = padded_ref_words[:max_length] if len(padded_ref_words) > max_length else padded_ref_words + [None] * (max_length - len(padded_ref_words))
                    padded_transcriptions_words[model_index] = padded_ref_words
                else:
                    if transcription is None:
                        continue
                    else:
                        transcription_words = transcription.split()
                        matcher = SequenceMatcher(None, longest_words, transcription_words)
                        insert_positions = []
                        for op, long_start, long_end, ref_start, ref_end in matcher.get_opcodes():
                            if op == 'delete':
                                for i in range(long_start, long_end):
                                    insert_positions.append((ref_start, longest_words[i]))
                        for pos, _ in sorted(insert_positions, key=lambda x: x[0]):
                            transcription_words.insert(pos, None)
                        transcription_words = transcription_words[:max_length] if len(transcription_words) > max_length else transcription_words + [None] * (max_length - len(transcription_words))
                        # Save padded transcription words
                        padded_transcriptions_words[model_index] = transcription_words
                        # transcriptions[model_index] = ' '.join([w for w in transcription_words if w is not None])

            padded_ref_length = len(padded_ref_words)

            # Step 2: Align all transcriptions to padded reference
            inserted_words = {}
            for model_index, transcription in enumerate(transcriptions):
                alignment_result = {
                    'model_index': model_index,
                    'reference_type': reference_type,
                    'is_reference': (model_index == reference_index)
                }
                if model_index == reference_index:
                    alignment_result['operations'] = ["<KEEP>"] * padded_ref_length
                    alignment_result['tokens'] = padded_ref_words.copy()
                else:
                    if model_index in padded_transcriptions_words:
                        trans_words = padded_transcriptions_words[model_index]
                    else:
                        trans_words = []

                    if not trans_words:
                        alignment_result['operations'] = ["<DELETE>"] * padded_ref_length
                        alignment_result['tokens'] = [None] * padded_ref_length
                    else:
                        operations = ["<DELETE>"] * padded_ref_length
                        tokens = [None] * padded_ref_length
                        matcher = SequenceMatcher(None, padded_ref_words, trans_words)
                        inserted_words[model_index] = []
                        for op, ref_start, ref_end, trans_start, trans_end in matcher.get_opcodes():
                            # ref_text = padded_ref_words[ref_start:ref_end]
                            # trans_text = trans_words[trans_start:trans_end]

                            # print(f"{op.upper():<9} | "
                            #     f"ref[{ref_start}:{ref_end}] = '{ref_text}' | "
                            #     f"trans[{trans_start}:{trans_end}] = '{trans_text}'")
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
                                    else:
                                        tokens[i] = None
                            elif op == 'delete':
                                for i in range(ref_start, ref_end):
                                    operations[i] = "<DELETE>"
                                    tokens[i] = None
                            elif op == 'insert':
                                inserted_words[model_index].extend(trans_words[trans_start:trans_end])
                        alignment_result['operations'] = operations
                        alignment_result['tokens'] = tokens
                    # print("--------------")
                alignment_results.append(alignment_result)

            return alignment_results
        
        def align_with_longest_fallback_strategy(weights, reference, transcriptions, reference_index=reference_index, reference_type=reference_type):
            pass
        
        if reference is None or not transcriptions:
            return []
        if reference_type == "longest":
            alignment_results = align_with_longest_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        elif reference_type == "common_words":
            alignment_results = align_with_common_words_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        else: # Handle it later
            alignment_results = align_with_longest_strategy(reference=reference, transcriptions=transcriptions, reference_index=reference_index, reference_type=reference_type)
        return alignment_results
    
    def voting_scheme(self, audio_path, alignment_results):
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
                priority_order = ["<REPLACE>", "<DELETE>", "<KEEP>", "<SKIP>", "<INSERT>"]
                for preferred_op in priority_order:
                    if preferred_op in tied_operations:
                        majority_operation = preferred_op
                        break
            
            return majority_operation, operation_counts
        
        def vote_for_token(position_votes, majority_operation):
            """Determine final token based on operation with random selection"""
            final_token = None
            if majority_operation == "<KEEP>":
                # Find reference token or use first available
                for i, is_ref in enumerate(position_votes['is_reference_flags']):
                    if is_ref:
                        final_token = position_votes['tokens'][i]
                        break
                
                if final_token is None:
                    final_token = position_votes['tokens'][0]
                    
            elif majority_operation in ["<REPLACE>", "<INSERT>"]:
                # Random selection among models that chose this operation
                operation_tokens = []
                
                for i, op in enumerate(position_votes['operations']):
                    if op == majority_operation and position_votes['tokens'][i] is not None:
                        operation_tokens.append(position_votes['tokens'][i])
                
                if operation_tokens:
                    final_token = random.choice(operation_tokens)
                    
            elif majority_operation in ["<DELETE>", "<SKIP>"]:
                final_token = None
            
            return final_token
        
        def create_voting_detail(position, operation_counts, majority_operation, final_token, position_votes):
            """Create detailed voting information for a position"""
            token_weights = {}
            
            if majority_operation in ["<REPLACE>", "<INSERT>"]:
                for token in set(position_votes['tokens']):
                    if token is not None:
                        token_weights[token] = sum(
                            1  # Count occurrences instead of weights
                            for i, t in enumerate(position_votes['tokens']) 
                            if t == token and position_votes['operations'][i] == majority_operation
                        )
            
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
        
        def calculate_confidence_score(alignment_results, operations_length):
            """Calculate overall confidence based on operation agreement"""
            total_positions = operations_length
            operation_confidence = sum(
                max(Counter([result['operations'][i] for result in alignment_results]).values()) / len(alignment_results)
                for i in range(total_positions)
            ) / total_positions if total_positions > 0 else 0
            
            return operation_confidence
        
        def create_metadata(alignment_results, fusion_operations):
            """Create metadata about the voting results"""
            return {
                'reference_type': alignment_results[0]['reference_type'],
                'total_keep': fusion_operations.count('<KEEP>'),
                'total_replace': fusion_operations.count('<REPLACE>'),
                'total_insert': fusion_operations.count('<INSERT>'),
                'total_delete': fusion_operations.count('<DELETE>'),
                'total_skip': fusion_operations.count('<SKIP>')
            }
        
        voting_result = {}
        if not alignment_results or len(alignment_results) == 0:
            voting_result[audio_path] = {
                "fusion_transcript": "",
                "fusion_operations": [],
                "fusion_tokens": [],
                "candidates_tokens": [],
                "voting_details": {},
                "confidence_score": 0,
                "total_models": 0,
                "operations_length": 0,
                "metadata": {},
            }
            return voting_result
                
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
        confidence_score = calculate_confidence_score(alignment_results, operations_length)
        metadata = create_metadata(alignment_results, fusion_operations)
        candidates_tokens = [element["tokens"] for element in alignment_results]
        voting_result[audio_path] = {
            "fusion_transcript": fusion_transcript,
            "fusion_operations": fusion_operations,
            "fusion_tokens": fusion_tokens,
            "candidates_tokens": candidates_tokens,
            "voting_details": voting_details,
            "confidence_score": confidence_score,
            "total_models": len(alignment_results),
            "operations_length": operations_length,
            "metadata": metadata,
        }
        return voting_result

    def llm_reinforcer(self, fusion_tokens, candidates_tokens, max_tokens, chunk_size, overlap):
        def postprocess_reinforced_output(response: reinforcer.GeneratedResponse) -> Dict[str, Any]:
            results = response.reinforced_results
            final_tokens = []
            modifications = 0
            for item in results:
                token = item.get("token")
                final_tokens.append(token)
                if item.get("is_modified"):
                    modifications += 1

            safe_tokens = [t for t in final_tokens if t not in [None, "None", "Null", "null"]]
            final_transcript = " ".join(safe_tokens).strip()
            return {
                "final_tokens": final_tokens,
                "final_transcript": final_transcript,
                "modifications": modifications,
                "total_tokens": len(final_tokens),
                "modification_ratio": modifications / len(final_tokens) if final_tokens else 0.0,
                "is_chunked": response.is_chunked
            }
        
        llm_response = {}
        if self.use_llm:
            response = self.REINFORCER.main(
                fusion_tokens=fusion_tokens,
                candidate_tokens=candidates_tokens,
                max_tokens=max_tokens, chunk_size=chunk_size, overlap=overlap, 
            )
            llm_response = postprocess_reinforced_output(response=response)
        return llm_response

    def fusion(self, **kwargs):
        def fuse_sample_transcriptions(audio_path, transcriptions):
            reference_type, reference, reference_index = self.get_reference_from_transcriptions(transcriptions)
            alignment_results = self.align_transcriptions_to_reference(reference=reference, reference_type=reference_type, reference_index=reference_index,
                                                                       transcriptions=transcriptions)
            voting_result = self.voting_scheme(audio_path, alignment_results)
            data = voting_result[audio_path]
            llm_response = self.llm_reinforcer(
                fusion_tokens=data["fusion_tokens"],
                candidates_tokens=data["candidates_tokens"],
                max_tokens=kwargs.get("max_tokens", 25),
                chunk_size=kwargs.get("chunk_size", 15),
                overlap=kwargs.get("overlap", 3),
            )
            data["llm_response"] = llm_response
            voting_result[audio_path] = data
            return voting_result

        def process_item(item):
            key, value = item
            transcriptions_lists = [t for t in value]
            voting_result = fuse_sample_transcriptions(key, transcriptions_lists)
            return voting_result
        
        if not self.input_to_fusion:
            raise ValueError("Run EnsembleInference.align_model_records first.")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_item, item): item for item in self.input_to_fusion.items()}
            fusion_results = {}
            fusion_results = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fusing Inputs..."):
                result = future.result()  # this is {audio_path: {...}}
                fusion_results.update(result)

        self._fusion_results = dict(sorted(fusion_results.items()))

    def eval(self, audios_chunk):
        def _eval_common(data):
            return data.get("fusion_transcript")

        def _eval_llm(data):
            llm_resp = data.get("llm_response", {})
            return llm_resp.get("final_transcript")

        refs_lookup = {sample["audio_path"]: sample["normalized_transcription"] for sample in audios_chunk}
        all_refs_normalized = []
        all_hyps_normalized = []
        for audio_path, data in self._fusion_results.items():
            norm_transcript = _eval_llm(data) if self.use_llm else _eval_common(data)
            data["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(
                norm_transcript, substitute=True
            )
            ref = refs_lookup.get(audio_path)

            if ref is not None:
                data["normalized_transcription"] = ref
                try:
                    sample_metrics = metrics.BasicSTTMetrics.evaluate(refs=ref, hyps=norm_transcript)
                    data["metrics"] = sample_metrics
                    all_refs_normalized.append(ref)
                    all_hyps_normalized.append(norm_transcript)
                except Exception as e:
                    LOGGER.error(f"Metric calc failed for {audio_path}: {e}")
                    data["metrics"] = helpers._empty_metrics()
            else:
                LOGGER.warning(f"No reference transcription found for {audio_path}")
                data["normalized_transcription"] = None
                data["metrics"] = helpers._empty_metrics()

        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=all_refs_normalized, hyps=all_hyps_normalized)

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
    def fusion_results(self):
        """
        Get the fusion results.
        """
        if not self._fusion_results:
            raise ValueError("Run EnsembleInference.fusion first.")
        
        return self._fusion_results
    
    @property
    def overall_metrics(self):
        return self._overall_metrics