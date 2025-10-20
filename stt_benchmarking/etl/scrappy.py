import re

import newspaper
import praw
from langdetect import detect

def clean_arabic_text(text):
    """
    Clean Arabic newspaper text by removing unwanted characters and formatting.
    
    Args:
        text (str): Raw text from newspaper scraping
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove newlines and replace with spaces
    text = text.replace('\n', ' ')
    
    # Remove excessive whitespace (multiple spaces, tabs, etc.)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common newspaper artifacts
    text = re.sub(r'موضوعات مقترحة', '', text)
    text = re.sub(r'اقرأ أيضا[ً]?', '', text)
    text = re.sub(r'المزيد من التفاصيل', '', text)
    text = re.sub(r'تابع القراءة', '', text)
    text = re.sub(r'للمزيد من الأخبار', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove extra punctuation and special characters
    text = re.sub(r'[•◦▪▫■□●○◆◇★☆]+', '', text)
    text = re.sub(r'[-]{2,}', '-', text)
    text = re.sub(r'[=]{2,}', '', text)
    
    # Remove standalone numbers that might be page numbers or section numbers
    text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)  # Remove fractions like "4 / صفر"
    
    # Remove common footer/header text patterns
    text = re.sub(r'جميع الحقوق محفوظة', '', text)
    text = re.sub(r'© \d{4}', '', text)
    
    # Remove single characters that are artifacts
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    
    # Clean up quotation marks
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('\'', "'").replace('\'', "'")
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    # Final cleanup - remove extra spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def contains_english(text):
    """Checks if a string contains any English alphabet characters."""
    return bool(re.search(r'[a-zA-Z]', text))

def scrape_newspapers_with_cleaning(no_articles_per_paper):
    """
    Scrapes Arabic newspaper articles, cleans the text, and creates a new record for each sentence.

    Args:
        no_articles_per_paper (int): The number of articles to scrape from each newspaper.

    Returns:
        list: A list of dictionaries, where each dictionary represents a single sentence from an article.
    """
    # --- Full List of Newspapers ---
    news_sources = {
        "Egypt": {
            "AlAhram": "https://gate.ahram.org.eg/",
            "Youm7": "https://www.youm7.com",
            "AlMasryAlYoum": "https://www.almasryalyoum.com"
        },
        "Saudi": {
            "Okaz": "https://www.okaz.com.sa",
            "AlRiyadh": "https://www.alriyadh.com",
            "AlWatan": "https://www.alwatan.com.sa",
            "AlEqtisadiah": "https://www.aleqt.com"
        },
        "Qatar": {
            "AlJazeera": "https://www.aljazeera.net"
        }
    }

    articles_data = []
    for country, papers in news_sources.items():
        for paper, url in papers.items():
            print(f"Scraping {paper} from {country}...")
            try:
                # Increased timeout to handle slow connections
                paper_obj = newspaper.build(url, memoize_articles=False, language='ar', request_timeout=15)
                
                print(f"  - Found {len(paper_obj.articles)} articles for {paper}.") # Debug print

                processed_articles_count = 0
                # Loop through all found articles until the desired number of valid ones are processed
                for content in paper_obj.articles:
                    # If we have enough articles, stop processing this source
                    if processed_articles_count >= no_articles_per_paper:
                        break

                    try:
                        # Skip non-Arabic subdomains to avoid irrelevant content and errors
                        if 'french.' in content.url or 'english.' in content.url:
                            print(f"  - Skipping non-Arabic URL: {content.url}")
                            continue

                        content.download()
                        content.parse()

                        # Continue to the next article if the text is empty
                        if not content.text:
                            continue

                        # If we get here, the article is valid for processing.
                        # Increment the counter of successfully processed articles.
                        processed_articles_count += 1

                        # --- 1. Remove headers/titles from the beginning of the text ---
                        text = content.text
                        title = content.title.strip() if content.title else ""
                        if title and text.strip().startswith(title):
                            text = text.strip()[len(title):].strip()

                        # Clean the text using the dedicated function
                        cleaned_text = clean_arabic_text(text)
                        if not cleaned_text:
                            continue

                        # --- 2. Create a new record per sentence ---
                        # Split on Arabic full stop, standard period, and Arabic question mark
                        sentences = re.split(r'[\u06D4\u002E\u061F]+', cleaned_text)

                        for sentence in sentences:
                            sentence = sentence.strip()
                            # Ensure the sentence is not empty after stripping whitespace
                            if sentence:
                                word_count = len(sentence.split())
                                if 4 <= word_count <= 80:
                                    # --- 3. Append data without keywords ---
                                    articles_data.append({
                                        "country": country,
                                        "source": paper,
                                        "title": content.title,
                                        "text": sentence,
                                        "word_count": len(sentence.split()),
                                        "url": content.url
                                    })

                    except Exception as e:
                        print(f"  - Error processing article {content.url}: {e}")

            except Exception as e:
                print(f"Error building newspaper object for {paper}: {e}")

    return articles_data

def scrape_reddit_with_cleaning(no_posts_per_sub):
    """
    Scrapes a specific number of Arabic Reddit posts per subreddit,
    filters out any with English text, and creates a new record for each sentence.
    """
    # --- IMPORTANT ---
    # You must configure your Reddit API credentials for this to work.
    # See PRAW documentation: https://praw.readthedocs.io/en/latest/getting_started/quick_start.html
    try:
        reddit = praw.Reddit(
            client_id="MC_Oh3jCvIyFQn_9HQd2iw",
            client_secret="B2iTl5CaYim9gbsmMFKMEKnmfIz2Wg",
            user_agent="Glittering_Lock_1575",
            read_only=True # Recommended for scraping
        )
    except Exception as e:
        print("Could not initialize PRAW. Please check your Reddit API credentials.")
        print(e)
        return []

    subreddits = [
        ("Cairo", "Egypt"),
        ("Personalfinanceegypt", "Egypt"),
        ("saudiarabia", "Saudi"),
        ("Riyadh", "Saudi")
    ]

    reddit_data = []
    for sub_name, country in subreddits:
        print(f"Scraping subreddit: r/{sub_name}...")
        try:
            subreddit = reddit.subreddit(sub_name)
            # Fetch the number of posts specified by the user
            for submission in subreddit.new(limit=no_posts_per_sub):
                
                # Skip submissions that are direct links to images
                if submission.url.endswith(('.png', '.jpg', '.jpeg', '.gif')): continue

                # Skip NSFW (+18) posts
                if submission.over_18: continue
                
                # Use only the selftext, not the title
                post_text = submission.selftext

                # Skip if the post has no body text
                if not post_text: continue

                # Skip the entire post if it contains any English characters
                if contains_english(post_text): continue

                cleaned_text = clean_arabic_text(post_text)
                if not cleaned_text: continue
                
                # Split the cleaned text into sentences
                sentences = re.split(r'[\u06D4\u002E\u061F]+', cleaned_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        word_count = len(sentence.split())
                        # Add new limitation: only include sentences between 4 and 80 words
                        if 4 <= word_count <= 80:
                            reddit_data.append({
                                "country": country,
                                "source": f"r/{sub_name}",
                                "author": submission.author.name if submission.author else "[deleted]",
                                "title": submission.title, # Keep the original post title for each sentence
                                "text": sentence,
                                "word_count": word_count,
                                "url": submission.url
                            })
        except Exception as e:
            print(f"  - Error processing subreddit r/{sub_name}: {e}")

    return reddit_data

    return reddit_data