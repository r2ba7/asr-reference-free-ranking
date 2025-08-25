import re
import newspaper


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
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    # Final cleanup - remove extra spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Updated scraping code with text cleaning
def scrape_newspapers_with_cleaning(no_articles):
    news_sources = {
        "Egypt": {
            "AlAhram": "https://www.ahram.org.eg",
            "AlMasryAlYoum": "https://www.almasryalyoum.com"
        },
        "Saudi": {
            "Okaz": "https://www.okaz.com.sa",
            "AlRiyadh": "https://www.alriyadh.com"
        }
    }

    articles_data = []
    for country, papers in news_sources.items():
        for paper, url in papers.items():
            print(f"Scraping {paper}...")
            paper_obj = newspaper.build(url, memoize_articles=False, language='ar')
            for content in paper_obj.articles[:no_articles]:
                try:
                    content.download()
                    content.parse()
                    content.nlp()

                    if not content.text:
                        continue 

                    # Clean the text
                    cleaned_text = clean_arabic_text(content.text)
                    
                    if not cleaned_text:  
                        continue

                    # Split words
                    words = cleaned_text.split()
                    word_count = len(words)

                    # If longer than 100, take only first 100 words
                    if word_count > 100:
                        words = words[:100]
                        cleaned_text = " ".join(words)
                        word_count = 100
                        print(f"Truncated {paper} article to 100 words")

                    articles_data.append({
                        "country": country,
                        "source": paper,
                        "title": content.title,
                        "text": cleaned_text,
                        "word_count": word_count,
                        "keywords": content.keywords,
                        "url": content.url
                    })

                except Exception as e:
                    print(f"Error scraping {paper}: {e}")

    return articles_data