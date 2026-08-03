"""
Historical News Database Enhancer - Quad-API Edition

This module enhances the existing filtered news database by adding historical data:
- Month-by-month collection with minimum article targets
- Appends to existing GPT-filtered database  
- User-specified date ranges and thresholds
- Preserves existing data while adding historical coverage

🚀 QUAD-API STRATEGY (Maximum Rate Limit Avoidance):
• API 1: GNews API (100 requests/day, 100 articles/request)
• API 2: NewsAPI.org (1000 requests/day, 100 articles/request)
• API 3: TheNewsAPI.com (150 requests/day, 100 articles/request)
• API 4: NY Times API (500 requests/day, 10 articles/page)

Total Daily Capacity:
- 1,750 total requests across all APIs
- Up to 175,000 potential articles per day
- Cascading fallback strategy for maximum coverage

Waterfall Strategy:
1. GNews for relevance (100/day)
2. NewsAPI for volume (1000/day)
3. TheNewsAPI for additional coverage (150/day)
4. NYTimes as final fallback (500/day)

Usage:
    python historical_news_enhancer.py
"""

from openai import OpenAI
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import requests
import time
import sys
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
from calendar import monthrange

# --- load API keys from .env (kept out of git); safe no-op if .env is absent ---
import os as _os, pathlib as _pl
for _d in [_pl.Path(__file__).resolve().parent, *_pl.Path(__file__).resolve().parents]:
    _env = _d / ".env"
    if _env.exists():
        for _line in _env.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _os.environ.setdefault(_k.strip(), _v.strip().strip(chr(34)).strip(chr(39)))
        break
# --- end .env loader ---
warnings.filterwarnings('ignore')

class HistoricalNewsEnhancer:
    """
    Enhances existing news database with historical data
    """
    
    def __init__(self, openai_api_key: str, nyt_api_key: str = None, gnews_api_key: str = None, 
                 newsapi_key: str = None, thenewsapi_key: str = None):
        """Initialize with API keys"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.nyt_api_key = nyt_api_key or os.environ.get("NYT_API_KEY", "")
        self.gnews_api_key = gnews_api_key or os.environ.get("GNEWS_API_KEY", "")
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_ORG_KEY", "")  # NewsAPI.org
        self.thenewsapi_key = thenewsapi_key or os.environ.get("THENEWSAPI_KEY", "")  # TheNewsAPI.com
        self.nyt_requests_today = 0  # Track daily NYT API usage
        self.gnews_requests_today = 0  # Track daily GNews API usage (100/day limit)
        self.newsapi_requests_today = 0  # Track NewsAPI.org usage
        self.thenewsapi_requests_today = 0  # Track TheNewsAPI.com usage
        
    def enhance_database(self) -> pd.DataFrame:
        """
        Main enhancement process with user input
        """
        print("HISTORICAL NEWS DATABASE ENHANCER")
        print("=" * 50)
        print("This tool adds historical data to your existing GPT-filtered news database.")
        print("It collects month-by-month until minimum article targets are met.")
        print()
        
        # Get user inputs
        start_year, start_month, end_year, end_month, min_articles_per_month = self._get_user_inputs()
        
        # Load existing database
        existing_df = self._load_existing_database()
        print(f"Loaded existing database: {len(existing_df)} articles")
        
        # Collect historical data month by month
        new_articles = self._collect_historical_data(
            start_year, start_month, end_year, end_month, min_articles_per_month
        )
        
        # Convert new articles to DataFrame - they're already filtered month by month
        print(f"\nTotal filtered articles collected: {len(new_articles)}")
        
        if not new_articles:
            print("No new articles collected.")
            return existing_df
            
        new_df = pd.DataFrame(new_articles)
        
        # CHECKPOINT: Save filtered articles 
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = f"checkpoint_filtered_historical_{timestamp}.csv"
        new_df.to_csv(checkpoint_file, index=False)
        print(f"Checkpoint saved: {checkpoint_file}")
        
        # Articles are already GPT filtered, so use them directly
        filtered_new_df = new_df
            
        # Combine with existing database
        combined_df = self._merge_databases(existing_df, filtered_new_df)
        
        # Save enhanced database
        enhanced_file = f"enhanced_investment_news_{timestamp}.csv"
        combined_df.to_csv(enhanced_file, index=False)
        
        print(f"\nENHANCEMENT COMPLETE!")
        print(f"Enhanced database saved: {enhanced_file}")
        print(f"Original articles: {len(existing_df)}")
        print(f"New articles added: {len(filtered_new_df)}")
        print(f"Total articles: {len(combined_df)}")
        
        return combined_df
    
    def _get_user_inputs(self) -> Tuple[int, int, int, int, int]:
        """Get collection parameters from user"""
        print("COLLECTION PARAMETERS")
        print("-" * 30)
        
        # Default to historical period (2020-2024)
        print("Historical data collection (pre-2025)")
        print("Default range: January 2020 to December 2024")
        print()
        
        # Start date
        try:
            start_year = int(input("Enter start year [2020]: ") or "2020")
            start_month = int(input("Enter start month (1-12) [1]: ") or "1")
        except ValueError:
            print("Using defaults: January 2020")
            start_year, start_month = 2020, 1
            
        # End date
        try:
            end_year = int(input("Enter end year [2024]: ") or "2024")
            end_month = int(input("Enter end month (1-12) [12]: ") or "12")
        except ValueError:
            print("Using defaults: December 2024")
            end_year, end_month = 2024, 12
            
        # Minimum articles per month (AFTER FILTERING)
        try:
            min_articles = int(input("Minimum FILTERED articles per month [20]: ") or "20")
        except ValueError:
            print("Using default: 20 filtered articles per month")
            min_articles = 20
            
        # Validate dates
        if start_year >= 2025 or end_year >= 2025:
            print("WARNING: Collecting pre-2025 data only. Adjusting end date to December 2024.")
            end_year, end_month = 2024, 12
            
        print(f"\nCollection Plan:")
        print(f"Period: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        print(f"Target: {min_articles} FILTERED articles per month minimum")
        print(f"Note: Will collect raw articles until GPT filtering produces {min_articles} quality articles")
        print()
        
        return start_year, start_month, end_year, end_month, min_articles
    
    def _load_existing_database(self) -> pd.DataFrame:
        """Load the existing GPT-filtered database"""
        # Look for the existing filtered database
        possible_files = [
            "news_collection_archive/gpt_filtered/gpt_filtered_investment_news_20251115_042932.csv",
            "gpt_filtered_investment_news_20251115_042932.csv",
            "../news_collection_archive/gpt_filtered/gpt_filtered_investment_news_20251115_042932.csv"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                print(f"Loading existing database: {file_path}")
                return pd.read_csv(file_path)
                
        print("WARNING: Could not find existing filtered database. Creating new database.")
        return pd.DataFrame()
    
    def _collect_historical_data(self, start_year: int, start_month: int, 
                                end_year: int, end_month: int, 
                                min_articles_per_month: int) -> List[Dict]:
        """
        Collect historical data month by month until targets are met
        """
        all_articles = []
        current_year, current_month = start_year, start_month
        
        # FAST API KEYWORDS: Optimized for NewsAPI, TheNewsAPI, GNews
        general_keywords = [
            "stock", "market", "investment", "trading", "economy", "finance", 
            "earnings", "revenue", "profit", "IPO", "merger", "acquisition",
            "inflation", "recession", "growth", "decline", "volatility",
            "portfolio", "dividend", "cryptocurrency", "bonds", "commodity"
        ]
        
        total_months = 0
        while True:
            # Calculate month boundaries
            _, days_in_month = monthrange(current_year, current_month)
            month_start = f"{current_year}-{current_month:02d}-01"
            month_end = f"{current_year}-{current_month:02d}-{days_in_month}"
            
            total_months += 1
            print(f"\n📅 MONTH {total_months}: {current_year}-{current_month:02d}")
            print(f"Target: {min_articles_per_month} FILTERED articles minimum")
            print("-" * 40)
            
            month_articles = []
            filtered_count = 0
            
            # Collect articles for this month until we have enough filtered articles
            # Use general keywords for fast APIs only (SKIP NYT - rate limited)
            all_keywords_used = set()
            
            # Use general keywords for fast APIs (NewsAPI, TheNewsAPI, GNews)
            for keyword_idx, keyword in enumerate(general_keywords, 1):
                if keyword in all_keywords_used:
                    continue
                all_keywords_used.add(keyword)
                
                print(f"  [{keyword_idx:2d}] Keyword: '{keyword}'")
                
                try:
                    # FAST API ORDER: Only working APIs (skip TheNewsAPI - 402 payment required)
                    keyword_articles = []
                    
                    # 1. NewsAPI.org (1000/day, NO minute limits) - PRIMARY
                    if self.newsapi_requests_today < 950:
                        try:
                            newsapi_articles = self._fetch_newsapi_articles(keyword, month_start, month_end)
                            keyword_articles.extend(newsapi_articles)
                            print(f"    NewsAPI: {len(newsapi_articles)} articles")
                        except Exception as newsapi_error:
                            print(f"    NewsAPI error: {newsapi_error}")
                    
                    # 2. GNews (100/day, NO minute limits) - SECONDARY
                    if self.gnews_requests_today < 85:
                        try:
                            gnews_articles = self._fetch_gnews_articles_single(keyword, month_start, month_end)
                            keyword_articles.extend(gnews_articles)
                            print(f"    GNews: {len(gnews_articles)} articles")
                        except Exception as gnews_error:
                            print(f"    GNews error: {gnews_error}")
                    
                    # SKIP: TheNewsAPI (402 payment required error)
                    # SKIP: NYT (rate limited)
                    
                    month_articles.extend(keyword_articles)
                    print(f"    Total raw collected: {len(keyword_articles)} articles")
                    
                    # Check if we've met the filtered target with working APIs only
                    unique_articles = self._deduplicate_articles(month_articles)
                    if unique_articles:
                        temp_df = pd.DataFrame(unique_articles)
                        filtered_df = self._filter_articles_with_gpt_batch(temp_df)
                        filtered_count = len(filtered_df)
                        
                        print(f"    Month total: {len(unique_articles)} raw → {filtered_count} filtered")
                        
                        if filtered_count >= min_articles_per_month:
                            print(f"    ✅ Filtered target reached!")
                            all_articles.extend(filtered_df.to_dict('records'))
                            break
                        
                except Exception as e:
                    print(f"    ❌ Error with '{keyword}': {e}")
                    continue
            
            # If we didn't reach the target, save what we have
            if filtered_count < min_articles_per_month:
                print(f"    ⚠️ Only {filtered_count} filtered articles found for {current_year}-{current_month:02d}")
                if month_articles:
                    unique_articles = self._deduplicate_articles(month_articles)
                    temp_df = pd.DataFrame(unique_articles)
                    filtered_df = self._filter_articles_with_gpt_batch(temp_df)
                    all_articles.extend(filtered_df.to_dict('records'))
            
            print(f"✅ Month {current_year}-{current_month:02d} complete: {filtered_count} filtered articles")
            
            # IMMEDIATE SAVE: Save progress after each month
            if all_articles:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                monthly_save_file = f"historical_progress_{current_year}_{current_month:02d}_{timestamp}.csv"
                temp_df = pd.DataFrame(all_articles)
                temp_df.to_csv(monthly_save_file, index=False)
                print(f"    💾 Progress saved: {monthly_save_file} ({len(all_articles)} total articles)")
            
            # Move to next month
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1
                
            # Check if we've reached the end date
            if current_year > end_year or (current_year == end_year and current_month > end_month):
                break
                
        print(f"\n🎉 HISTORICAL COLLECTION COMPLETE!")
        print(f"Total months processed: {total_months}")
        print(f"Total FILTERED articles collected: {len(all_articles)}")
        print(f"Note: All articles have been GPT-filtered month by month")
        
        return all_articles
    
    def _fetch_gnews_articles_single(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch articles from GNews API using SINGLE optimized request per keyword"""
        articles = []
        
        # Check daily limit
        if self.gnews_requests_today >= 90:
            print(f"      GNews daily limit reached ({self.gnews_requests_today}/100)")
            return articles
        
        try:
            # Single optimized request with maximum articles
            params = {
                'q': f'"{keyword}" AND (stock OR market OR investment OR growth OR decline)',
                'lang': 'en',
                'country': 'us',
                'max': 100,  # Maximum articles per request
                'apikey': self.gnews_api_key,
                'from': start_date,
                'to': end_date,
                'sortby': 'relevance'
            }
            
            # Make API request
            response = self._make_api_request("https://gnews.io/api/v4/search", params)
            self.gnews_requests_today += 1
            print(f"      GNews requests today: {self.gnews_requests_today}/100")
            
            if response and response.status_code == 200:
                data = response.json()
                
                if 'articles' in data and data['articles']:
                    gnews_articles = data['articles']
                    
                    for article in gnews_articles:
                        articles.append({
                            'headline': article.get('title', ''),
                            'pub_date': article.get('publishedAt', ''),
                            'snippet': article.get('description', ''),
                            'abstract': article.get('description', ''),
                            'web_url': article.get('url', ''),
                            'section': 'Business',
                            'news_desk': 'General',
                            'source': f"GNews - {article.get('source', {}).get('name', 'Unknown')}",
                            'keyword': keyword,
                            'fetch_date': datetime.now().isoformat(),
                            'article_id': f"gnews_single_{keyword}_{len(articles)}"
                        })
                        
                    print(f"      GNews found: {len(gnews_articles)} articles")
                else:
                    print(f"      No articles found for '{keyword}'")
                    
            elif response.status_code == 429:
                print(f"      GNews rate limit hit")
                time.sleep(10)
            else:
                print(f"      GNews API error {response.status_code}")
                
        except Exception as e:
            print(f"      GNews API error: {e}")
            
        return articles
    
    def _fetch_gnews_articles_multiple(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch articles from GNews API using multiple smaller requests for better success rate"""
        all_articles = []
        
        # Multiple search variations for better coverage
        search_queries = [
            f'"{keyword}"',
            f'"{keyword}" AND finance',
            f'"{keyword}" AND investment', 
            f'"{keyword}" AND stock',
            f'"{keyword}" AND market'
        ]
        
        for query_idx, query in enumerate(search_queries, 1):
            try:
                print(f"      GNews query {query_idx}/5: {query[:30]}...")
                
                # Smaller requests (10 articles each) for better reliability
                params = {
                    'q': query,
                    'lang': 'en',
                    'country': 'us',
                    'max': 10,  # Much smaller requests
                    'apikey': self.gnews_api_key,
                    'from': start_date,
                    'to': end_date,
                    'sortby': 'relevance'
                }
                
                # Make API request
                response = self._make_api_request("https://gnews.io/api/v4/search", params)
                
                if response and response.status_code == 200:
                    data = response.json()
                    
                    if 'articles' in data and data['articles']:
                        query_articles = data['articles']
                        
                        for article in query_articles:
                            all_articles.append({
                                'headline': article.get('title', ''),
                                'pub_date': article.get('publishedAt', ''),
                                'snippet': article.get('description', ''),
                                'abstract': article.get('description', ''),
                                'web_url': article.get('url', ''),
                                'section': 'Business',
                                'news_desk': 'General',
                                'source': f"GNews - {article.get('source', {}).get('name', 'Unknown')}",
                                'keyword': keyword,
                                'fetch_date': datetime.now().isoformat(),
                                'article_id': f"gnews_multi_{keyword}_{query_idx}_{len(all_articles)}"
                            })
                            
                        print(f"        Found: {len(query_articles)} articles")
                    else:
                        print(f"        No articles for this query")
                        
                elif response.status_code == 429:
                    print(f"        GNews rate limit hit, waiting...")
                    time.sleep(10)
                    continue
                else:
                    print(f"        API error {response.status_code}")
                    
                # Small delay between queries
                time.sleep(2)
                
            except Exception as e:
                print(f"        Error with query {query_idx}: {e}")
                continue
                
        # Remove duplicates by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get('web_url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
                
        print(f"      GNews total: {len(all_articles)} collected → {len(unique_articles)} unique")
        return unique_articles
    
    def _fetch_gnews_articles(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch articles from GNews API for a specific keyword and month"""
        base_url = "https://gnews.io/api/v4/search"
        articles = []
        
        try:
            # GNews API parameters
            params = {
                'q': f'"{keyword}" AND (finance OR investment OR stock OR market OR business OR earnings)',
                'lang': 'en',
                'country': 'us',
                'max': 100,  # GNews allows up to 100 articles per request
                'apikey': self.gnews_api_key,
                'from': start_date,
                'to': end_date,
                'sortby': 'relevance'
            }
            
            # Make API request
            response = self._make_api_request(base_url, params)
            
            if response and response.status_code == 200:
                data = response.json()
                
                if 'articles' in data:
                    gnews_articles = data['articles']
                    
                    for article in gnews_articles:
                        articles.append({
                            'headline': article.get('title', ''),
                            'pub_date': article.get('publishedAt', ''),
                            'snippet': article.get('description', ''),
                            'abstract': article.get('description', ''),
                            'web_url': article.get('url', ''),
                            'section': 'Business',  # GNews doesn't provide section
                            'news_desk': 'General',
                            'source': f"GNews - {article.get('source', {}).get('name', 'Unknown')}",
                            'keyword': keyword,
                            'fetch_date': datetime.now().isoformat(),
                            'article_id': f"gnews_{keyword}_{len(articles)}"
                        })
                        
            elif response.status_code == 429:
                print(f"      GNews rate limit hit")
                time.sleep(5)
                
        except Exception as e:
            print(f"      GNews API error: {e}")
            
        return articles
    
    def _fetch_newsapi_articles(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch articles from NewsAPI.org (1000 requests/day) - LIMITED TO RECENT DATES"""
        articles = []
        
        # Check if the date range is within NewsAPI's limitation (past 30 days from now)
        from datetime import datetime, timedelta
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        request_start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if request_start_date < thirty_days_ago:
            print(f"      NewsAPI: Skipping historical date {start_date} (free plan limited to past 30 days)")
            return articles
        
        if self.newsapi_requests_today >= 950:
            print(f"      NewsAPI daily limit reached ({self.newsapi_requests_today}/1000)")
            return articles
        
        try:
            # NewsAPI.org parameters
            params = {
                'q': f'"{keyword}" AND (stock OR market OR investment OR growth OR decline)',
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 100,  # Maximum articles per request
                'apiKey': self.newsapi_key,
                'from': start_date,
                'to': end_date
            }
            
            # Enhanced headers for NewsAPI
            headers = {
                'User-Agent': 'HistoricalNewsCollector/1.0 (Python/requests)',
                'Accept': 'application/json',
                'X-API-Key': self.newsapi_key,
                'Authorization': f'Bearer {self.newsapi_key}'
            }
            
            response = requests.get("https://newsapi.org/v2/everything", params=params, headers=headers, timeout=30)
            self.newsapi_requests_today += 1
            print(f"      NewsAPI requests today: {self.newsapi_requests_today}/1000")
            
            if response and response.status_code == 200:
                data = response.json()
                
                if 'articles' in data and data['articles']:
                    for article in data['articles']:
                        articles.append({
                            'headline': article.get('title', ''),
                            'pub_date': article.get('publishedAt', ''),
                            'snippet': article.get('description', ''),
                            'abstract': article.get('description', ''),
                            'web_url': article.get('url', ''),
                            'section': 'Business',
                            'news_desk': 'General',
                            'source': f"NewsAPI - {article.get('source', {}).get('name', 'Unknown')}",
                            'keyword': keyword,
                            'fetch_date': datetime.now().isoformat(),
                            'article_id': f"newsapi_{keyword}_{len(articles)}"
                        })
                        
                    print(f"      NewsAPI found: {len(data['articles'])} articles")
                else:
                    print(f"      No NewsAPI articles for '{keyword}'")
                    
            elif response.status_code == 426:
                # This is actually the date limitation error
                print(f"      NewsAPI: Date range limitation - can only access past 30 days")
            elif response.status_code == 429:
                print(f"      NewsAPI rate limit hit")
                time.sleep(10)
            elif response.status_code == 401:
                print(f"      NewsAPI authentication error - check API key")
            else:
                print(f"      NewsAPI error {response.status_code}")
                if response:
                    print(f"      Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"      NewsAPI error: {e}")
            
        return articles
    
    def _fetch_thenewsapi_articles(self, keyword: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch articles from TheNewsAPI.com (150 requests/day)"""
        articles = []
        
        if self.thenewsapi_requests_today >= 140:
            print(f"      TheNewsAPI daily limit reached ({self.thenewsapi_requests_today}/150)")
            return articles
        
        try:
            # TheNewsAPI.com parameters
            params = {
                'search': f'"{keyword}" stock market investment growth decline',
                'language': 'en',
                'sort': 'relevance',
                'limit': 100,  # Maximum articles per request
                'api_token': self.thenewsapi_key,
                'published_after': start_date,
                'published_before': end_date
            }
            
            response = self._make_api_request("https://api.thenewsapi.com/v1/news/all", params)
            self.thenewsapi_requests_today += 1
            print(f"      TheNewsAPI requests today: {self.thenewsapi_requests_today}/150")
            
            if response and response.status_code == 200:
                data = response.json()
                
                if 'data' in data and data['data']:
                    for article in data['data']:
                        articles.append({
                            'headline': article.get('title', ''),
                            'pub_date': article.get('published_at', ''),
                            'snippet': article.get('description', ''),
                            'abstract': article.get('description', ''),
                            'web_url': article.get('url', ''),
                            'section': 'Business',
                            'news_desk': 'General',
                            'source': f"TheNewsAPI - {article.get('source', 'Unknown')}",
                            'keyword': keyword,
                            'fetch_date': datetime.now().isoformat(),
                            'article_id': f"thenewsapi_{keyword}_{len(articles)}"
                        })
                        
                    print(f"      TheNewsAPI found: {len(data['data'])} articles")
                else:
                    print(f"      No TheNewsAPI articles for '{keyword}'")
                    
            elif response.status_code == 402:
                print(f"      TheNewsAPI error 402: Payment required - API plan needs upgrade")
                return articles  # Return empty, don't continue trying
            elif response.status_code == 429:
                print(f"      TheNewsAPI rate limit hit")
                time.sleep(10)
            else:
                print(f"      TheNewsAPI error {response.status_code}")
                
        except Exception as e:
            print(f"      TheNewsAPI error: {e}")
            
        return articles
    
    def _fetch_month_articles(self, keyword: str, start_date: str, end_date: str, max_pages: int = 20) -> List[Dict]:
        """Fetch articles for a specific keyword and month"""
        base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        articles = []
        
        # Convert dates for API
        start_api = start_date.replace('-', '')
        end_api = end_date.replace('-', '')
        
        # Fetch multiple pages for comprehensive coverage
        for page in range(max_pages):
            # Check daily limit before making request
            if self.nyt_requests_today >= 450:  # Leave buffer for safety
                print(f"      ⚠️ NYT daily limit approaching ({self.nyt_requests_today}/500), stopping collection")
                break
                
            try:
                params = {
                    'q': keyword,
                    'begin_date': start_api,
                    'end_date': end_api,
                    'sort': 'newest',
                    'api-key': self.nyt_api_key,
                    'page': page,
                    'fl': 'headline,pub_date,snippet,web_url,section_name,news_desk,abstract'
                }
                
                # API call with retry logic
                response = self._make_api_request(base_url, params)
                self.nyt_requests_today += 1  # Increment request counter
                print(f"      NYT requests today: {self.nyt_requests_today}/500")
                
                if response and response.status_code == 200:
                    data = response.json()
                    
                    if 'response' in data and 'docs' in data['response']:
                        page_articles = data['response']['docs']
                        
                        if not page_articles:
                            break  # No more articles
                            
                        for article in page_articles:
                            articles.append({
                                'headline': article.get('headline', {}).get('main', ''),
                                'pub_date': article.get('pub_date', ''),
                                'snippet': article.get('snippet', ''),
                                'abstract': article.get('abstract', ''),
                                'web_url': article.get('web_url', ''),
                                'section': article.get('section_name', ''),
                                'news_desk': article.get('news_desk', ''),
                                'source': 'NY Times',
                                'keyword': keyword,
                                'fetch_date': datetime.now().isoformat(),
                                'article_id': f"hist_{keyword}_{page}_{len(articles)}"
                            })
                            
                        # If fewer than 10 articles, we've reached the end
                        if len(page_articles) < 10:
                            break
                            
                elif response.status_code == 429:
                    print(f"    Rate limit hit, waiting 60 seconds...")  # Extended wait for 429 errors
                    time.sleep(60)
                    continue
                else:
                    break
                    
                # Rate limiting for NYT API: 5 requests per minute = 12 seconds, but we're backup so faster
                time.sleep(6)  # Reduced from 12s since we're using it less
                
            except Exception as e:
                print(f"      Error on page {page}: {e}")
                break
                
        return articles
    
    def _make_api_request(self, url: str, params: dict, max_retries: int = 3):
        """Make API request with retry logic and proper headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=30)
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    raise e
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by headline"""
        seen_headlines = set()
        unique_articles = []
        
        for article in articles:
            headline = article.get('headline', '').strip()
            if headline and headline not in seen_headlines:
                seen_headlines.add(headline)
                unique_articles.append(article)
                
        return unique_articles
    
    def _filter_articles_with_gpt_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles with GPT for investment relevance - single batch processing"""
        if df.empty:
            return df
            
        # Remove duplicates first
        initial_count = len(df)
        df = df.drop_duplicates(subset=['headline'], keep='first')
        removed_dupes = initial_count - len(df)
        if removed_dupes > 0:
            print(f"      Removed {removed_dupes} duplicate headlines")
        
        # Process as single batch for immediate filtering
        articles_text = ""
        for idx, row in df.iterrows():
            headline = row.get('headline', '')[:100]
            snippet = row.get('snippet', '')[:150]
            articles_text += f"{idx}. {headline} - {snippet}\n"
            
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are an EXTREMELY STRICT financial news curator for SERIOUS STOCK INVESTMENT ANALYSIS. Only include articles that directly impact investment decisions.

STRICT CRITERIA - INCLUDE ONLY IF:
1. DIRECT STOCK/MARKET IMPACT: Stock prices, market movements, earnings, financial performance
2. MAJOR COMPANY NEWS: M&A, leadership changes, product launches for public companies
3. ECONOMIC INDICATORS: Fed decisions, inflation, GDP, unemployment affecting markets
4. SECTOR ANALYSIS: Industry trends affecting multiple stocks
5. STARTUP/IPO NEWS: Funding rounds, IPOs, acquisition potential
6. FINANCIAL TECHNOLOGY: Fintech disrupting financial institutions

ABSOLUTELY EXCLUDE:
- General tech news without financial angle
- Politics unless market-moving
- Opinion pieces without concrete data
- Entertainment, sports, lifestyle, health, crime, weather
- General business without stock relevance

BE EXTREMELY SELECTIVE. Only 15-30% should pass.

Return ONLY the numbers (indices) of relevant articles, separated by commas."""},
                    {"role": "user", "content": f"Filter these articles for STRICT investment relevance:\n\n{articles_text}"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse response
            relevant_indices = []
            try:
                response_text = response.choices[0].message.content.strip()
                if response_text and response_text != "None":
                    if ',' in response_text:
                        relevant_indices = [int(x.strip()) for x in response_text.split(',') if x.strip().isdigit()]
                    else:
                        numbers = response_text.replace(',', ' ').split()
                        relevant_indices = [int(x) for x in numbers if x.isdigit()]
            except:
                relevant_indices = []
            
            # Filter to relevant articles only
            if relevant_indices:
                filtered_df = df.loc[df.index.intersection(relevant_indices)].copy()
                print(f"      GPT filtered: {len(filtered_df)}/{len(df)} articles ({len(filtered_df)/len(df)*100:.1f}%)")
                return filtered_df
            else:
                print(f"      No articles passed GPT filter")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"      GPT error: {e}")
            return pd.DataFrame()  # Return empty if GPT fails
    
    def _filter_articles_with_gpt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles with GPT for investment relevance"""
        print("Applying strict GPT filtering to new articles...")
        
        if df.empty:
            return df
            
        # Remove duplicates first
        initial_count = len(df)
        df = df.drop_duplicates(subset=['headline'], keep='first')
        print(f"  Removed {initial_count - len(df)} duplicate headlines")
        
        # Process in batches
        batch_size = 30
        filtered_articles = []
        total_batches = (len(df) - 1) // batch_size + 1
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")
            
            # Create batch prompt
            articles_text = ""
            for idx, row in batch.iterrows():
                headline = row.get('headline', '')[:100]
                snippet = row.get('snippet', '')[:150]
                articles_text += f"{idx}. {headline} - {snippet}\n"
                
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """You are an EXTREMELY STRICT financial news curator for SERIOUS STOCK INVESTMENT ANALYSIS. Only include articles that directly impact investment decisions.

STRICT CRITERIA - INCLUDE ONLY IF:
1. DIRECT STOCK/MARKET IMPACT: Stock prices, market movements, earnings, financial performance
2. MAJOR COMPANY NEWS: M&A, leadership changes, product launches for public companies
3. ECONOMIC INDICATORS: Fed decisions, inflation, GDP, unemployment affecting markets
4. SECTOR ANALYSIS: Industry trends affecting multiple stocks
5. STARTUP/IPO NEWS: Funding rounds, IPOs, acquisition potential
6. FINANCIAL TECHNOLOGY: Fintech disrupting financial institutions

ABSOLUTELY EXCLUDE:
- General tech news without financial angle
- Politics unless market-moving
- Opinion pieces without concrete data
- Entertainment, sports, lifestyle, health, crime, weather
- General business without stock relevance
- Individual profiles unless major company executives

BE EXTREMELY SELECTIVE. Only 15-30% should pass.

Return ONLY the numbers (indices) of relevant articles, separated by commas."""},
                        {"role": "user", "content": f"Filter these articles for STRICT investment relevance:\n\n{articles_text}"}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                
                # Parse response
                relevant_indices = []
                try:
                    response_text = response.choices[0].message.content.strip()
                    if response_text and response_text != "None":
                        if ',' in response_text:
                            relevant_indices = [int(x.strip()) for x in response_text.split(',') if x.strip().isdigit()]
                        else:
                            numbers = response_text.replace(',', ' ').split()
                            relevant_indices = [int(x) for x in numbers if x.isdigit()]
                except:
                    relevant_indices = []
                
                # Add relevant articles
                relevant_count = 0
                for idx in relevant_indices:
                    if idx in batch.index:
                        filtered_articles.append(batch.loc[idx].to_dict())
                        relevant_count += 1
                        
                print(f"    Retained {relevant_count}/{len(batch)} articles")
                
            except Exception as e:
                print(f"    GPT error: {e}")
                # Keep all articles if GPT fails
                for idx, row in batch.iterrows():
                    filtered_articles.append(row.to_dict())
                    
            time.sleep(1)  # Rate limiting
            
        if filtered_articles:
            result_df = pd.DataFrame(filtered_articles)
            print(f"  ✅ GPT filtering: {len(result_df)}/{len(df)} articles retained ({len(result_df)/len(df)*100:.1f}%)")
            return result_df
        else:
            return pd.DataFrame()
    
    def _merge_databases(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge existing and new databases, removing duplicates"""
        print("\nMerging databases...")
        
        if existing_df.empty:
            return new_df
        if new_df.empty:
            return existing_df
            
        # Add metadata to new articles
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        new_df['filtered_date'] = timestamp
        new_df['source_database'] = 'historical_enhancement'
        new_df['filter_type'] = 'strict_investment_gpt'
        
        # Combine databases
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates by headline
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['headline'], keep='first')
        duplicates_removed = initial_count - len(combined_df)
        
        # Sort by publication date
        if 'pub_date_standardized' not in combined_df.columns and 'pub_date' in combined_df.columns:
            combined_df['pub_date_standardized'] = pd.to_datetime(combined_df['pub_date'], errors='coerce')
            
        combined_df = combined_df.sort_values('pub_date_standardized', na_last=True)
        
        print(f"  Combined: {len(existing_df)} + {len(new_df)} = {len(combined_df)} articles")
        print(f"  Removed {duplicates_removed} duplicates")
        
        return combined_df


def main():
    """Main execution function"""
    print("HISTORICAL NEWS DATABASE ENHANCER")
    print("Adds pre-2025 historical data to existing filtered database")
    print("=" * 60)
    
    # Hardcoded API keys - MULTI-API SETUP
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    nyt_key = os.environ.get("NYT_API_KEY", "")
    gnews_key = os.environ.get("GNEWS_API_KEY", "")
    newsapi_key = os.environ.get("NEWSAPI_ORG_KEY", "")  # NewsAPI.org
    thenewsapi_key = os.environ.get("THENEWSAPI_KEY", "")  # TheNewsAPI.com
    
    try:
        # Initialize enhancer with all 4 APIs
        enhancer = HistoricalNewsEnhancer(
            openai_api_key=openai_key, 
            nyt_api_key=nyt_key, 
            gnews_api_key=gnews_key,
            newsapi_key=newsapi_key,
            thenewsapi_key=thenewsapi_key
        )
        
        # Run enhancement process
        enhanced_df = enhancer.enhance_database()
        
        if not enhanced_df.empty:
            print(f"\n🎉 SUCCESS! Enhanced database created with {len(enhanced_df)} total articles!")
        else:
            print("\n❌ No enhancement performed.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Enhancement cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during enhancement: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()