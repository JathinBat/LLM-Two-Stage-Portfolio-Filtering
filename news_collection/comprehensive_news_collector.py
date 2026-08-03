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

"""
Practical Finance & Tech News Data Collector

This module fetches finance and technology news from NY Times API:
- Configurable date range (default: last 6 months)
- Keywords: finance, tech, technology, financial, investment, stock, market
- GPT-based article filtering for relevance
- Precise timestamp standardization
- CSV output with all metadata

Usage:
    python comprehensive_news_collector.py
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
warnings.filterwarnings('ignore')

class PracticalNewsCollector:
    """
    Practical finance and tech news collector
    """
    
    def __init__(self, openai_api_key: str, nyt_api_key: str = None):
        """Initialize with API keys"""
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.nyt_api_key = nyt_api_key or os.environ.get("NYT_API_KEY", "")
        
    def collect_news_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Collect finance and tech news data from NY Times API
        Default: Last 6 months of data
        """
        # Set reasonable defaults
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_dt = datetime.now() - relativedelta(years=1)  # Extended to 1 year for much more data
            start_date = start_dt.strftime('%Y-%m-%d')
            
        print(f"FINANCE & TECH NEWS COLLECTION")
        print(f"Period: {start_date} to {end_date}")
        print("Source: NY Times API")
        print("Keywords: Comprehensive finance, tech, and business coverage")
        print("=" * 80)
        
        # Collect from NY Times only (reliable and sufficient)
        print("\nNY Times API Collection")
        articles = self._fetch_nytimes_articles(start_date, end_date)
        
        if not articles:
            print("No articles collected")
            return pd.DataFrame()
            
        print(f"Total articles collected: {len(articles)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(articles)
        
        # CHECKPOINT: Save all collected articles before GPT filtering
        print("\nCheckpoint: Saving all collected articles...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = f"checkpoint_all_articles_{timestamp}.csv"
        df.to_csv(checkpoint_file, index=False)
        print(f"Checkpoint saved: {checkpoint_file} ({len(df)} articles)")
        
        # GPT-based filtering for relevance
        print("\nGPT-based Relevance Filtering")
        try:
            df_filtered = self._filter_articles_with_gpt(df)
        except Exception as gpt_error:
            print(f"WARNING: GPT filtering failed: {gpt_error}")
            print("Proceeding without GPT filtering...")
            df_filtered = self._filter_articles_with_gpt(df, skip_gpt=True)
        
        # Standardize timestamps and save
        print("\nTimestamp Standardization & Export")
        df_final = self._standardize_and_export(df_filtered)
        
        return df_final

    def _fetch_nytimes_articles(self, start_date: str, end_date: str) -> List[Dict]:
        """Fetch articles from NY Times API for the specified date range"""
        print("Fetching from NY Times API...")
        
        base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        all_articles = []
        
        # Expanded finance and tech keywords for comprehensive coverage
        # FULL MODE: Using all keywords for maximum collection
        test_mode = False  # Set to False for full collection
        
        if test_mode:
            keywords = ["finance", "tech", "investment", "stock"]  # Quick test with 4 keywords
            print("TEST MODE: Using limited keywords for quick testing")
        else:
            keywords = [
                # Core finance keywords
                "finance", "financial", "investment", "stock", "market", "banking", "trading", "economy", "economic",
                # Technology keywords  
                "tech", "technology", "AI", "artificial intelligence", "software", "cybersecurity", "cloud computing", 
                "data science", "machine learning", "semiconductors", "automation", "robotics", "quantum computing",
                # Business and corporate keywords
                "business", "startup", "venture capital", "merger", "acquisition", "earnings", "revenue", "IPO",
                "digital transformation", "corporate", "enterprise", "innovation", "growth", "profits",
                # Emerging sectors
                "cryptocurrency", "blockchain", "fintech", "biotech", "renewable energy", "electric vehicles",
                "clean energy", "sustainable", "ESG", "green technology", "solar", "battery",
                # Market indicators
                "Federal Reserve", "interest rates", "inflation", "GDP", "unemployment", "consumer spending",
                "retail sales", "manufacturing", "housing market", "commodities", "oil prices"
            ]
            print(f"FULL MODE: Using {len(keywords)} comprehensive keywords for maximum collection")
        
        # Convert dates for API (YYYYMMDD format)
        start_api = start_date.replace('-', '')
        end_api = end_date.replace('-', '')
        
        for keyword in keywords:
            print(f"  Searching for: '{keyword}'")
            
            try:
                # Fetch pages for each keyword (increased for maximum collection)
                pages_to_fetch = 2 if test_mode else 15  # Increased from 10 to 15 pages
                for page in range(pages_to_fetch):  # Up to 150 articles per keyword
                    params = {
                        'q': keyword,
                        'begin_date': start_api,
                        'end_date': end_api,
                        'sort': 'newest',
                        'api-key': self.nyt_api_key,
                        'page': page,
                        'fl': 'headline,pub_date,snippet,web_url,section_name,news_desk,abstract'
                    }
                    
                    # Retry logic for NY Times API
                    max_retries = 3
                    response = None
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(base_url, params=params, timeout=30)
                            break  # Success
                        except Exception as api_error:
                            if attempt < max_retries - 1:
                                wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                                print(f"      API error, retrying in {wait_time}s... ({api_error})")
                                time.sleep(wait_time)
                            else:
                                print(f"      Failed after {max_retries} attempts: {api_error}")
                                raise api_error
                    
                    if response and response.status_code == 200:
                        data = response.json()
                        
                        if 'response' in data and 'docs' in data['response']:
                            articles = data['response']['docs']
                            
                            if not articles:
                                break
                            
                            for article in articles:
                                all_articles.append({
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
                                    'article_id': f"nyt_{keyword}_{page}_{len(all_articles)}"
                                })
                            
                            print(f"    Page {page + 1}: {len(articles)} articles")
                            
                            # If fewer than 10 articles, we've reached the end
                            if len(articles) < 10:
                                break
                        else:
                            break
                    elif response.status_code == 429:
                        print(f"    Rate limit hit, waiting...")
                        time.sleep(12)
                        continue
                    else:
                        print(f"    ⚠️ API error {response.status_code}")
                        break
                        
                    # Rate limiting - NYT API allows 10 requests per minute
                    time.sleep(6)
                    
            except Exception as e:
                print(f"    ❌ Error fetching {keyword}: {e}")
                continue
                
        print(f"  ✅ NY Times collection complete: {len(all_articles)} articles")
        return all_articles

    def _standardize_and_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamps and export to CSV"""
        if df.empty:
            return df
            
        # Standardize timestamps
        def parse_timestamp(date_str):
            if pd.isna(date_str) or not date_str:
                return None
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return None
                
        df['pub_date_standardized'] = df['pub_date'].apply(parse_timestamp)
        
        # Remove articles with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=['pub_date_standardized'])
        print(f"  🗑️ Removed {initial_count - len(df)} articles with invalid dates")
        
        # Sort by publication date
        df = df.sort_values('pub_date_standardized')
        
        # Add formatted timestamps and metadata
        df['precise_timestamp'] = df['pub_date_standardized'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        df['year'] = df['pub_date_standardized'].dt.year
        df['month'] = df['pub_date_standardized'].dt.month
        df['day_of_week'] = df['pub_date_standardized'].dt.day_name()
        
        # Export to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"comprehensive_finance_tech_news_{timestamp}.csv"  # Updated filename for comprehensive collection
        df.to_csv(output_file, index=False)
        
        print(f"COMPREHENSIVE COLLECTION COMPLETE!")
        print(f"Dataset saved to: {output_file}")
        print(f"Final article count: {len(df)}")
        print(f"Collection scope: 2 years + 35+ keywords + GPT-4o-mini filtering")
        
        if not df.empty:
            print(f"📅 Date range: {df['pub_date_standardized'].min().date()} to {df['pub_date_standardized'].max().date()}")
            
            # Show source breakdown
            source_counts = df['source'].value_counts()
            print(f"📊 Articles by source: {dict(source_counts)}")
            
            # Show keyword breakdown
            keyword_counts = df['keyword'].value_counts()
            print(f"Articles by keyword: {dict(keyword_counts)}")
        
        return df

    def _filter_articles_with_gpt(self, df: pd.DataFrame, skip_gpt=False) -> pd.DataFrame:
        """Use GPT to filter articles for finance/tech relevance and remove duplicates"""
        print("Filtering articles with GPT for relevance...")
        
        if df.empty:
            return df
        
        # Remove obvious duplicates by headline first
        initial_count = len(df)
        df = df.drop_duplicates(subset=['headline'], keep='first')
        print(f"  Removed {initial_count - len(df)} duplicate headlines")
        
        if skip_gpt:
            print("  WARNING: Skipping GPT filtering due to API issues")
            print(f"  Keeping all {len(df)} articles without GPT filtering")
            return df
        initial_count = len(df)
        df = df.drop_duplicates(subset=['headline'], keep='first')
        print(f"  🗑️ Removed {initial_count - len(df)} duplicate headlines")
        
        # Process in batches for GPT filtering
        batch_size = 30  # Smaller batches for better processing
        filtered_articles = []
        
        total_batches = (len(df) - 1) // batch_size + 1
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")
            
            # Create batch prompt
            articles_text = ""
            for idx, row in batch.iterrows():
                headline = row.get('headline', '')[:100]  # Truncate long headlines
                snippet = row.get('snippet', '')[:150]    # Truncate long snippets
                articles_text += f"{idx}. {headline} - {snippet}\n"
                
            try:
                # Retry logic with exponential backoff for GPT API
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.openai_client.chat.completions.create(
                            model="gpt-4o-mini",  # Better model for improved filtering
                            messages=[
                                {"role": "system", "content": """You are an EXTREMELY STRICT financial news curator for SERIOUS STOCK INVESTMENT ANALYSIS. Only include articles that directly impact investment decisions.

STRICT CRITERIA - INCLUDE ONLY IF:
1. DIRECT STOCK/MARKET IMPACT: Article directly mentions stock prices, market movements, earnings, or financial performance
2. MAJOR COMPANY NEWS: Significant corporate developments (mergers, acquisitions, leadership changes, product launches) for publicly traded companies
3. ECONOMIC INDICATORS: Federal Reserve decisions, inflation data, GDP reports, unemployment that directly affect markets
4. SECTOR ANALYSIS: Tech sector growth, financial sector regulations, industry trends affecting multiple stocks
5. STARTUP/IPO NEWS: Only if discussing funding rounds, IPOs, or acquisition potential
6. FINANCIAL TECHNOLOGY: Only fintech innovations that could disrupt existing financial institutions

ABSOLUTELY EXCLUDE:
- General tech news without financial/business angle
- Politics unless directly market-moving (like trade policy)
- Opinion pieces or analysis without concrete data
- Entertainment, sports, lifestyle, health, crime, weather
- General business news without stock market relevance
- Articles about individuals unless they're major company executives
- Regulatory news unless it directly affects specific industries/stocks
- General economic news without clear market implications
- Any article that doesn't help make specific investment decisions

BE EXTREMELY SELECTIVE. Only 20-40% of articles should pass this filter.

Return ONLY the numbers (indices) of relevant articles, separated by commas.
Example: 1,3,5,7,12"""},
                                {"role": "user", "content": f"Filter these articles for STRICT investment relevance:\n\n{articles_text}"}
                            ],
                            temperature=0.1,  # Lower temperature for more consistent filtering
                            max_tokens=200    # Reduced since we expect fewer articles
                        )
                        break  # Success, exit retry loop
                        
                    except Exception as api_error:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                            print(f"    API attempt {attempt + 1} failed, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            raise api_error  # Re-raise after all retries exhausted
                
                # Parse GPT response
                relevant_indices = []
                try:
                    response_text = response.choices[0].message.content.strip()
                    if response_text and response_text != "None":
                        # Handle different response formats
                        if ',' in response_text:
                            relevant_indices = [int(x.strip()) for x in response_text.split(',') if x.strip().isdigit()]
                        else:
                            # Single number or space-separated
                            numbers = response_text.replace(',', ' ').split()
                            relevant_indices = [int(x) for x in numbers if x.isdigit()]
                            
                except Exception as parse_error:
                    print(f"    ⚠️ GPT response parsing error: {parse_error}")
                    # If parsing fails, keep all articles in this batch
                    relevant_indices = list(batch.index)
                
                # Add relevant articles
                relevant_count = 0
                for idx in relevant_indices:
                    if idx in batch.index:
                        filtered_articles.append(batch.loc[idx].to_dict())
                        relevant_count += 1
                        
                print(f"    ✅ Retained {relevant_count}/{len(batch)} articles from this batch")
                        
            except Exception as e:
                print(f"    ❌ GPT filtering error: {e}")
                # If GPT fails, keep all articles in this batch
                for idx, row in batch.iterrows():
                    filtered_articles.append(row.to_dict())
                
            # Rate limiting for OpenAI API
            time.sleep(1)
            
        # Convert back to DataFrame
        if filtered_articles:
            result_df = pd.DataFrame(filtered_articles)
            print(f"  ✅ GPT filtering complete: {len(result_df)}/{len(df)} articles retained ({len(result_df)/len(df)*100:.1f}%)")
            return result_df
        else:
            print("  ⚠️ No articles passed GPT filtering, returning original set")
            return df

    def _standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize publication timestamps to precise datetime format"""
        print("⏰ Standardizing timestamps...")
        
        def parse_timestamp(date_str):
            if pd.isna(date_str) or not date_str:
                return None
                
            try:
                # Try various datetime formats
                formats = [
                    '%Y-%m-%dT%H:%M:%S%z',  # ISO with timezone
                    '%Y-%m-%dT%H:%M:%SZ',   # ISO UTC
                    '%Y-%m-%dT%H:%M:%S',    # ISO without timezone
                    '%Y-%m-%d %H:%M:%S',    # Standard datetime
                    '%Y-%m-%d',             # Date only
                    '%a, %d %b %Y %H:%M:%S %Z',  # RFC format
                    '%d %b %Y %H:%M:%S',    # Alternative format
                    '%B %d, %Y',            # Month day, year
                    '%b %d, %Y',            # Short month day, year
                ]
                
                for fmt in formats:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                        
                # If all formats fail, try pandas auto-detection
                return pd.to_datetime(date_str, errors='coerce')
                
            except:
                return None
                
        df['pub_date_standardized'] = df['pub_date'].apply(parse_timestamp)
        
        # Remove articles with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=['pub_date_standardized'])
        print(f"  🗑️ Removed {initial_count - len(df)} articles with invalid dates")
        
        # Sort by publication date
        df = df.sort_values('pub_date_standardized')
        
        # Add precise timestamp string for CSV
        df['precise_timestamp'] = df['pub_date_standardized'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        df['year'] = df['pub_date_standardized'].dt.year
        df['month'] = df['pub_date_standardized'].dt.month
        df['day_of_week'] = df['pub_date_standardized'].dt.day_name()
        
        print(f"  ✅ Timestamp standardization complete")
        return df

    def _generate_summary_stats(self, df: pd.DataFrame, output_file: str):
        """Generate and save summary statistics"""
        print("\n📊 GENERATING SUMMARY STATISTICS")
        print("-" * 50)
        
        stats = {
            'collection_summary': {
                'total_articles': len(df),
                'date_range': {
                    'start': df['pub_date_standardized'].min().isoformat() if not df.empty else None,
                    'end': df['pub_date_standardized'].max().isoformat() if not df.empty else None
                },
                'sources': df['source'].value_counts().to_dict() if not df.empty else {},
                'keywords': df['keyword'].value_counts().to_dict() if not df.empty else {},
                'output_file': output_file,
                'collection_date': datetime.now().isoformat()
            }
        }
        
        if not df.empty:
            # Temporal distribution
            df['year_month'] = df['pub_date_standardized'].dt.to_period('M')
            monthly_counts = df['year_month'].value_counts().sort_index()
            
            stats['temporal_distribution'] = {
                'by_month': {str(period): count for period, count in monthly_counts.items()},
                'most_active_month': str(monthly_counts.idxmax()),
                'least_active_month': str(monthly_counts.idxmin()),
                'average_per_month': float(monthly_counts.mean())
            }
            
            # Source distribution
            stats['source_breakdown'] = df['source'].value_counts().to_dict()
            
            # Print summary
            print(f"📈 Total Articles: {len(df)}")
            print(f"📅 Date Range: {df['pub_date_standardized'].min().date()} to {df['pub_date_standardized'].max().date()}")
            print(f"📊 Sources: {dict(df['source'].value_counts())}")
            print(f"🏆 Most Active Month: {monthly_counts.idxmax()} ({monthly_counts.max()} articles)")
            print(f"📉 Least Active Month: {monthly_counts.idxmin()} ({monthly_counts.min()} articles)")
            
        # Save summary statistics
        stats_file = output_file.replace('.csv', '_summary.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
            
        print(f"📋 Summary statistics saved to: {stats_file}")

def main():
    """
    Main execution function
    """
    print("PRACTICAL NEWS COLLECTOR")
    print("Collecting finance and tech news with GPT filtering")
    print("=" * 60)
    
    # Hardcoded API keys
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    nyt_key = os.environ.get("NYT_API_KEY", "")  # Using the working NY Times API key
    
    # Initialize collector with hardcoded API keys
    collector = PracticalNewsCollector(openai_api_key=openai_key, nyt_api_key=nyt_key)
    
    try:
        # Run data collection (default: last 6 months)
        df_result = collector.collect_news_data()
        
        if not df_result.empty:
            print(f"\nSUCCESS! News dataset created!")
            print(f"Total articles: {len(df_result)}")
            print("CSV file created with standardized timestamps")
        else:
            print("No data collected. Please check API keys and network connection.")
            
    except Exception as e:
        print(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nCollection process complete!")


if __name__ == "__main__":
    main()

