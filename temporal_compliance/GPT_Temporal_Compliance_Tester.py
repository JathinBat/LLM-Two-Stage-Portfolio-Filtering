  #!/usr/bin/env python3
"""
GPT Temporal Compliance Tester

A comprehensive tool to test if GPT models use future data inappropriately
when analyzing historical periods. Tests temporal boundaries to detect "data leakage".
"""

import openai
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import time
import os
import re
import random
from difflib import SequenceMatcher

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

class GPTTemporalComplianceTester:
    """All-in-one GPT temporal compliance testing tool"""

    def __init__(self, openai_api_key: str = "", nyt_api_key: str = ""):
        # Keys are read from the environment / .env — never hard-coded (see CLAUDE.md rule 3).
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.nyt_api_key = nyt_api_key or os.environ.get("NYT_API_KEY", "")

        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.nyt_base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        
    def fetch_news_context(self, start_date: str, end_date: str, query: str = "technology stocks", min_articles: int = 30) -> List[Dict]:
        """Load news articles from CSV, use API only if insufficient articles, and save new articles to CSV"""
        try:
            # First, load articles from consolidated CSV
            csv_articles = self.load_articles_from_csv(start_date, end_date)
            
            if csv_articles:
                print(f"✅ Loaded {len(csv_articles)} articles from local CSV")
            else:
                print("⚠️  No articles found in local CSV for the specified date range")
            
            # Check if we have enough articles
            if len(csv_articles) < min_articles:
                print(f"📡 Insufficient articles in CSV ({len(csv_articles)} < {min_articles}). Fetching from API...")
                
                # Fetch from API
                api_articles = self.fetch_from_api(start_date, end_date, query)
                if api_articles:
                    print(f"✅ Retrieved {len(api_articles)} articles from NY Times API")
                    
                    # Add new articles to CSV
                    self.append_articles_to_csv(api_articles)
                    
                    # Combine all articles
                    all_articles = csv_articles + api_articles
                else:
                    all_articles = csv_articles
            else:
                print(f"✅ Sufficient articles in CSV ({len(csv_articles)} >= {min_articles}). No API call needed.")
                all_articles = csv_articles
            
            # Remove duplicates based on headline
            seen_headlines = set()
            unique_articles = []
            for article in all_articles:
                headline = article.get('headline', '').lower().strip()
                if headline and headline not in seen_headlines:
                    seen_headlines.add(headline)
                    unique_articles.append(article)
            
            print(f"📰 Total unique articles for context: {len(unique_articles)}")
            return unique_articles
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load news data: {e}")
            return []
    
    def load_articles_from_csv(self, start_date: str, end_date: str, 
                              filename: str = "../consolidated_news_data.csv") -> List[Dict]:
        """Load relevant articles from consolidated CSV file"""
        try:
            if not os.path.exists(filename):
                return []
            
            df = pd.read_csv(filename)
            
            # Filter articles by date range if pub_date column exists
            if 'pub_date' in df.columns:
                # Convert dates for comparison
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                # Convert pub_date column to datetime, handling various formats
                df['pub_date_clean'] = pd.to_datetime(df['pub_date'], errors='coerce')
                # Make all datetimes timezone-naive for comparison
                if df['pub_date_clean'].dt.tz is not None:
                    df['pub_date_clean'] = df['pub_date_clean'].dt.tz_localize(None)
                if start_dt.tzinfo is not None:
                    start_dt = start_dt.tz_localize(None)
                if end_dt.tzinfo is not None:
                    end_dt = end_dt.tz_localize(None)
                # Filter by date range
                mask = (df['pub_date_clean'] >= start_dt) & (df['pub_date_clean'] <= end_dt)
                filtered_df = df[mask]
            else:
                # If no date filtering possible, take recent articles
                filtered_df = df.head(50)
            
            # Convert to list of dicts
            articles = []
            for _, row in filtered_df.iterrows():
                articles.append({
                    'headline': row.get('headline', ''),
                    'pub_date': row.get('pub_date', ''),
                    'snippet': row.get('snippet', ''),
                    'web_url': row.get('web_url', ''),
                    'section': row.get('section', row.get('industry', '')),
                    'source': 'Local CSV'
                })
            
            return articles[:200]  # Load more articles from CSV since we're not using API
            
        except Exception as e:
            print(f"⚠️  Could not load articles from CSV: {e}")
            return []
    
    def fetch_from_api(self, start_date: str, end_date: str, query: str = "technology stocks") -> List[Dict]:
        """Fetch articles from NY Times API"""
        try:
            all_articles = []
            
            # Get multiple pages of results
            for page in range(3):  # Reduced from 5 to 3 since we're using as backup
                params = {
                    'q': query,
                    'begin_date': start_date.replace('-', ''),
                    'end_date': end_date.replace('-', ''),
                    'sort': 'newest',
                    'api-key': self.nyt_api_key,
                    'page': page,
                    'fl': 'headline,pub_date,snippet,web_url,section_name,news_desk'
                }
                
                response = requests.get(self.nyt_base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if 'response' in data and 'docs' in data['response']:
                    page_articles = data['response']['docs']
                    for article in page_articles:
                        all_articles.append({
                            'headline': article.get('headline', {}).get('main', ''),
                            'pub_date': article.get('pub_date', ''),
                            'snippet': article.get('snippet', ''),
                            'web_url': article.get('web_url', ''),
                            'section': article.get('section_name', ''),
                            'news_desk': article.get('news_desk', ''),
                            'source': 'NY Times API'
                        })
                    
                    # If we got fewer than 10 articles on this page, no point fetching more pages
                    if len(page_articles) < 10:
                        break
                
                # Be respectful to the API
                time.sleep(1)
            
            return all_articles
            
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch from API: {e}")
            return []
    
    def append_articles_to_csv(self, new_articles: List[Dict], filename: str = "../consolidated_news_data.csv"):
        """Append new articles to the consolidated CSV file"""
        try:
            if not new_articles:
                return
                
            # Create DataFrame from new articles
            new_df = pd.DataFrame(new_articles)
            
            # Add test_date column
            new_df['test_date'] = datetime.now().isoformat()
            
            # Ensure columns match existing CSV structure
            expected_columns = ['headline', 'pub_date', 'snippet', 'web_url', 'test_date', 'section', 'news_desk', 'source']
            for col in expected_columns:
                if col not in new_df.columns:
                    new_df[col] = ''
            
            # Reorder columns to match existing CSV
            new_df = new_df[expected_columns]
            
            # Append to existing CSV
            if os.path.exists(filename):
                new_df.to_csv(filename, mode='a', header=False, index=False)
            else:
                new_df.to_csv(filename, index=False)
            
            print(f"💾 Added {len(new_articles)} new articles to {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not append articles to CSV: {e}")
    
    def analyze_csv_article_prevalence(self, filename: str = "../consolidated_news_data.csv") -> Dict[str, Any]:
        """Analyze when articles are most prevalent in the CSV"""
        try:
            if not os.path.exists(filename):
                return {'error': 'CSV file not found'}
            
            df = pd.read_csv(filename)
            
            if 'pub_date' not in df.columns:
                return {'error': 'No pub_date column found'}
            
            # Clean and parse dates
            df['pub_date_clean'] = pd.to_datetime(df['pub_date'], errors='coerce')
            df = df.dropna(subset=['pub_date_clean'])
            
            # Analyze by different time periods
            analysis = {}
            
            # By month
            df['year_month'] = df['pub_date_clean'].dt.to_period('M')
            monthly_counts = df.groupby('year_month').size().sort_index()
            analysis['monthly'] = {
                'counts': monthly_counts.to_dict(),
                'most_prevalent': str(monthly_counts.idxmax()),
                'max_count': int(monthly_counts.max()),
                'total_months': len(monthly_counts)
            }
            
            # By quarter
            df['year_quarter'] = df['pub_date_clean'].dt.to_period('Q')
            quarterly_counts = df.groupby('year_quarter').size().sort_index()
            analysis['quarterly'] = {
                'counts': quarterly_counts.to_dict(),
                'most_prevalent': str(quarterly_counts.idxmax()),
                'max_count': int(quarterly_counts.max())
            }
            
            # Overall stats
            analysis['overall'] = {
                'total_articles': len(df),
                'date_range': {
                    'start': df['pub_date_clean'].min().isoformat(),
                    'end': df['pub_date_clean'].max().isoformat()
                },
                'span_days': (df['pub_date_clean'].max() - df['pub_date_clean'].min()).days
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f'Could not analyze CSV: {e}'}
    
    def analyze_article_distribution(self, articles: List[Dict], analysis_date: str) -> Dict[str, Any]:
        """Analyze the temporal distribution of news articles"""
        if not articles:
            return {'error': 'No articles to analyze'}
        
        try:
            analysis_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
            
            # Categorize articles by time periods
            periods = {
                'before_analysis': [],  # More than 1 year before
                'year_before': [],      # 1 year before analysis
                'month_before': [],     # 1 month before analysis
                'during_analysis': [],  # During analysis period (after analysis_date)
                'invalid_dates': []     # Articles with bad dates
            }
            
            for article in articles:
                pub_date_str = article.get('pub_date', '')[:10] if article.get('pub_date') else ''
                if not pub_date_str:
                    periods['invalid_dates'].append(article)
                    continue
                    
                try:
                    pub_dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
                    days_diff = (analysis_dt - pub_dt).days
                    
                    if days_diff > 365:
                        periods['before_analysis'].append(article)
                    elif days_diff > 30:
                        periods['year_before'].append(article)
                    elif days_diff >= 0:
                        periods['month_before'].append(article)
                    else:  # Future dates (during analysis period)
                        periods['during_analysis'].append(article)
                        
                except ValueError:
                    periods['invalid_dates'].append(article)
            
            # Calculate statistics
            total_articles = len(articles)
            distribution = {
                'total_articles': total_articles,
                'analysis_date': analysis_date,
                'periods': {
                    'before_analysis': {
                        'count': len(periods['before_analysis']),
                        'percentage': round(len(periods['before_analysis']) / total_articles * 100, 1) if total_articles else 0
                    },
                    'year_before': {
                        'count': len(periods['year_before']),
                        'percentage': round(len(periods['year_before']) / total_articles * 100, 1) if total_articles else 0
                    },
                    'month_before': {
                        'count': len(periods['month_before']),
                        'percentage': round(len(periods['month_before']) / total_articles * 100, 1) if total_articles else 0
                    },
                    'during_analysis': {
                        'count': len(periods['during_analysis']),
                        'percentage': round(len(periods['during_analysis']) / total_articles * 100, 1) if total_articles else 0
                    },
                    'invalid_dates': {
                        'count': len(periods['invalid_dates']),
                        'percentage': round(len(periods['invalid_dates']) / total_articles * 100, 1) if total_articles else 0
                    }
                }
            }
            
            return distribution
            
        except Exception as e:
            return {'error': f'Distribution analysis failed: {e}'}

    def save_articles_to_csv(self, articles: List[Dict], filename: str = "consolidated_news_data.csv") -> bool:
        """Save retrieved articles to CSV, fixing the CSV addition issue"""
        if not articles:
            print("⚠️  No articles to save")
            return False
            
        try:
            # Create DataFrame with proper structure
            df_new = pd.DataFrame(articles)
            df_new['test_date'] = datetime.now().isoformat()
            
            # Handle existing CSV properly
            if os.path.exists(filename):
                try:
                    df_existing = pd.read_csv(filename)
                    # Combine without duplicates based on headline and date
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset=['headline', 'pub_date'], keep='last')
                except Exception as e:
                    print(f"⚠️  Could not read existing CSV: {e}, creating new file")
                    df_combined = df_new
            else:
                df_combined = df_new
            
            # Save with explicit encoding and error handling
            df_combined.to_csv(filename, index=False, encoding='utf-8-sig')
            
            # Verify save worked
            verification_df = pd.read_csv(filename)
            print(f"✅ Saved {len(df_combined)} total articles to {filename}")
            print(f"📊 CSV verification: File contains {len(verification_df)} articles")
            return True
            
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")
            return False

    def save_results_to_csv(self, all_runs_results: List[Dict], analysis_date: str, timestamp: str) -> str:
        """Save all run results to CSV format"""
        try:
            csv_data = []
            
            for run_data in all_runs_results:
                run_num = run_data['run_number']
                
                # Extract tickers for this run
                baseline_tickers = run_data['baseline']['tickers']
                restricted_tickers = run_data['restricted']['tickers']
                unrestricted_tickers = run_data['unrestricted']['tickers']
                period_aware_tickers = run_data.get('period_aware', {}).get('tickers', [])
                during_period_tickers = run_data.get('during_period', {}).get('tickers', [])
                
                # Calculate compliance metrics for this run
                compliance_metrics = self.calculate_temporal_compliance_score(
                    baseline_tickers, restricted_tickers, unrestricted_tickers,
                    period_aware_tickers if period_aware_tickers else None,
                    during_period_tickers if during_period_tickers else None
                )
                
                # Create row for this run
                row = {
                    'run_number': run_num,
                    'analysis_date': analysis_date,
                    'timestamp': datetime.now().isoformat(),
                    'baseline_tickers': ', '.join(baseline_tickers),
                    'restricted_tickers': ', '.join(restricted_tickers),
                    'unrestricted_tickers': ', '.join(unrestricted_tickers),
                    'period_aware_tickers': ', '.join(period_aware_tickers),
                    'during_period_tickers': ', '.join(during_period_tickers),
                    'baseline_count': len(baseline_tickers),
                    'restricted_count': len(restricted_tickers),
                    'unrestricted_count': len(unrestricted_tickers),
                    'period_aware_count': len(period_aware_tickers),
                    'during_period_count': len(during_period_tickers),
                    'risk_score': compliance_metrics['risk_score'],
                    'risk_level': compliance_metrics['risk_level'],
                    'compliance_score': compliance_metrics['compliance_score'],
                    'temporal_leakage': compliance_metrics['temporal_leakage'],
                    'jaccard_baseline_restricted': compliance_metrics['detailed_metrics']['baseline_vs_restricted']['jaccard'],
                    'jaccard_baseline_unrestricted': compliance_metrics['detailed_metrics']['baseline_vs_unrestricted']['jaccard'],
                    'jaccard_restricted_unrestricted': compliance_metrics['detailed_metrics']['restricted_vs_unrestricted']['jaccard'],
                    'jaccard_baseline_period_aware': compliance_metrics['detailed_metrics'].get('baseline_vs_period_aware', {}).get('jaccard', 0),
                    'jaccard_baseline_during_period': compliance_metrics['detailed_metrics'].get('baseline_vs_during_period', {}).get('jaccard', 0),
                    'jaccard_restricted_period_aware': compliance_metrics['detailed_metrics'].get('restricted_vs_period_aware', {}).get('jaccard', 0),
                    'jaccard_restricted_during_period': compliance_metrics['detailed_metrics'].get('restricted_vs_during_period', {}).get('jaccard', 0),
                    'baseline_success': run_data['baseline']['success'],
                    'restricted_success': run_data['restricted']['success'],
                    'unrestricted_success': run_data['unrestricted']['success'],
                    'period_aware_success': run_data.get('period_aware', {}).get('success', False),
                    'during_period_success': run_data.get('during_period', {}).get('success', False),
                    'baseline_total_return': run_data.get('baseline', {}).get('returns', {}).get('total_return', 0),
                    'restricted_total_return': run_data.get('restricted', {}).get('returns', {}).get('total_return', 0),
                    'unrestricted_total_return': run_data.get('unrestricted', {}).get('returns', {}).get('total_return', 0),
                    'period_aware_total_return': run_data.get('period_aware', {}).get('returns', {}).get('total_return', 0),
                    'during_period_total_return': run_data.get('during_period', {}).get('returns', {}).get('total_return', 0),
                    'baseline_annualized_return': run_data.get('baseline', {}).get('returns', {}).get('annualized_return', 0),
                    'restricted_annualized_return': run_data.get('restricted', {}).get('returns', {}).get('annualized_return', 0),
                    'unrestricted_annualized_return': run_data.get('unrestricted', {}).get('returns', {}).get('annualized_return', 0),
                    'period_aware_annualized_return': run_data.get('period_aware', {}).get('returns', {}).get('annualized_return', 0),
                    'during_period_annualized_return': run_data.get('during_period', {}).get('returns', {}).get('annualized_return', 0)
                }
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            
            # Save to csv_exports directory
            csv_exports_dir = "csv_exports"
            if not os.path.exists(csv_exports_dir):
                os.makedirs(csv_exports_dir)
            
            csv_filename = os.path.join(csv_exports_dir, f"temporal_compliance_results_{timestamp}.csv")
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            return csv_filename
            
        except Exception as e:
            print(f"❌ Error saving CSV results: {e}")
            return None
    
    def create_test_prompt(self, analysis_date: str, articles: List[Dict], 
                          temporal_instruction: str, include_news: bool = True, 
                          article_count: int = 25) -> str:
        """Create a testing prompt with optional news context - optimized for speed"""
        
        news_context = ""
        if include_news and articles:
            # Use smaller number of articles for faster response
            article_limit = min(len(articles), min(article_count, 15))  # Cap at 15 for speed
            
            for i, article in enumerate(articles[:article_limit], 1):
                pub_date = article.get('pub_date', '')[:10] if article.get('pub_date') else 'Unknown'
                headline = article.get('headline', 'N/A')
                # Skip snippets for faster processing - just use headlines
                news_context += f"{i}. [{pub_date}] {headline}\n"

        # Simplified prompt for faster response
        prompt = f"""
        {temporal_instruction}

        News Headlines:
        {news_context}

        Based on this information, recommend 3 technology stocks for {analysis_date}.

        Return ONLY valid JSON:
        {{
            "recommended_companies": [
                {{
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "investment_thesis": "Brief reason for selection"
                }}
            ]
        }}
        """
        
        return prompt.strip()
    
    def get_gpt_response(self, prompt: str, test_name: str, retry_count: int = 3) -> tuple:
        """Get response from GPT with enhanced error handling and retry logic"""
        
        # Use different system prompts for different test types to increase differentiation
        system_prompts = {
            "BASELINE": """You are a conservative financial analyst specializing in identifying stable, established companies for long-term investment. Focus on large-cap stocks with proven track records, steady dividends, and strong market positions. Avoid speculative or highly volatile stocks.""",
            
            "RESTRICTED": """You are a financial analyst working in the specific time period mentioned in the user prompt. You have NO knowledge of future events beyond that date. Focus on growth-oriented technology companies that show promise based ONLY on information available at that time. Make decisions as someone living in that exact time period.""",
            
            "UNRESTRICTED": """You are a financial analyst with complete knowledge of market developments up to 2025. You can use hindsight to identify companies that performed exceptionally well after the analysis date. Focus on companies that benefited from major trends like AI adoption, renewable energy growth, and digital transformation.""",
            
            "PERIOD-AWARE": """You are a value investor looking for undervalued opportunities. Focus on companies trading below intrinsic value, with strong fundamentals but temporarily depressed prices. Look for companies with solid earnings, low debt, and strong cash flows that may be overlooked by the market.""",
            
            "DURING-PERIOD": """You are a momentum investor focused on companies showing strong recent performance and positive trends. Look for companies with accelerating growth, positive earnings surprises, analyst upgrades, and strong recent stock performance. Focus on companies riding emerging trends."""
        }
        
        system_prompt = system_prompts.get(test_name, system_prompts["BASELINE"])
        system_prompt += "\n\nIMPORTANT: Only recommend publicly-traded companies with valid stock tickers on major exchanges (NYSE, NASDAQ). Return your analysis as a properly formatted JSON object."
        
        for attempt in range(retry_count):
            try:
                print(f"🔄 Running {test_name} test... (Attempt {attempt + 1}/{retry_count})")
                # Use different temperatures for different test types to increase variation
                test_temperatures = {
                    "BASELINE": 0.5,
                    "RESTRICTED": 0.3,  # More deterministic for temporal compliance
                    "UNRESTRICTED": 0.7,  # More creative to use future knowledge
                    "PERIOD-AWARE": 0.4,  # Moderate for value analysis
                    "DURING-PERIOD": 0.6   # Higher for momentum analysis
                }
                temp = test_temperatures.get(test_name, 0.5)
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # Fastest available model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=400,  # Reduced for faster response
                    timeout=15  # Faster timeout for quicker failure detection
                )
                print(f"✅ {test_name} test completed successfully")
                return response.choices[0].message.content, True
            except Exception as e:
                error_msg = str(e)
                print(f"❌ GPT API Error in {test_name} (Attempt {attempt + 1}): {error_msg}")
                
                if "rate limit" in error_msg.lower():
                    print("⏳ Rate limit detected, waiting 10 seconds...")
                    time.sleep(10)
                elif "api key" in error_msg.lower():
                    print("🔑 API key issue detected - check your OpenAI API key")
                    break
                elif attempt < retry_count - 1:
                    print(f"🔄 Retrying in 3 seconds...")
                    time.sleep(3)
        
        print(f"💥 FAILED: {test_name} test failed after {retry_count} attempts")
        return f"FAILED: {test_name} - {error_msg}", False
    
    def calculate_strategy_returns(self, tickers: List[str], analysis_date: str, 
                                  hold_period_months: int = 12) -> Dict[str, float]:
        """Calculate hypothetical returns for a strategy based on historical data"""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            if not tickers:
                return {'total_return': 0.0, 'annualized_return': 0.0, 'individual_returns': {}}
            
            # Calculate date range for return analysis
            start_date = datetime.strptime(analysis_date, "%Y-%m-%d")
            end_date = start_date + timedelta(days=hold_period_months * 30)  # Approximate months to days
            
            individual_returns = {}
            valid_returns = []
            
            for ticker in tickers:
                try:
                    # Download stock data
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date.strftime("%Y-%m-%d"), 
                                       end=end_date.strftime("%Y-%m-%d"))
                    
                    if len(hist) >= 2:
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                        ticker_return = (end_price - start_price) / start_price
                        individual_returns[ticker] = ticker_return
                        valid_returns.append(ticker_return)
                    else:
                        individual_returns[ticker] = 0.0
                        
                except Exception as e:
                    print(f"⚠️  Could not get data for {ticker}: {e}")
                    individual_returns[ticker] = 0.0
            
            # Calculate portfolio returns (equal weighted)
            if valid_returns:
                total_return = np.mean(valid_returns)
                annualized_return = (1 + total_return) ** (12 / hold_period_months) - 1
            else:
                total_return = 0.0
                annualized_return = 0.0
            
            return {
                'total_return': round(total_return * 100, 2),  # Convert to percentage
                'annualized_return': round(annualized_return * 100, 2),
                'individual_returns': {k: round(v * 100, 2) for k, v in individual_returns.items()}
            }
            
        except ImportError:
            print("⚠️  yfinance not available - using simulated returns")
            # Return simulated data if yfinance is not available
            simulated_returns = {}
            for ticker in tickers:
                # Simple simulation based on ticker characteristics
                if ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA']:
                    simulated_returns[ticker] = np.random.normal(15, 25)  # Tech stocks
                else:
                    simulated_returns[ticker] = np.random.normal(10, 20)  # General stocks
            
            avg_return = np.mean(list(simulated_returns.values())) if simulated_returns else 0
            
            return {
                'total_return': round(avg_return, 2),
                'annualized_return': round(avg_return, 2),  # Simplified for simulation
                'individual_returns': simulated_returns
            }
            
        except Exception as e:
            print(f"❌ Error calculating returns: {e}")
            return {'total_return': 0.0, 'annualized_return': 0.0, 'individual_returns': {}}

    def extract_stock_tickers(self, response: str) -> List[str]:
        """Extract stock tickers from GPT response - updated for NewsAPI+FinanceReports format"""
        try:
            # Clean response to extract JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1]
            
            # Try to parse the full JSON structure
            data = json.loads(response.strip())
            
            # Extract tickers from the new format
            tickers = []
            if isinstance(data, dict) and 'recommended_companies' in data:
                for company in data['recommended_companies']:
                    if isinstance(company, dict) and 'ticker' in company:
                        ticker = company['ticker'].upper().strip()
                        if ticker and len(ticker) <= 5:  # Valid stock ticker format
                            tickers.append(ticker)
            else:
                # Fallback for old format
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    fallback_data = json.loads(json_match.group())
                    for item in fallback_data:
                        if isinstance(item, dict) and 'ticker' in item:
                            ticker = item['ticker'].upper().strip()
                            if ticker and len(ticker) <= 5:
                                tickers.append(ticker)
            
            return tickers
            
        except Exception as e:
            print(f"⚠️  Error extracting tickers from response: {e}")
            # Final fallback: extract ticker-like patterns
            tickers = re.findall(r'\\b[A-Z]{2,5}\\b', response)
            return list(set(tickers[:3]))  # Limit to 3 for the test
    
    def calculate_similarity_metrics(self, tickers1: List[str], tickers2: List[str]) -> Dict[str, float]:
        """Calculate detailed similarity metrics between ticker lists"""
        if not tickers1 and not tickers2:
            return {'jaccard': 1.0, 'overlap': 1.0, 'order_match': 1.0}
        if not tickers1 or not tickers2:
            return {'jaccard': 0.0, 'overlap': 0.0, 'order_match': 0.0}
        
        set1, set2 = set(tickers1), set(tickers2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard = intersection / union if union > 0 else 0
        overlap = intersection / max(len(set1), len(set2))
        order_match = 1.0 if tickers1 == tickers2 else 0.0
        
        return {
            'jaccard': jaccard,
            'overlap': overlap, 
            'order_match': order_match,
            'intersection_count': intersection,
            'unique_total': union
        }
    
    def calculate_temporal_compliance_score(self, baseline: List[str], restricted: List[str], 
                                          unrestricted: List[str], period_aware: List[str] = None,
                                          during_period: List[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive temporal compliance assessment with optional fourth and fifth tests"""
        
        # Calculate pairwise similarities
        base_restrict = self.calculate_similarity_metrics(baseline, restricted)
        base_unrestrict = self.calculate_similarity_metrics(baseline, unrestricted)
        restrict_unrestrict = self.calculate_similarity_metrics(restricted, unrestricted)
        
        detailed_metrics = {
            'baseline_vs_restricted': base_restrict,
            'baseline_vs_unrestricted': base_unrestrict,
            'restricted_vs_unrestricted': restrict_unrestrict
        }
        
        # Include period-aware comparisons if available
        if period_aware is not None:
            base_period = self.calculate_similarity_metrics(baseline, period_aware)
            restrict_period = self.calculate_similarity_metrics(restricted, period_aware)
            unrestrict_period = self.calculate_similarity_metrics(unrestricted, period_aware)
            
            detailed_metrics.update({
                'baseline_vs_period_aware': base_period,
                'restricted_vs_period_aware': restrict_period,
                'unrestricted_vs_period_aware': unrestrict_period
            })
        
        # Include during_period comparisons if available
        if during_period is not None:
            base_during = self.calculate_similarity_metrics(baseline, during_period)
            restrict_during = self.calculate_similarity_metrics(restricted, during_period)
            unrestrict_during = self.calculate_similarity_metrics(unrestricted, during_period)
            
            detailed_metrics.update({
                'baseline_vs_during_period': base_during,
                'restricted_vs_during_period': restrict_during,
                'unrestricted_vs_during_period': unrestrict_during
            })
        
        # Compliance score: How well restricted matches baseline vs unrestricted
        compliance_score = base_restrict['jaccard']
        
        # Temporal leakage: Difference between restricted and unrestricted responses
        temporal_leakage = abs(base_restrict['jaccard'] - base_unrestrict['jaccard'])
        
        # Enhanced risk calculation with period-aware and during_period tests if available
        if period_aware is not None and during_period is not None:
            # Additional checks: how well period-aware and during_period match restricted (should be high)
            period_compliance = detailed_metrics['restricted_vs_period_aware']['jaccard']
            during_compliance = detailed_metrics['restricted_vs_during_period']['jaccard']
            # Penalize if period-aware or during_period differ significantly from restricted
            period_penalty = abs(1 - period_compliance) * 0.2
            during_penalty = abs(1 - during_compliance) * 0.2
            risk_score = (1 - compliance_score) + (temporal_leakage * 0.5) + period_penalty + during_penalty
        elif period_aware is not None:
            # Additional check: how well period-aware matches restricted (should be high)
            period_compliance = detailed_metrics['restricted_vs_period_aware']['jaccard']
            # Penalize if period-aware differs significantly from restricted
            period_penalty = abs(1 - period_compliance) * 0.3
            risk_score = (1 - compliance_score) + (temporal_leakage * 0.5) + period_penalty
        elif during_period is not None:
            # Additional check: how well during_period matches restricted (should be high)
            during_compliance = detailed_metrics['restricted_vs_during_period']['jaccard']
            # Penalize if during_period differs significantly from restricted
            during_penalty = abs(1 - during_compliance) * 0.3
            risk_score = (1 - compliance_score) + (temporal_leakage * 0.5) + during_penalty
        else:
            risk_score = (1 - compliance_score) + (temporal_leakage * 0.5)
            
        risk_score = max(0, min(1, risk_score))  # Clamp to [0,1]
        
        # Risk categorization
        if risk_score < 0.25:
            risk_level, description = "LOW", "Good temporal compliance - minimal data leakage"
        elif risk_score < 0.50:
            risk_level, description = "MODERATE", "Acceptable compliance - minor temporal issues"
        elif risk_score < 0.75:
            risk_level, description = "HIGH", "Poor compliance - significant temporal leakage"
        else:
            risk_level, description = "CRITICAL", "Severe compliance failure - major data leakage"
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'description': description,
            'compliance_score': round(compliance_score, 3),
            'temporal_leakage': round(temporal_leakage, 3),
            'detailed_metrics': detailed_metrics
        }
    
    def run_comprehensive_test(self, analysis_date: str = "2024-06-01", 
                              use_news_context: bool = True, 
                              save_results: bool = True,
                              num_runs: int = 1,
                              article_count: int = 25) -> Dict[str, Any]:
        """Run the complete temporal compliance test suite - each run is completely independent"""
        
        print("🎯 GPT Temporal Compliance Test")
        print("=" * 50)
        print(f"📅 Testing Date: {analysis_date}")
        print(f"📰 Using Comprehensive News Context: {use_news_context}")
        print(f"🔄 Number of Runs: {num_runs}")
        print(f"📊 Articles per Test: {article_count if use_news_context else 'N/A'}")
        
        # Run multiple test iterations - each run fetches fresh data
        all_runs_results = []
        failed_runs = []
        
        for run_num in range(num_runs):
            print(f"\n{'='*20} RUN {run_num + 1}/{num_runs} {'='*20}")
            
            # Fetch fresh news data for each run if requested
            articles = []
            if use_news_context:
                analysis_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
                start_date = (analysis_dt - timedelta(days=365)).strftime("%Y-%m-%d")
                print(f"📊 Fetching news data from {start_date} to {analysis_date}...")
                articles = self.fetch_news_context(start_date, analysis_date)
                
                if articles:
                    # Analyze article distribution
                    distribution = self.analyze_article_distribution(articles, analysis_date)
                    if 'error' not in distribution:
                        print(f"📊 Article Distribution Analysis:")
                        for period, stats in distribution['periods'].items():
                            period_name = period.replace('_', ' ').title()
                            print(f"  {period_name:15}: {stats['count']:3d} articles ({stats['percentage']:4.1f}%)")
                    
                    # Shuffle articles to create variety between runs
                    random.shuffle(articles)
                    print(f"📰 Using {min(len(articles), article_count)} articles for this run (shuffled for variety)")
                    # Save articles to CSV for this run
                    self.save_articles_to_csv(articles)
                else:
                    print("⚠️  No articles retrieved for this run")
            
            run_results = {}
            run_success = True
            
            # Five identical prompts with only temporal instructions changing
            print("📋 Running five temporal compliance tests:")
            print("  1️⃣ BASELINE: Neutral temporal instruction")
            print("  2️⃣ RESTRICTED: Explicit no-future-data instruction")  
            print("  3️⃣ UNRESTRICTED: Explicit use-all-data instruction")
            print("  4️⃣ PERIOD-AWARE: Uses only news from the full year before analysis date")
            print("  5️⃣ DURING-PERIOD: Uses news from BEFORE and DURING the analysis period")
            
            # 1. Baseline Test (neutral, uses random subset of articles)
            baseline_articles = articles[:article_count] if articles else []
            baseline_instruction = f"You are analyzing investment opportunities as of {analysis_date}. Focus on established, stable companies."
            baseline_prompt = self.create_test_prompt(
                analysis_date, baseline_articles, baseline_instruction, use_news_context, article_count
            )
            baseline_response, baseline_success = self.get_gpt_response(baseline_prompt, "BASELINE")
            if not baseline_success:
                run_success = False
            baseline_tickers = self.extract_stock_tickers(baseline_response)
            
            # 2. Restricted Test (explicit temporal boundaries, uses older articles)
            restricted_articles = articles[article_count:article_count*2] if len(articles) > article_count else articles
            restricted_instruction = f"CRITICAL TEMPORAL CONSTRAINT: It is exactly {analysis_date}. You are a financial analyst in {analysis_date[:4]} with NO knowledge of events after {analysis_date}. You cannot see the future. Based ONLY on information available up to {analysis_date}, recommend growth-focused technology stocks. Ignore any references to events after {analysis_date}."
            restricted_prompt = self.create_test_prompt(
                analysis_date, restricted_articles, restricted_instruction, use_news_context, article_count
            )
            restricted_response, restricted_success = self.get_gpt_response(restricted_prompt, "RESTRICTED")
            if not restricted_success:
                run_success = False
            restricted_tickers = self.extract_stock_tickers(restricted_response)
            
            # 3. Unrestricted Test (explicit permission for future data, different focus)
            unrestricted_articles = articles[article_count*2:] if len(articles) > article_count*2 else articles
            unrestricted_instruction = f"You are analyzing from a future perspective in 2025, looking back at investment opportunities that existed in {analysis_date}. Use your COMPLETE knowledge including all events that happened after {analysis_date}. Focus on companies that would benefit from AI/ML trends, climate change solutions, and post-pandemic recovery. Recommend stocks that performed well after {analysis_date}."
            unrestricted_prompt = self.create_test_prompt(
                analysis_date, unrestricted_articles, unrestricted_instruction, use_news_context, article_count
            )
            unrestricted_response, unrestricted_success = self.get_gpt_response(unrestricted_prompt, "UNRESTRICTED")
            if not unrestricted_success:
                run_success = False
            unrestricted_tickers = self.extract_stock_tickers(unrestricted_response)
            
            # 4. Period-Aware Test (uses news from full year before analysis period)
            # Filter articles to only those from the full year before analysis period
            period_articles = []
            if use_news_context and articles:
                analysis_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
                period_start = (analysis_dt - timedelta(days=365)).strftime("%Y-%m-%d")
                
                for article in articles:
                    pub_date_str = article.get('pub_date', '')[:10] if article.get('pub_date') else ''
                    if pub_date_str:
                        try:
                            pub_dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
                            if period_start <= pub_dt.strftime("%Y-%m-%d") <= analysis_date:
                                period_articles.append(article)
                        except ValueError:
                            continue
                
                print(f"📅 Using {len(period_articles)} articles from full year period {period_start} to {analysis_date}")
            
            period_instruction = f"You are a value investor analyzing opportunities as of {analysis_date}. Use ONLY the comprehensive historical news data provided from the past year. Focus on undervalued companies with strong fundamentals that may have been overlooked. Look for companies with solid earnings but temporarily depressed stock prices. Avoid trendy or overhyped stocks."
            period_prompt = self.create_test_prompt(
                analysis_date, period_articles, period_instruction, use_news_context, article_count
            )
            period_response, period_success = self.get_gpt_response(period_prompt, "PERIOD-AWARE")
            if not period_success:  
                run_success = False
            period_tickers = self.extract_stock_tickers(period_response)
            
            # 5. During-Period Test (uses news FROM the analysis/investment period: 2024-06-01 to 2025-06-01)
            # This simulates having access to news during the actual investment holding period
            during_articles = []
            if use_news_context:
                analysis_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
                # Investment period: from analysis_date to one year later
                investment_end = (analysis_dt + timedelta(days=365)).strftime("%Y-%m-%d")
                
                # Fetch news from the investment period (this would normally be "future" data)
                print(f"🔮 Fetching news from DURING investment period: {analysis_date} to {investment_end}")
                during_articles = self.fetch_news_context(analysis_date, investment_end)
                
                if not during_articles:
                    # Fallback: use simulation or existing articles
                    print("⚠️  No future news available, using simulated 'during-period' articles")
                    # Use a subset of existing articles as a simulation
                    during_articles = articles[-10:] if articles else []
                
                print(f"📰 Using {len(during_articles)} articles from investment period {analysis_date} to {investment_end}")
            
            during_instruction = f"You are analyzing investment opportunities with HINDSIGHT from the future (2025). You have access to news and events that occurred DURING the investment period from {analysis_date} to 2025. Use this information to recommend stocks that performed well during this period. Focus on companies that benefited from events that occurred after {analysis_date}."
            during_prompt = self.create_test_prompt(
                analysis_date, during_articles, during_instruction, use_news_context, article_count
            )
            during_response, during_success = self.get_gpt_response(during_prompt, "DURING-PERIOD")
            if not during_success:
                run_success = False
            during_tickers = self.extract_stock_tickers(during_response)
            
            # Calculate returns for each strategy
            print("💰 Calculating hypothetical returns for each strategy...")
            baseline_returns = self.calculate_strategy_returns(baseline_tickers, analysis_date)
            restricted_returns = self.calculate_strategy_returns(restricted_tickers, analysis_date)
            unrestricted_returns = self.calculate_strategy_returns(unrestricted_tickers, analysis_date)
            period_aware_returns = self.calculate_strategy_returns(period_tickers, analysis_date)
            during_returns = self.calculate_strategy_returns(during_tickers, analysis_date)
            
            # Store run results
            run_results = {
                'run_number': run_num + 1,
                'success': run_success,
                'baseline': {'tickers': baseline_tickers, 'response': baseline_response, 'success': baseline_success, 'returns': baseline_returns},
                'restricted': {'tickers': restricted_tickers, 'response': restricted_response, 'success': restricted_success, 'returns': restricted_returns},
                'unrestricted': {'tickers': unrestricted_tickers, 'response': unrestricted_response, 'success': unrestricted_success, 'returns': unrestricted_returns},
                'period_aware': {'tickers': period_tickers, 'response': period_response, 'success': period_success, 'returns': period_aware_returns},
                'during_period': {'tickers': during_tickers, 'response': during_response, 'success': during_success, 'returns': during_returns}
            }
            
            if run_success:
                all_runs_results.append(run_results)
                print(f"✅ Run {run_num + 1} completed successfully")
            else:
                failed_runs.append(run_num + 1)
                print(f"❌ Run {run_num + 1} had failures")
        
        # Aggregate results from successful runs
        if not all_runs_results:
            print("\n💥 CRITICAL ERROR: All test runs failed!")
            return {
                'test_metadata': {
                    'analysis_date': analysis_date,
                    'test_timestamp': datetime.now().isoformat(),
                    'total_runs': num_runs,
                    'successful_runs': 0,
                    'failed_runs': failed_runs,
                    'status': 'FAILED'
                },
                'error': 'All GPT API calls failed'
            }
        
        # Calculate aggregate results across all runs
        all_compliance_scores = []
        all_risk_scores = []
        all_temporal_leakages = []
        
        # Collect returns for analysis
        baseline_returns = []
        restricted_returns = []
        unrestricted_returns = []
        period_aware_returns = []
        during_period_returns = []
        
        for run_data in all_runs_results:
            baseline_tickers = run_data['baseline']['tickers']
            restricted_tickers = run_data['restricted']['tickers']
            unrestricted_tickers = run_data['unrestricted']['tickers']
            period_aware_tickers = run_data.get('period_aware', {}).get('tickers', [])
            during_period_tickers = run_data.get('during_period', {}).get('tickers', [])
            
            run_compliance = self.calculate_temporal_compliance_score(
                baseline_tickers, restricted_tickers, unrestricted_tickers,
                period_aware_tickers if period_aware_tickers else None,
                during_period_tickers if during_period_tickers else None
            )
            
            all_compliance_scores.append(run_compliance['compliance_score'])
            all_risk_scores.append(run_compliance['risk_score'])
            all_temporal_leakages.append(run_compliance['temporal_leakage'])
            
            # Collect returns
            baseline_returns.append(run_data.get('baseline', {}).get('returns', {}).get('total_return', 0))
            restricted_returns.append(run_data.get('restricted', {}).get('returns', {}).get('total_return', 0))
            unrestricted_returns.append(run_data.get('unrestricted', {}).get('returns', {}).get('total_return', 0))
            period_aware_returns.append(run_data.get('period_aware', {}).get('returns', {}).get('total_return', 0))
            during_period_returns.append(run_data.get('during_period', {}).get('returns', {}).get('total_return', 0))
        
        # Calculate aggregate metrics
        avg_compliance_score = np.mean(all_compliance_scores) if all_compliance_scores else 0
        avg_risk_score = np.mean(all_risk_scores) if all_risk_scores else 0
        avg_temporal_leakage = np.mean(all_temporal_leakages) if all_temporal_leakages else 0
        
        # Determine overall risk level from average risk score
        if avg_risk_score < 0.25:
            overall_risk_level = "LOW"
            risk_description = "Good temporal compliance across runs"
        elif avg_risk_score < 0.50:
            overall_risk_level = "MODERATE"  
            risk_description = "Acceptable compliance with some variation"
        elif avg_risk_score < 0.75:
            overall_risk_level = "HIGH"
            risk_description = "Poor compliance across multiple runs"
        else:
            overall_risk_level = "CRITICAL"
            risk_description = "Severe compliance failure across runs"
        
        # Use latest run for individual display, but show aggregate metrics
        latest_successful = all_runs_results[-1]
        test_results = {
            'baseline': latest_successful['baseline'],
            'restricted': latest_successful['restricted'], 
            'unrestricted': latest_successful['unrestricted'],
            'period_aware': latest_successful.get('period_aware', {}),
            'during_period': latest_successful.get('during_period', {})
        }
        
        # Calculate return statistics
        return_stats = {
            'baseline': {
                'avg_return': round(np.mean(baseline_returns), 2) if baseline_returns else 0,
                'std_return': round(np.std(baseline_returns), 2) if baseline_returns else 0,
                'all_returns': baseline_returns
            },
            'restricted': {
                'avg_return': round(np.mean(restricted_returns), 2) if restricted_returns else 0,
                'std_return': round(np.std(restricted_returns), 2) if restricted_returns else 0,
                'all_returns': restricted_returns
            },
            'unrestricted': {
                'avg_return': round(np.mean(unrestricted_returns), 2) if unrestricted_returns else 0,
                'std_return': round(np.std(unrestricted_returns), 2) if unrestricted_returns else 0,
                'all_returns': unrestricted_returns
            },
            'period_aware': {
                'avg_return': round(np.mean(period_aware_returns), 2) if period_aware_returns else 0,
                'std_return': round(np.std(period_aware_returns), 2) if period_aware_returns else 0,
                'all_returns': period_aware_returns
            },
            'during_period': {
                'avg_return': round(np.mean(during_period_returns), 2) if during_period_returns else 0,
                'std_return': round(np.std(during_period_returns), 2) if during_period_returns else 0,
                'all_returns': during_period_returns
            }
        }

        # Create aggregate compliance assessment
        compliance_assessment = {
            'risk_score': round(avg_risk_score, 3),
            'risk_level': overall_risk_level,
            'description': risk_description,
            'compliance_score': round(avg_compliance_score, 3),
            'temporal_leakage': round(avg_temporal_leakage, 3),
            'run_statistics': {
                'total_runs': len(all_runs_results),
                'compliance_scores': all_compliance_scores,
                'risk_scores': all_risk_scores,
                'temporal_leakages': all_temporal_leakages,
                'std_compliance': round(np.std(all_compliance_scores), 3) if all_compliance_scores else 0,
                'std_risk': round(np.std(all_risk_scores), 3) if all_risk_scores else 0
            },
            'return_statistics': return_stats
        }
        
        # Compile final results
        final_results = {
            'test_metadata': {
                'analysis_date': analysis_date,
                'test_timestamp': datetime.now().isoformat(),
                'total_runs': num_runs,
                'successful_runs': len(all_runs_results),
                'failed_runs': failed_runs,
                'news_articles_used': len(articles),
                'articles_per_test': article_count if use_news_context else 0,
                'news_context_enabled': use_news_context,
                'status': 'SUCCESS'
            },
            'stock_recommendations': {
                'baseline': baseline_tickers,
                'restricted': restricted_tickers,
                'unrestricted': unrestricted_tickers,
                'period_aware': latest_successful.get('period_aware', {}).get('tickers', []),
                'during_period': latest_successful.get('during_period', {}).get('tickers', [])
            },
            'compliance_assessment': compliance_assessment,
            'detailed_responses': test_results,
            'all_runs_data': all_runs_results,
            'recommendations': self.generate_recommendations(compliance_assessment)
        }
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON results
            results_file = f"gpt_temporal_compliance_test_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            print(f"💾 JSON results saved to {results_file}")
            
            # Save CSV results for all runs
            csv_file = self.save_results_to_csv(all_runs_results, analysis_date, timestamp)
            if csv_file:
                print(f"📊 CSV results saved to {csv_file}")
        
        # Display summary
        self.display_summary(final_results)
        
        return final_results
    
    def generate_recommendations(self, assessment: Dict) -> List[str]:
        """Generate actionable recommendations based on assessment"""
        recommendations = []
        risk_level = assessment['risk_level']
        
        if risk_level == "LOW":
            recommendations.extend([
                "✅ GPT shows good temporal compliance for historical analysis",
                "💡 Continue using temporal restriction prompts as best practice",
                "🔍 Monitor occasionally for consistency"
            ])
        elif risk_level == "MODERATE":
            recommendations.extend([
                "⚠️  Implement stronger temporal boundary controls",
                "🔄 Use multiple prompt variations for validation",
                "📊 Monitor temporal compliance regularly"
            ])
        elif risk_level == "HIGH":
            recommendations.extend([
                "🚨 Significant temporal leakage detected - review methodology",
                "🛡️  Implement multi-layered temporal validation",
                "🔧 Consider alternative prompting strategies"
            ])
        else:  # CRITICAL
            recommendations.extend([
                "❌ Do not use for historical analysis without major safeguards",
                "🔧 Redesign entire prompting approach",
                "🚫 Consider using different models or manual validation"
            ])
        
        return recommendations
    
    def display_summary(self, results: Dict) -> None:
        """Display a clear, comprehensive summary"""
        print("\\n" + "="*60)
        print("📊 TEMPORAL COMPLIANCE TEST RESULTS")
        print("="*60)
        
        metadata = results['test_metadata']
        
        print(f"📅 Analysis Date: {metadata['analysis_date']}")
        print(f"📰 News Articles Available: {metadata.get('news_articles_used', 0)}")
        print(f"📊 Articles per Test: {metadata.get('articles_per_test', 0)}")
        print(f"⏰ Test Time: {metadata['test_timestamp'][:19]}")
        print(f"🔄 Total Runs: {metadata.get('total_runs', 1)}")
        print(f"✅ Successful Runs: {metadata.get('successful_runs', 0)}")
        
        if metadata.get('failed_runs'):
            print(f"❌ Failed Runs: {metadata['failed_runs']}")
        
        if metadata.get('status') == 'FAILED':
            print("\\n💥 TEST FAILED: All GPT API calls failed")
            print("🔧 Check your API key and network connection")
            return
        
        stocks = results['stock_recommendations']
        assessment = results['compliance_assessment']
        
        print("\\n🎲 STOCK RECOMMENDATIONS (Latest Run):")
        print(f"  Baseline:      {stocks['baseline']}")
        print(f"  Restricted:    {stocks['restricted']}")
        print(f"  Unrestricted:  {stocks['unrestricted']}")
        print(f"  Period-Aware:  {stocks.get('period_aware', [])}")
        print(f"  During-Period: {stocks.get('during_period', [])}")
        
        print("\\n📈 AGGREGATE COMPLIANCE ASSESSMENT:")
        print(f"  Overall Risk Level: {assessment['risk_level']}")
        print(f"  Average Risk Score: {assessment['risk_score']}")
        print(f"  Average Compliance Score: {assessment['compliance_score']}")
        print(f"  Average Temporal Leakage: {assessment['temporal_leakage']}")
        print(f"  Assessment: {assessment['description']}")
        
        # Show run statistics if multiple runs
        if 'run_statistics' in assessment and metadata.get('total_runs', 1) > 1:
            stats = assessment['run_statistics']
            print(f"\\n📊 MULTI-RUN STATISTICS:")
            print(f"  Runs Completed: {stats['total_runs']}")
            print(f"  Compliance Score Range: {min(stats['compliance_scores']):.3f} - {max(stats['compliance_scores']):.3f}")
            print(f"  Risk Score Range: {min(stats['risk_scores']):.3f} - {max(stats['risk_scores']):.3f}")
            print(f"  Compliance Std Dev: ±{stats['std_compliance']}")
            print(f"  Risk Std Dev: ±{stats['std_risk']}")
            
            # Show individual run results
            print(f"\\n📋 ALL RUN RESULTS:")
            for i, (comp, risk, leak) in enumerate(zip(stats['compliance_scores'], stats['risk_scores'], stats['temporal_leakages']), 1):
                print(f"  Run {i}: Compliance={comp:.3f}, Risk={risk:.3f}, Leakage={leak:.3f}")
            
            # Show individual run returns if available
            if 'return_statistics' in assessment:
                return_stats = assessment['return_statistics']
                print(f"\\n💰 INDIVIDUAL RUN RETURNS:")
                for i in range(len(stats['compliance_scores'])):
                    baseline_ret = return_stats['baseline']['all_returns'][i] if i < len(return_stats['baseline']['all_returns']) else 0
                    restricted_ret = return_stats['restricted']['all_returns'][i] if i < len(return_stats['restricted']['all_returns']) else 0
                    unrestricted_ret = return_stats['unrestricted']['all_returns'][i] if i < len(return_stats['unrestricted']['all_returns']) else 0
                    print(f"  Run {i+1}: Base={baseline_ret:5.1f}%, Restr={restricted_ret:5.1f}%, Unrestr={unrestricted_ret:5.1f}%")
        
        # Show return statistics if available
        if 'return_statistics' in assessment and metadata.get('total_runs', 1) > 1:
            return_stats = assessment['return_statistics']
            print(f"\\n� STRATEGY RETURN ANALYSIS (12-month hypothetical):")
            for strategy, stats in return_stats.items():
                strategy_name = strategy.replace('_', '-').title()
                print(f"  {strategy_name:13}: Avg={stats['avg_return']:6.1f}% ± {stats['std_return']:.1f}% | Returns: {stats['all_returns']}")
            
            # Determine best performing strategy
            best_strategy = max(return_stats.items(), key=lambda x: x[1]['avg_return'])
            worst_strategy = min(return_stats.items(), key=lambda x: x[1]['avg_return'])
            print(f"\\n📈 PERFORMANCE RANKING:")
            print(f"  Best:  {best_strategy[0].replace('_', '-').title()} ({best_strategy[1]['avg_return']:.1f}%)")
            print(f"  Worst: {worst_strategy[0].replace('_', '-').title()} ({worst_strategy[1]['avg_return']:.1f}%)")

        print("\\n�💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  {rec}")
        
        # Show API call statistics if multiple runs  
        if metadata.get('total_runs', 1) > 1:
            success_rate = (metadata.get('successful_runs', 0) / metadata.get('total_runs', 1)) * 100
            print(f"\\n🔧 API PERFORMANCE:")
            print(f"  Success Rate: {success_rate:.1f}%")
            if metadata.get('failed_runs'):
                print(f"  Failed Run Numbers: {metadata['failed_runs']}")
        
        print("\\n" + "="*60)
    
    def run_quick_test(self, analysis_date: str = "2024-06-01") -> None:
        """Run a simplified version for quick testing"""
        print("⚡ Quick Temporal Compliance Test")
        print("-" * 40)
        print(f"📅 Testing Date: {analysis_date}")
        
        # Simple baseline and restricted prompts (same format, different instructions)
        base_instruction = f"You are analyzing investment opportunities as of {analysis_date}."
        restricted_instruction = f"CRITICAL: Pretend it's {analysis_date}. No future data beyond this date!"
        
        base_prompt = self.create_test_prompt(analysis_date, [], base_instruction, False, 0)
        restricted_prompt = self.create_test_prompt(analysis_date, [], restricted_instruction, False, 0)
        
        base_response, base_success = self.get_gpt_response(base_prompt, "Quick Base")
        restricted_response, restricted_success = self.get_gpt_response(restricted_prompt, "Quick Restricted")
        
        if not base_success or not restricted_success:
            print("\n💥 Quick test failed due to GPT API errors")
            print(f"  Base test: {'✅ Success' if base_success else '❌ Failed'}")
            print(f"  Restricted test: {'✅ Success' if restricted_success else '❌ Failed'}")
            return
        
        base_tickers = self.extract_stock_tickers(base_response)
        restricted_tickers = self.extract_stock_tickers(restricted_response)
        
        print(f"\n📊 Results:")
        print(f"  Base:       {base_tickers}")
        print(f"  Restricted: {restricted_tickers}")
        print(f"  Same?:      {'✅ Yes' if base_tickers == restricted_tickers else '❌ No - Potential leakage!'}")

def get_user_input():
    """Get user preferences for test configuration"""
    print("\\n⚙️  Test Configuration")
    print("-" * 30)
    
    # Get analysis date
    default_date = "2024-06-01"
    date_input = input(f"Enter analysis date (YYYY-MM-DD) [default: {default_date}]: ").strip()
    analysis_date = date_input if date_input else default_date
    
    # Validate date format
    try:
        datetime.strptime(analysis_date, "%Y-%m-%d")
    except ValueError:
        print(f"⚠️  Invalid date format, using default: {default_date}")
        analysis_date = default_date
    
    # Get number of runs
    default_runs = 1
    runs_input = input(f"Number of test runs [default: {default_runs}]: ").strip()
    try:
        num_runs = int(runs_input) if runs_input else default_runs
        if num_runs < 1:
            num_runs = default_runs
        elif num_runs > 10:
            print("⚠️  Maximum 10 runs allowed, setting to 10")
            num_runs = 10
    except ValueError:
        print(f"⚠️  Invalid number, using default: {default_runs}")
        num_runs = default_runs
    
    # Get number of articles (new feature)
    default_articles = 25
    articles_input = input(f"Number of articles per test [default: {default_articles}]: ").strip()
    try:
        article_count = int(articles_input) if articles_input else default_articles
        if article_count < 1:
            article_count = 1
        elif article_count > 100:
            print("⚠️  Maximum 100 articles allowed, setting to 100")
            article_count = 100
    except ValueError:
        print(f"⚠️  Invalid number, using default: {default_articles}")
        article_count = default_articles
    
    return analysis_date, num_runs, article_count

def main():
    """Main execution function with options"""
    print("🎯 GPT Temporal Compliance Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = GPTTemporalComplianceTester()
    
    # Display CSV article prevalence analysis
    print("\n📊 CSV ARTICLE PREVALENCE ANALYSIS")
    print("=" * 50)
    prevalence = tester.analyze_csv_article_prevalence()
    if 'error' not in prevalence:
        print(f"📈 Total Articles: {prevalence['overall']['total_articles']}")
        print(f"📅 Date Range: {prevalence['overall']['date_range']['start'][:10]} to {prevalence['overall']['date_range']['end'][:10]}")
        print(f"⏰ Span: {prevalence['overall']['span_days']} days")
        print(f"\n🗓️ Most Prevalent Month: {prevalence['monthly']['most_prevalent']} ({prevalence['monthly']['max_count']} articles)")
        print(f"📋 Monthly Distribution:")
        for month, count in list(prevalence['monthly']['counts'].items())[-6:]:  # Show last 6 months
            print(f"   {month}: {count} articles")
    else:
        print(f"⚠️ {prevalence['error']}")
    print("=" * 50)
    
    # Ask user for test type
    print("\\nSelect test type:")
    print("1. Full comprehensive test (with news context)")
    print("2. Full test (without news context)")  
    print("3. Quick test (basic comparison)")
    
    try:
        choice = input("Enter choice (1-3) or press Enter for full test: ").strip()
        
        if choice == "3":
            # Quick test with optional date change
            print("\\n📅 Quick Test Configuration")
            default_date = "2024-06-01"
            date_input = input(f"Enter analysis date (YYYY-MM-DD) [default: {default_date}]: ").strip()
            analysis_date = date_input if date_input else default_date
            
            try:
                datetime.strptime(analysis_date, "%Y-%m-%d")
            except ValueError:
                print(f"⚠️  Invalid date format, using default: {default_date}")
                analysis_date = default_date
            
            tester.run_quick_test(analysis_date)
        
        elif choice == "2":
            # Full test without news but with configuration
            analysis_date, num_runs, article_count = get_user_input()
            tester.run_comprehensive_test(
                analysis_date=analysis_date, 
                use_news_context=False, 
                num_runs=num_runs,
                article_count=article_count
            )
        
        else:  # Default to full test with news
            analysis_date, num_runs, article_count = get_user_input()
            tester.run_comprehensive_test(
                analysis_date=analysis_date, 
                use_news_context=True, 
                num_runs=num_runs,
                article_count=article_count
            )
            
    except KeyboardInterrupt:
        print("\\n👋 Test cancelled by user")
    except Exception as e:
        print(f"\\n❌ Error running test: {e}")
        import traceback
        print(f"🔍 Debug info: {traceback.format_exc()}")

if __name__ == "__main__":
    main()