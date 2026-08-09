  #!/usr/bin/env python3
"""
GPT Temporal Compliance Tester (Cheat Tester)

Comprehensive tool to test if GPT models use future data inappropriately when analyzing 
historical periods. This module tests temporal boundaries to detect "data leakage" and 
ensures investment analyses don't rely on information that wouldn't have been available 
at the time.

Merged and enhanced with complete functionality for investment analysis validation.
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
        # Use your working API keys
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.nyt_api_key = nyt_api_key or os.environ.get("NYT_API_KEY", "")
        
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        self.nyt_base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
        
    def fetch_news_context(self, start_date: str, end_date: str, query: str = "technology stocks") -> List[Dict]:
        """Fetch the top article for every week of the time period from NYTimes API"""
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            all_articles = []
            current_week_start = start_dt
            
            print(f"📅 Fetching top article for each week from {start_date} to {end_date}")
            
            week_count = 0
            while current_week_start <= end_dt:
                # Calculate week end (Sunday to Saturday)
                current_week_end = min(current_week_start + timedelta(days=6), end_dt)
                
                week_start_str = current_week_start.strftime("%Y-%m-%d")
                week_end_str = current_week_end.strftime("%Y-%m-%d")
                
                print(f"🔍 Week {week_count + 1}: {week_start_str} to {week_end_str}")
                
                # Fetch articles for this specific week
                try:
                    params = {
                        'q': query,
                        'begin_date': week_start_str.replace('-', ''),
                        'end_date': week_end_str.replace('-', ''),
                        'sort': 'relevance',  # Get most relevant article for the week
                        'api-key': self.nyt_api_key,
                        'page': 0,  # Only first page to get top article
                        'fl': 'headline,pub_date,snippet,web_url,section_name,news_desk'
                    }
                    
                    response = requests.get(self.nyt_base_url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if 'response' in data and 'docs' in data['response'] and data['response']['docs']:
                        # Get the top article from this week
                        top_article = data['response']['docs'][0]
                        
                        article_data = {
                            'headline': top_article.get('headline', {}).get('main', ''),
                            'pub_date': top_article.get('pub_date', ''),
                            'snippet': top_article.get('snippet', ''),
                            'web_url': top_article.get('web_url', ''),
                            'section': top_article.get('section_name', ''),
                            'news_desk': top_article.get('news_desk', ''),
                            'source': 'NY Times API',
                            'week_period': f"{week_start_str} to {week_end_str}"
                        }
                        
                        all_articles.append(article_data)
                        print(f"✅ Found article: {article_data['headline'][:60]}...")
                    else:
                        print(f"⚠️  No articles found for week {week_start_str} to {week_end_str}")
                
                except Exception as week_error:
                    error_str = str(week_error)
                    if "429" in error_str or "Too Many Requests" in error_str:
                        print(f"⏳ Rate limit hit for week {week_start_str}, waiting 60 seconds...")
                        time.sleep(60)
                        # Retry once after rate limit
                        try:
                            response = requests.get(self.nyt_base_url, params=params)
                            response.raise_for_status()
                            data = response.json()
                            
                            if 'response' in data and 'docs' in data['response'] and data['response']['docs']:
                                top_article = data['response']['docs'][0]
                                
                                article_data = {
                                    'headline': top_article.get('headline', {}).get('main', ''),
                                    'pub_date': top_article.get('pub_date', ''),
                                    'snippet': top_article.get('snippet', ''),
                                    'web_url': top_article.get('web_url', ''),
                                    'section': top_article.get('section_name', ''),
                                    'news_desk': top_article.get('news_desk', ''),
                                    'source': 'NY Times API',
                                    'week_period': f"{week_start_str} to {week_end_str}"
                                }
                                
                                all_articles.append(article_data)
                                print(f"✅ Retry successful: {article_data['headline'][:60]}...")
                            else:
                                print(f"⚠️  No articles found for week {week_start_str} to {week_end_str} (after retry)")
                        except Exception as retry_error:
                            print(f"❌ Retry failed for week {week_start_str}: {retry_error}")
                    else:
                        print(f"❌ Error fetching week {week_start_str}: {week_error}")
                
                # Move to next week
                current_week_start += timedelta(days=7)
                week_count += 1
                
                # Be respectful to the API - rate limiting
                # NYTimes API allows 4000 requests per day, 10 per minute
                if week_count % 10 == 0:
                    print(f"⏳ Processed {week_count} weeks, waiting 60 seconds for rate limit...")
                    time.sleep(60)  # Wait 1 minute every 10 requests
                else:
                    time.sleep(6)  # 6 second delay to stay under 10 requests per minute
            
            print(f"✅ Retrieved {len(all_articles)} weekly top articles from NY Times API")
            
            # Also load articles from consolidated CSV if it exists for additional context
            csv_articles = self.load_articles_from_csv(start_date, end_date)
            if csv_articles:
                # Limit CSV articles to avoid overwhelming with data
                csv_articles = csv_articles[:20]  # Take top 20 from CSV
                all_articles.extend(csv_articles)
                print(f"✅ Added {len(csv_articles)} articles from local CSV")
            
            # Remove duplicates based on headline
            seen_headlines = set()
            unique_articles = []
            for article in all_articles:
                headline = article.get('headline', '').lower().strip()
                if headline and headline not in seen_headlines:
                    seen_headlines.add(headline)
                    unique_articles.append(article)
            
            print(f"📰 Total unique articles for context: {len(unique_articles)}")
            print(f"📊 Coverage: {week_count} weeks analyzed")
            return unique_articles
            
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch weekly news data: {e}")
            return []
    
    def load_articles_from_csv(self, start_date: str, end_date: str, 
                              filename: str = "consolidated_news_data.csv") -> List[Dict]:
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
            
            return articles[:50]  # Limit to 50 most relevant articles from CSV (more for year-long range)
            
        except Exception as e:
            print(f"⚠️  Could not load articles from CSV: {e}")
            return []
    
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
    
    def create_test_prompt(self, analysis_date: str, articles: List[Dict], 
                          temporal_instruction: str, include_news: bool = True, 
                          article_count: int = 25) -> str:
        """Create a testing prompt with optional news context"""
        
        news_context = ""
        if include_news and articles:
            news_context = "\n\nComprehensive Market Context and News Analysis:\n"
            news_context += "="*60 + "\n"
            
            # Use configurable number of articles for context
            article_limit = min(len(articles), article_count)  # Use up to specified articles
            
            for i, article in enumerate(articles[:article_limit], 1):
                pub_date = article.get('pub_date', '')[:10] if article.get('pub_date') else 'Unknown'
                headline = article.get('headline', 'N/A')
                snippet = article.get('snippet', '')
                
                news_context += f"{i}. [{pub_date}] {headline}\n"
                if snippet:
                    # Include full snippet, not truncated
                    news_context += f"   Summary: {snippet}\n"
                news_context += f"   URL: {article.get('web_url', 'N/A')}\n"
                news_context += "-" * 40 + "\n"
            
            news_context += f"\nTotal articles analyzed: {article_limit} from {len(articles)} available\n"
            news_context += "="*60 + "\n"
        
        prompt = f"""
{temporal_instruction}

ANALYSIS DATE: {analysis_date}
{news_context}

TASK: Based on the comprehensive market context and news analysis provided above, recommend 3 technology stocks for investment as of {analysis_date}.

Consider:
- Market trends and sentiment from the news articles
- Company-specific developments and announcements  
- Industry dynamics and competitive landscape
- Financial performance indicators mentioned in articles
- Regulatory or policy changes affecting the sector

Provide thoughtful analysis based on the extensive information available as of {analysis_date}.

Respond ONLY in this JSON format:
[
    {{"ticker": "SYMBOL", "company": "Company Name", "reason": "Investment rationale based on available information (max 100 words)"}},
    {{"ticker": "SYMBOL", "company": "Company Name", "reason": "Investment rationale based on available information (max 100 words)"}},
    {{"ticker": "SYMBOL", "company": "Company Name", "reason": "Investment rationale based on available information (max 100 words)"}}
]
        """
        
        return prompt.strip()
    
    def get_gpt_response(self, prompt: str, test_name: str, retry_count: int = 3) -> tuple:
        """Get response from GPT with enhanced error handling and retry logic"""
        for attempt in range(retry_count):
            try:
                print(f"🔄 Running {test_name} test... (Attempt {attempt + 1}/{retry_count})")
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500  # Increased for larger context processing
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
    
    def extract_stock_tickers(self, response: str) -> List[str]:
        """Extract stock tickers from GPT response"""
        try:
            # Try to parse JSON
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                tickers = []
                for item in data:
                    if isinstance(item, dict) and 'ticker' in item:
                        ticker = item['ticker'].upper().strip()
                        if ticker and len(ticker) <= 5:  # Valid stock ticker format
                            tickers.append(ticker)
                return tickers
            else:
                # Fallback: extract ticker-like patterns
                tickers = re.findall(r'\\b[A-Z]{2,5}\\b', response)
                return list(set(tickers[:5]))  # Limit to prevent noise
        except Exception as e:
            print(f"⚠️  Error extracting tickers from response: {e}")
            return []
    
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
                                          unrestricted: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive temporal compliance assessment"""
        
        # Calculate pairwise similarities
        base_restrict = self.calculate_similarity_metrics(baseline, restricted)
        base_unrestrict = self.calculate_similarity_metrics(baseline, unrestricted)
        restrict_unrestrict = self.calculate_similarity_metrics(restricted, unrestricted)
        
        # Compliance score: How well restricted matches baseline vs unrestricted
        compliance_score = base_restrict['jaccard']
        
        # Temporal leakage: Difference between restricted and unrestricted responses
        temporal_leakage = abs(base_restrict['jaccard'] - base_unrestrict['jaccard'])
        
        # Overall risk calculation
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
            'detailed_metrics': {
                'baseline_vs_restricted': base_restrict,
                'baseline_vs_unrestricted': base_unrestrict,
                'restricted_vs_unrestricted': restrict_unrestrict
            }
        }
    
    def run_comprehensive_test(self, analysis_date: str = "2024-06-01", 
                              use_news_context: bool = True, 
                              save_results: bool = True,
                              num_runs: int = 1,
                              article_count: int = 25) -> Dict[str, Any]:
        """Run the complete temporal compliance test suite"""
        
        print("🎯 GPT Temporal Compliance Test")
        print("=" * 50)
        print(f"📅 Testing Date: {analysis_date}")
        print(f"📰 Using Comprehensive News Context: {use_news_context}")
        print(f"🔄 Number of Runs: {num_runs}")
        print(f"📊 Articles per Test: {article_count if use_news_context else 'N/A'}")
        
        # Fetch comprehensive news context if requested
        articles = []
        if use_news_context:
            analysis_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
            start_date = (analysis_dt - timedelta(days=365)).strftime("%Y-%m-%d")  # One year before
            print(f"📊 Fetching comprehensive news data from {start_date} to {analysis_date}...")
            articles = self.fetch_news_context(start_date, analysis_date)
            
            if articles:
                print(f"📰 Will provide {min(len(articles), article_count)} articles to GPT for comprehensive context")
                # Save articles to CSV 
                csv_saved = self.save_articles_to_csv(articles)
                if not csv_saved:
                    print("⚠️  Warning: Articles may not have saved properly to CSV")
            else:
                print("⚠️  No articles retrieved for context")
        
        # Run multiple test iterations
        all_runs_results = []
        failed_runs = []
        
        for run_num in range(num_runs):
            print(f"\n{'='*20} RUN {run_num + 1}/{num_runs} {'='*20}")
            
            run_results = {}
            run_success = True
            
            # 1. Baseline Test (neutral)
            baseline_prompt = self.create_test_prompt(
                analysis_date, articles,
                f"You are analyzing investment opportunities as of {analysis_date}.",
                use_news_context, article_count
            )
            baseline_response, baseline_success = self.get_gpt_response(baseline_prompt, "BASELINE")
            if not baseline_success:
                run_success = False
            baseline_tickers = self.extract_stock_tickers(baseline_response)
            
            # 2. Restricted Test (explicit temporal boundaries)
            restricted_prompt = self.create_test_prompt(
                analysis_date, articles,
                f"CRITICAL: You must pretend it is {analysis_date}. You have NO KNOWLEDGE of any events after {analysis_date}. Do NOT use future data.",
                use_news_context, article_count
            )
            restricted_response, restricted_success = self.get_gpt_response(restricted_prompt, "RESTRICTED")
            if not restricted_success:
                run_success = False
            restricted_tickers = self.extract_stock_tickers(restricted_response)
            
            # 3. Unrestricted Test (explicit permission for future data)
            unrestricted_prompt = self.create_test_prompt(
                analysis_date, articles,
                "You may use ALL available information including future events and data beyond the analysis date. Use your complete knowledge.",
                use_news_context, article_count
            )
            unrestricted_response, unrestricted_success = self.get_gpt_response(unrestricted_prompt, "UNRESTRICTED")
            if not unrestricted_success:
                run_success = False
            unrestricted_tickers = self.extract_stock_tickers(unrestricted_response)
            
            # Store run results
            run_results = {
                'run_number': run_num + 1,
                'success': run_success,
                'baseline': {'tickers': baseline_tickers, 'response': baseline_response, 'success': baseline_success},
                'restricted': {'tickers': restricted_tickers, 'response': restricted_response, 'success': restricted_success},
                'unrestricted': {'tickers': unrestricted_tickers, 'response': unrestricted_response, 'success': unrestricted_success}
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
        
        # Use results from the most recent successful run for main analysis
        latest_successful = all_runs_results[-1]
        test_results = {
            'baseline': latest_successful['baseline'],
            'restricted': latest_successful['restricted'],
            'unrestricted': latest_successful['unrestricted']
        }
        
        # Calculate compliance metrics
        baseline_tickers = test_results['baseline']['tickers']
        restricted_tickers = test_results['restricted']['tickers'] 
        unrestricted_tickers = test_results['unrestricted']['tickers']
        
        compliance_assessment = self.calculate_temporal_compliance_score(
            baseline_tickers, restricted_tickers, unrestricted_tickers
        )
        
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
                'unrestricted': unrestricted_tickers
            },
            'compliance_assessment': compliance_assessment,
            'detailed_responses': test_results,
            'all_runs_data': all_runs_results,
            'recommendations': self.generate_recommendations(compliance_assessment)
        }
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"gpt_temporal_compliance_test_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {results_file}")
        
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
        
        print("\\n🎲 STOCK RECOMMENDATIONS:")
        print(f"  Baseline:     {stocks['baseline']}")
        print(f"  Restricted:   {stocks['restricted']}")
        print(f"  Unrestricted: {stocks['unrestricted']}")
        
        print("\\n📈 COMPLIANCE ASSESSMENT:")
        print(f"  Risk Level: {assessment['risk_level']}")
        print(f"  Risk Score: {assessment['risk_score']}")
        print(f"  Compliance Score: {assessment['compliance_score']}")
        print(f"  Temporal Leakage: {assessment['temporal_leakage']}")
        print(f"  Assessment: {assessment['description']}")
        
        print("\\n💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  {rec}")
        
        # Show API call statistics if multiple runs
        if metadata.get('total_runs', 1) > 1:
            success_rate = (metadata.get('successful_runs', 0) / metadata.get('total_runs', 1)) * 100
            print(f"\\n📊 API PERFORMANCE:")
            print(f"  Success Rate: {success_rate:.1f}%")
            if metadata.get('failed_runs'):
                print(f"  Failed Run Numbers: {metadata['failed_runs']}")
        
        print("\\n" + "="*60)
    
    def run_quick_test(self, analysis_date: str = "2024-06-01") -> None:
        """Run a simplified version for quick testing"""
        print("⚡ Quick Temporal Compliance Test")
        print("-" * 40)
        print(f"📅 Testing Date: {analysis_date}")
        
        # Simple prompts without news context
        base_prompt = f"Recommend 3 tech stocks for investment as of {analysis_date}. JSON format only."
        restricted_prompt = f"CRITICAL: Pretend it's {analysis_date}. No future data! Recommend 3 tech stocks. JSON format only."
        
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