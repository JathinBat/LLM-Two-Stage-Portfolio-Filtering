import sys
import os

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

# Add the BASE directory to the Python path
BASE_DIR = r"C:\Users\jathi\Documents\ASDRP-LLM-Long-Term-Investment-Strategy\BASE"
sys.path.append(BASE_DIR)

from historical_news_enhancer import HistoricalNewsEnhancer
import pandas as pd

def run_full_historical_collection():
    print("🚀 FULL HISTORICAL COLLECTION - 2020 to June 2025")
    print("=" * 65)
    print("Tech & Finance Keywords: Apple, Microsoft, Google, Amazon, Tesla, etc.")
    print("Target: 10 FILTERED articles per month minimum")
    print("Period: January 2020 to June 2025 (65 months)")
    print("=" * 65)
    
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    enhancer = HistoricalNewsEnhancer(openai_api_key=openai_key)
    
    print("Loading existing database...")
    enhancer.existing_df = enhancer._load_existing_database()
    print(f"Loaded: {len(enhancer.existing_df)} existing articles")
    
    print("\n🎯 COLLECTION STARTING...")
    print("Note: Collection will save progress after each month")
    print("You can safely interrupt and resume if needed")
    print("-" * 65)
    
    # Run the full historical collection
    try:
        new_articles = enhancer._collect_historical_data(
            start_year=2020, 
            start_month=1, 
            end_year=2025, 
            end_month=6,  # January 2020 to June 2025
            min_articles_per_month=10  # 10 filtered articles per month
        )
        
        # Combine with existing data
        if new_articles:
            new_df = pd.DataFrame(new_articles)
            enhanced_df = enhancer._merge_databases(enhancer.existing_df, new_df)
        else:
            enhanced_df = enhancer.existing_df
        
        print(f"\n🎉 FULL COLLECTION COMPLETED!")
        print(f"="*50)
        print(f"Total articles in database: {len(enhanced_df)}")
        print(f"New historical articles: {len(new_articles) if new_articles else 0}")
        print(f"Period covered: January 2020 - June 2025")
        print(f"="*50)
        
        # Save the final enhanced database
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"FINAL_historical_enhanced_2020_2025_{timestamp}.csv"
        enhanced_df.to_csv(output_file, index=False)
        print(f"🎯 FINAL DATABASE SAVED: {output_file}")
        
        return enhanced_df
        
    except KeyboardInterrupt:
        print("\n⏹️ Collection interrupted by user")
        print("Progress has been saved in monthly checkpoint files")
        return enhancer.existing_df
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("Check monthly checkpoint files for partial progress")
        return enhancer.existing_df

if __name__ == "__main__":
    print("Starting in 3 seconds... Press Ctrl+C to cancel")
    import time
    time.sleep(3)
    result = run_full_historical_collection()