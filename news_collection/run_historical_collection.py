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

def run_historical_collection():
    print("HISTORICAL NEWS COLLECTION - AUTOMATED RUN")
    print("=" * 55)
    
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    enhancer = HistoricalNewsEnhancer(openai_api_key=openai_key)
    
    print("Loading existing database...")
    enhancer.existing_df = enhancer._load_existing_database()
    print(f"Loaded: {len(enhancer.existing_df)} existing articles")
    
    print("\nStarting historical collection...")
    print("Target: January 2020 - February 2020 (2 months)")
    print("Minimum filtered articles per month: 5")
    
    # Run the historical data collection
    try:
        enhanced_df = enhancer._collect_historical_data(
            start_year=2020, 
            start_month=1, 
            end_year=2020, 
            end_month=2, 
            min_articles_per_month=5
        )
        
        print(f"\nCollection completed!")
        print(f"Total articles in enhanced database: {len(enhanced_df)}")
        print(f"New articles added: {len(enhanced_df) - len(enhancer.existing_df)}")
        
        # Save the enhanced database
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"historical_enhanced_news_{timestamp}.csv"
        enhanced_df.to_csv(output_file, index=False)
        print(f"Enhanced database saved: {output_file}")
        
        return enhanced_df
        
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
        return enhancer.existing_df
    except Exception as e:
        print(f"\nError during collection: {e}")
        return enhancer.existing_df

if __name__ == "__main__":
    result = run_historical_collection()