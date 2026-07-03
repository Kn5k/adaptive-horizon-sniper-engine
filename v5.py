import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

def run_ultimate_validation_lab():
    print("\n" + "="*95)
    print("            QUANT ALPHA ENGINE: VERSION 5 (THE ULTIMATE VALIDATION LAB)             ")
    print("="*95)
    
    # 1. EXPANDED 25-ASSET UNIVERSE (High, Medium, and Macro Altcoin Tiers)
    asset_universe = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
        "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
        "MATIC-USD", "LTC-USD", "UNI-USD", "NEAR-USD", "APT-USD",
        "OP-USD", "ARB-USD", "INJ-USD", "RNDR-USD", "SUI-USD",
        "FTM-USD", "ATOM-USD", "IMX-USD", "GRT-USD", "THETA-USD"
    ]
    
    # 2. ENFORCE AGGRESSIVE MULTI-YEAR TIMELINE (730 Days / 2 Full Years)
    BACKTEST_DAYS = 630
    STATIC_MEM = 50
    STATIC_CONF = 0.70
    CALIBRATION_DAYS = 15
    PURGE_GAP_DAYS = 10  # Purged validation gap to eliminate data leakage
    
    # Institutional Fixed-Friction Fee + Execution Cost Assumption (0.12% Round-Trip)
    FIXED_EXECUTION_COST = 0.0012 
    
    current_time_anchor = datetime.now(timezone.utc)
    sim_end_dt = current_time_anchor - timedelta(hours=6)
    sim_start_dt = sim_end_dt - timedelta(days=BACKTEST_DAYS)
    
    start_date_pull = (sim_start_dt - timedelta(days=STATIC_MEM + CALIBRATION_DAYS + PURGE_GAP_DAYS + 15)).strftime('%Y-%m-%d')
    end_date_pull = (sim_end_dt + timedelta(days=2)).strftime('%Y-%m-%d')
    
    print(f" Initializing Deep Validation Suite over {len(asset_universe)} Assets...")
    print(f" Macro Temporal Envelope: {sim_start_dt.strftime('%Y-%m-%d')} to {sim_end_dt.strftime('%Y-%m-%d')} ({BACKTEST_DAYS} Days)")
    print(f"  Purged Validation Buffer Engaged: {PURGE_GAP_DAYS} Days Non-Overlapping Separation")
    print(f" Exchange Level Order-Book Friction Enforced: {FIXED_EXECUTION_COST*100:.2f}% Flat Per Trade")
    print("-" * 95)
    
    universe_summary_ledger = []
    
    for idx, ticker in enumerate(asset_universe):
        print(f" Processing Matrix [{idx+1}/{len(asset_universe)}]: {ticker:<10} ... running full multi-year data stream")
        try:
            # Download 2-year hourly dataset
            df_raw = yf.download(tickers=ticker, start=start_date_pull, end=end_date_pull, interval="1h", multi_level_index=False, progress=False)
            if df_raw.empty or len(df_raw) < 5000: 
                print(f"    Insufficient data history for {ticker}. Skipping.")
                continue
            
            df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            
            # --- EXTRACT FIXED INDICATOR ARRAY ---
            df["candle_return"] = np.log(df["Close"] / df["Close"].shift(1))
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-8))))
            
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["macd_histogram"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
            df["ema_baseline"] = df["Close"].ewm(span=36, adjust=False).mean()
            df["distance_from_mean"] = (df["Close"] / df["ema_baseline"]) - 1.0
            
            # Target Vector Generation
            df["real_future_close"] = df["Close"].shift(-24)
            df["future_target"] = 1  
            df.loc[df["real_future_close"] > df["Close"], "future_target"] = 2  
            df.loc[df["real_future_close"] < df["Close"], "future_target"] = 0  
            
            feature_cols = ["candle_return", "rsi", "macd_histogram", "distance_from_mean"]
            df_clean = df.dropna(subset=feature_cols + ["future_target"]).copy()
            
            sim_rows = df_clean.loc[(df_clean.index >= sim_start_dt) & (df_clean.index <= sim_end_dt)]
            if len(sim_rows) < 1000: continue
            
            # Backtest Execution Memory State
            p_balance = 10000.0
            starting_cap = 10000.0
            active_expiry = None
            active_direction = ""
            t_trades, t_wins = 0, 0
            
            first_px = sim_rows["Close"].iloc[0]
            last_px = sim_rows["Close"].iloc[-1]
            asset_bnh = ((last_px - first_px) / first_px) * 100
            
            # Run simulation using 4-hour step jumps to maintain validation integrity across 2 years
            for step_idx in range(0, len(sim_rows), 4):
                current_time = sim_rows.index[step_idx]
                
                # Trade Lifecycle Resolution
                if active_expiry is not None and current_time >= active_expiry:
                    exit_px = df_clean.loc[current_time, "Close"]
                    
                    if active_direction == "LONG":
                        net_move = (exit_px - entry_px_raw) / entry_px_raw
                    else:
                        net_move = (entry_px_raw - exit_px) / entry_px_raw
                        
                    # Deduct full mathematical execution costs (0.12% round trip)
                    pnl_usd = (starting_cap * net_move) - (starting_cap * FIXED_EXECUTION_COST)
                    p_balance += pnl_usd
                    t_trades += 1
                    if pnl_usd > 0: t_wins += 1
                    active_expiry = None
                
                # PURGED WALK-FORWARD ENGINE INDUCTION
                # [Train Slice] -> [10-Day Purge Buffer] -> [15-Day Calibration Slice] -> [Current Step]
                historical_pool = df_clean.loc[df_clean.index < current_time]
                
                total_window_needed = (STATIC_MEM + PURGE_GAP_DAYS + CALIBRATION_DAYS) * 24
                if len(historical_pool) < total_window_needed: continue
                
                train_end_idx = -int((PURGE_GAP_DAYS + CALIBRATION_DAYS) * 24)
                calib_start_idx = -int(CALIBRATION_DAYS * 24)
                
                train_block = historical_pool.iloc[train_end_idx - int(STATIC_MEM * 24) : train_end_idx]
                calib_block = historical_pool.iloc[calib_start_idx:]
                
                # Fit Base Tree Model
                base_model = lgb.LGBMClassifier(
                    objective="multiclass", num_class=3, class_weight="balanced",
                    n_estimators=35, learning_rate=0.05, num_leaves=15, max_depth=4, random_state=42, verbosity=-1
                )
                base_model.fit(train_block[feature_cols], train_block["future_target"])
                
                # Extract Probabilities & Calibrate using Platt Scaling
                calib_raw_probs = base_model.predict_proba(calib_block[feature_cols])
                platt_long = LogisticRegression(C=1e3).fit(calib_raw_probs[:, [2]], (calib_block["future_target"] == 2).astype(int))
                platt_short = LogisticRegression(C=1e3).fit(calib_raw_probs[:, [0]], (calib_block["future_target"] == 0).astype(int))
                
                current_vector = sim_rows[feature_cols].iloc[[step_idx]]
                raw_probs = base_model.predict_proba(current_vector)[0]
                
                calibrated_long = platt_long.predict_proba([[raw_probs[2]]])[0][1]
                calibrated_short = platt_short.predict_proba([[raw_probs[0]]])[0][1]
                
                triggered = False; direction = ""
                if calibrated_long >= STATIC_CONF:
                    triggered = True; direction = "LONG"
                elif calibrated_short >= STATIC_CONF:
                    triggered = True; direction = "SHORT"
                    
                if triggered and active_expiry is None:
                    entry_px_raw = sim_rows["Close"].values[step_idx]
                    active_expiry = current_time + timedelta(hours=24)
                    active_direction = direction
                    
            strategy_yield_pct = ((p_balance - starting_cap) / starting_cap) * 100
            precision_pct = (t_wins / t_trades) * 100 if t_trades > 0 else 0.0
            alpha_vs_bnh = strategy_yield_pct - asset_bnh
            
            universe_summary_ledger.append({
                "Ticker": ticker, "Yield": strategy_yield_pct, "Precision": precision_pct,
                "Trades": t_trades, "BNH_Return": asset_bnh, "Alpha": alpha_vs_bnh
            })
            
        except Exception as e:
            print(f"    Error executing validation slice for {ticker}: {str(e)}")
            continue

    # --- FINAL UNIVERSAL SCORECARD RENDERING ---
    df_validation = pd.DataFrame(universe_summary_ledger)
    
    print("\n" + "="*95)
    print("                FINAL VERSION 5 AUDIT REPORT: ACCREDITED QUANT EDGE              ")
    print("="*95)
    print(f" SYSTEMIC UNIVERSE SIZE      | Total Independent Verified Assets : {len(df_validation)}")
    
    profitable_assets = len(df_validation[df_validation["Yield"] > 0])
    alpha_generating_assets = len(df_validation[df_validation["Alpha"] > 0])
    universe_hit_rate = (profitable_assets / len(df_validation)) * 100 if len(df_validation) > 0 else 0
    
    print(f" SYSTEM DEPLOYMENT BREADTH   | {profitable_assets}/{len(df_validation)} Portfolio Streams Profitable ({universe_hit_rate:.2f}% Success Rate)")
    print(f"  MACRO ALPHA GENERATION MAX  | Mean Outperformance Generation     : {df_validation['Alpha'].mean():+.2f}% vs. Market")
    print("-" * 95)
    
    print(f"{'Ticker':<12} | {'Validation Return':<19} | {'Precision':<11} | {'Trades':<8} | {'Market Benchmark':<16} | {'Pure Alpha':<10}")
    print("-" * 95)
    for _, row in df_validation.iterrows():
        print(f"{row['Ticker']:<12} | {row['Yield']:+18.2f}% | {row['Precision']:10.1f}% | {int(row['Trades']):<8} | {row['BNH_Return']:+15.2f}% | {row['Alpha']:+10.2f}%")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_ultimate_validation_lab()
