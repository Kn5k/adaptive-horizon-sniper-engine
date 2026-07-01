# basically x5 but on bigger and fatter margin and more coins 

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

def run_quant_research_platform():
    print("\n" + "="*95)
    print("             QUANT ALPHA ENGINE: VERSION 4 (INSTITUTIONAL QUANT PLATFORM)          ")
    print("="*95)
    
    # 1. DEFINE UNTOUCHED UNIVERSE BASKET (Diverse 20-Asset Cross-Sectional Test)
    asset_universe = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
    ]
    
    backtest_days = int(input("📅 Enter Macro Simulation Horizon Depth in Days (e.g., 90, 180, 365): "))
    
    # Freeze Absolute Baseline Operational Parameters (Strictly No Re-tuning)
    STATIC_MEM = 50
    STATIC_CONF = 0.70
    CALIBRATION_DAYS = 15
    BASE_FEE_RATE = 0.0005
    
    current_time_anchor = datetime.now(timezone.utc)
    sim_end_dt = current_time_anchor - timedelta(hours=6)
    sim_start_dt = sim_end_dt - timedelta(days=backtest_days)
    
    start_date_pull = (sim_start_dt - timedelta(days=STATIC_MEM + CALIBRATION_DAYS + 10)).strftime('%Y-%m-%d')
    end_date_pull = (sim_end_dt + timedelta(days=2)).strftime('%Y-%m-%d')
    
    print(f"\n🚀 Initializing Cross-Sectional Invariance Suite over {len(asset_universe)} Assets...")
    print(f"📈 Timeline Phase: {start_date_pull} to {end_date_pull}")
    
    # Global Registries for Platform Metrics
    universe_summary_ledger = []
    global_regime_trades_record = []
    global_feature_gains = {"candle_return": [], "rsi": [], "macd_histogram": [], "distance_from_mean": []}
    
    for ticker in asset_universe:
        print(f" -> Processing Matrix: {ticker:<9} | Ingesting and building models...", end="\r")
        try:
            df_raw = yf.download(tickers=ticker, start=start_date_pull, end=end_date_pull, interval="1h", multi_level_index=False, progress=False)
            if df_raw.empty or len(df_raw) < 500: continue
            
            df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            
            # --- FEATURE LAYER ---
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
            
            # Slippage Framework
            high_low_range = df["High"] - df["Low"]
            high_close_shift = abs(df["High"] - df["Close"].shift(1))
            low_close_shift = abs(df["Low"] - df["Close"].shift(1))
            true_range = pd.concat([high_low_range, high_close_shift, low_close_shift], axis=1).max(axis=1)
            df["atr_pct"] = true_range.rolling(window=14).mean() / df["Close"]
            
            # --- REGIME DETECTOR LAYERS (4H and Daily Proxies smoothed into 1H space) ---
            df["macro_trend_filter"] = df["Close"].ewm(span=200, adjust=False).mean()
            df["regime_trend"] = np.where(df["Close"] > df["macro_trend_filter"], "BULLISH_TREND", "BEARISH_TREND")
            
            df["rolling_vol_std"] = df["candle_return"].rolling(window=24).std()
            df["macro_vol_median"] = df["rolling_vol_std"].rolling(window=240).median()
            df["regime_volatility"] = np.where(df["rolling_vol_std"] > df["macro_vol_median"], "HIGH_VOL", "LOW_VOL")
            
            # Targets
            df["real_future_close"] = df["Close"].shift(-24)
            df["future_target"] = 1  
            df.loc[df["real_future_close"] > df["Close"], "future_target"] = 2  
            df.loc[df["real_future_close"] < df["Close"], "future_target"] = 0  
            
            feature_cols = ["candle_return", "rsi", "macd_histogram", "distance_from_mean"]
            df_clean = df.dropna(subset=feature_cols + ["future_target", "atr_pct", "regime_trend", "regime_volatility"]).copy()
            
            sim_rows = df_clean.loc[(df_clean.index >= sim_start_dt) & (df_clean.index <= sim_end_dt)]
            if len(sim_rows) < 100: continue
            
            # Simulation Environment
            p_balance = 10000.0
            starting_cap = 10000.0
            active_expiry = None
            active_data = {}
            t_trades, t_wins = 0, 0
            
            first_px = sim_rows["Close"].iloc[0]
            last_px = sim_rows["Close"].iloc[-1]
            asset_bnh = ((last_px - first_px) / first_px) * 100
            
            for step_idx in range(len(sim_rows)):
                current_time = sim_rows.index[step_idx]
                
                # Close Trade Sequence
                if active_expiry is not None and current_time >= active_expiry:
                    exit_px_raw = df_clean.loc[current_time, "Close"]
                    slip = df_clean.loc[current_time, "atr_pct"] * 0.10
                    
                    if active_data["Type"] == "LONG":
                        exit_px = exit_px_raw * (1.0 - slip)
                        net_move = (exit_px - active_data["Entry_Px"]) / active_data["Entry_Px"]
                    else:
                        exit_px = exit_px_raw * (1.0 + slip)
                        net_move = (active_data["Entry_Px"] - exit_px) / active_data["Entry_Px"]
                        
                    pnl_usd = (10000.0 * net_move) - (10000.0 * BASE_FEE_RATE * 2)
                    p_balance += pnl_usd
                    t_trades += 1
                    is_win = 1 if pnl_usd > 0 else 0
                    if is_win: t_wins += 1
                    
                    # Store trade telemetry mapped to the entry environment labels
                    global_regime_trades_record.append({
                        "Ticker": ticker, "PnL": pnl_usd, "Win": is_win,
                        "Trend": active_data["Env_Trend"], "Vol": active_data["Env_Vol"]
                    })
                    active_expiry = None
                    active_data = {}
                
                # Ingest & Model Fit Slices
                historical_pool = df_clean.loc[df_clean.index < current_time]
                train_block = historical_pool.iloc[-int((STATIC_MEM + CALIBRATION_DAYS) * 24): -int(CALIBRATION_DAYS * 24)]
                calib_block = historical_pool.iloc[-int(CALIBRATION_DAYS * 24):]
                
                if len(train_block) < 200 or len(calib_block) < 100: continue
                
                base_model = lgb.LGBMClassifier(
                    objective="multiclass", num_class=3, class_weight="balanced",
                    n_estimators=45, learning_rate=0.03, num_leaves=15, max_depth=4, random_state=42, verbosity=-1
                )
                base_model.fit(train_block[feature_cols], train_block["future_target"])
                
                # Record Feature Importance Drift Telemetry
                gains = base_model.booster_.feature_importance(importance_type='gain')
                for f_idx, f_name in enumerate(feature_cols):
                    global_feature_gains[f_name].append(gains[f_idx])
                
                # Probability Calibration
                calib_raw_probs = base_model.predict_proba(calib_block[feature_cols])
                platt_long = LogisticRegression(C=1e5).fit(calib_raw_probs[:, [2]], (calib_block["future_target"] == 2).astype(int))
                platt_short = LogisticRegression(C=1e5).fit(calib_raw_probs[:, [0]], (calib_block["future_target"] == 0).astype(int))
                
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
                    entry_slip = sim_rows["atr_pct"].values[step_idx] * 0.10
                    raw_entry = sim_rows["Close"].values[step_idx]
                    real_entry = raw_entry * (1.0 + entry_slip) if direction == "LONG" else raw_entry * (1.0 - entry_slip)
                    
                    active_expiry = current_time + timedelta(hours=24)
                    active_data = {
                        "Type": direction, "Entry_Px": real_entry,
                        "Env_Trend": sim_rows["regime_trend"].values[step_idx],
                        "Env_Vol": sim_rows["regime_volatility"].values[step_idx]
                    }
                    
            strategy_yield_pct = ((p_balance - starting_cap) / starting_cap) * 100
            precision_pct = (t_wins / t_trades) * 100 if t_trades > 0 else 0.0
            alpha_vs_bnh = strategy_yield_pct - asset_bnh
            
            universe_summary_ledger.append({
                "Ticker": ticker, "Yield": strategy_yield_pct, "Precision": precision_pct,
                "Trades": t_trades, "BNH_Return": asset_bnh, "Alpha": alpha_vs_bnh
            })
            
        except Exception as e:
            continue

    # --- COMPILE UNIVERSAL CROSS-SECTIONAL RESEARCH REPORT ---
    df_universe = pd.DataFrame(universe_summary_ledger)
    df_regimes = pd.DataFrame(global_regime_trades_record)
    
    print("\n" + "="*95)
    print("                    🏆 PLATFORM REPORT: CROSS-SECTIONAL INDEPENDENT AUDIT          ")
    print("="*95)
    print(f"📊 BASKET DISTRIBUTION METRICS | Total Verified Universe Assets: {len(df_universe)}")
    profitable_assets_count = len(df_universe[df_universe["Yield"] > 0])
    profitable_ratio = (profitable_assets_count / len(df_universe)) * 100
    print(f"🔥 UNIVERSE BREADTH EFFICIENCY | {profitable_assets_count}/{len(df_universe)} Assets Profitable ({profitable_ratio:.2f}% Hit Rate)")
    print(f"⚔️  AVERAGE OUT-OF-SAMPLE ALPHA | {df_universe['Alpha'].mean():+.2f}% Generation vs. Buy-and-Hold Baseline")
    print("-" * 95)
    
    print(f"{'Ticker':<10} | {'Strategy Yield':<15} | {'Precision':<11} | {'Volume':<8} | {'Benchmark (B&H)':<16} | {'Net Alpha':<10}")
    print("-" * 95)
    for _, row in df_universe.iterrows():
        print(f"{row['Ticker']:<10} | {row['Yield']:+14.2f}% | {row['Precision']:10.1f}% | {int(row['Trades']):<8} | {row['BNH_Return']:+15.2f}% | {row['Alpha']:+10.2f}%")
    print("-" * 95)
    
    # Render Macro Regime Deep-Dive Report
    print("🧠 MARKET ENVIRONMENT REGIME PROFILE')")
    if not df_regimes.empty:
        regime_perf = df_regimes.groupby(['Trend', 'Vol']).agg(
            Total_Trades=('Win', 'count'),
            Win_Rate=('Win', 'mean'),
            Net_USD_PnL=('PnL', 'sum')
        )
        print(regime_perf.to_string())
    else:
        print(" -> Insufficient trade volume distributions to extract environmental characteristics.")
    print("-" * 95)
    
    # Render Feature Power Analytics
    print("🧬 STRUCTURAL FEATURE EXPLAINABILITY (GAIN IMPORTANCE VALUES OVER TIME)")
    for feat, gain_list in global_feature_gains.items():
        avg_gain_power = np.mean(gain_list) if gain_list else 0.0
        print(f" * {feat:<20} -> Average Retraining Information Gain: {avg_gain_power:.4f}")
    print("="*95 + "\n")

if __name__ == "__main__":
    run_quant_research_platform();