import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta, timezone

def run_robustness_matrix_engine():
    print("\n" + "="*95)
    print("      QUANT ALPHA ENGINE: VERSION 3 (MODULE B - PARAMETER ROBUSTNESS MATRIX)       ")
    print("="*95)
    
    target_coin = input(" Enter Single Ticker to Hunt (e.g., BTC-USD, ETH-USD, SOL-USD): ").strip().upper()
    starting_capital = 10000.0
    allocation_per_trade = 10000.0
    backtest_days = int(input(" Enter Live Simulation Window Depth (e.g., 30, 45 days): "))
    
    # Define Parameter Sweep Dimensions
    memory_options = [30, 40, 50, 60]
    confidence_options = [0.68, 0.70, 0.72, 0.75]
    
    calibration_days = 15
    current_time_anchor = datetime.now(timezone.utc)
    sim_end_dt = current_time_anchor - timedelta(hours=6)
    sim_start_dt = sim_end_dt - timedelta(days=backtest_days)
    
    # Maximum data lookback buffer required for the largest loop configuration
    max_history_days = max(memory_options) + calibration_days + backtest_days + 5
    start_date_pull = (sim_start_dt - timedelta(days=max(memory_options) + calibration_days + 5)).strftime('%Y-%m-%d')
    end_date_pull = (sim_end_dt + timedelta(days=2)).strftime('%Y-%m-%d')
    
    print(f"\n[1/2] Ingesting macro matrix block from {start_date_pull}...")
    df_raw = yf.download(tickers=target_coin, start=start_date_pull, end=end_date_pull, interval="1h", multi_level_index=False, progress=False)
    
    if df_raw.empty or len(df_raw) < 500:
        print(" Operational Failure: Insufficient data packet streamed.")
        return
        
    df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    
    # Technical Feature Pipeline
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
    
    # ATR Volatility Proxy for Slippage Modeling
    high_low_range = df["High"] - df["Low"]
    high_close_shift = abs(df["High"] - df["Close"].shift(1))
    low_close_shift = abs(df["Low"] - df["Close"].shift(1))
    true_range = pd.concat([high_low_range, high_close_shift, low_close_shift], axis=1).max(axis=1)
    df["atr_pct"] = true_range.rolling(window=14).mean() / df["Close"]
    
    # Dynamic Class Labels
    df["real_future_close"] = df["Close"].shift(-24)
    df["future_target"] = 1  
    df.loc[df["real_future_close"] > df["Close"], "future_target"] = 2  
    df.loc[df["real_future_close"] < df["Close"], "future_target"] = 0  
    
    feature_cols = ["candle_return", "rsi", "macd_histogram", "distance_from_mean"]
    df_clean = df.dropna(subset=feature_cols + ["future_target", "atr_pct"]).copy()
    
    sim_rows = df_clean.loc[(df_clean.index >= sim_start_dt) & (df_clean.index <= sim_end_dt)]
    
    # Robustness Ledger Grid to collect results
    heatmap_yields = {mem: {conf: 0.0 for conf in confidence_options} for mem in memory_options}
    heatmap_precisions = {mem: {conf: 0.0 for conf in confidence_options} for mem in memory_options}
    heatmap_trades = {mem: {conf: 0 for conf in confidence_options} for mem in memory_options}
    
    print(f"[2/2] Running Multi-Dimensional Robustness Sweep ({len(memory_options) * len(confidence_options)} Configurations)...")
    
    for current_mem in memory_options:
        for current_conf in confidence_options:
            
            # Reset simulation environment metrics for this unique pairing
            portfolio_balance = starting_capital
            active_position_expiry = None
            active_position_data = {}
            total_trades, total_wins = 0, 0
            
            for step_idx in range(len(sim_rows)):
                current_time = sim_rows.index[step_idx]
                
                # Trade Expiration & Settlement
                if active_position_expiry is not None and current_time >= active_position_expiry:
                    p_data = active_position_data
                    exit_price_raw = df_clean.loc[current_time, "Close"]
                    current_atr = df_clean.loc[current_time, "atr_pct"]
                    exit_slippage = current_atr * 0.10
                    
                    if p_data["Type"] == "LONG":
                        exit_px = exit_price_raw * (1.0 - exit_slippage)
                        net_move = (exit_px - p_data["Entry_Px"]) / p_data["Entry_Px"]
                    else:
                        exit_px = exit_price_raw * (1.0 + exit_slippage)
                        net_move = (p_data["Entry_Px"] - exit_px) / p_data["Entry_Px"]
                        
                    pnl = (allocation_per_trade * net_move) - (allocation_per_trade * 0.0005 * 2)
                    portfolio_balance += pnl
                    total_trades += 1
                    if pnl > 0: total_wins += 1
                    
                    active_position_expiry = None
                    active_position_data = {}
                
                # Calibration and Feature Slicing
                historical_pool = df_clean.loc[df_clean.index < current_time]
                train_block = historical_pool.iloc[-int((current_mem + calibration_days) * 24): -int(calibration_days * 24)]
                calib_block = historical_pool.iloc[-int(calibration_days * 24):]
                
                if len(train_block) < 200 or len(calib_block) < 100: continue
                
                # Base Classifier Fit
                base_model = lgb.LGBMClassifier(
                    objective="multiclass", num_class=3, class_weight="balanced",
                    n_estimators=45, learning_rate=0.03, num_leaves=15, max_depth=4, random_state=42, verbosity=-1
                )
                base_model.fit(train_block[feature_cols], train_block["future_target"])
                
                # Platt Scaling Mapping
                calib_raw_probs = base_model.predict_proba(calib_block[feature_cols])
                platt_long = LogisticRegression(C=1e5, solver='lbfgs').fit(calib_raw_probs[:, [2]], (calib_block["future_target"] == 2).astype(int))
                platt_short = LogisticRegression(C=1e5, solver='lbfgs').fit(calib_raw_probs[:, [0]], (calib_block["future_target"] == 0).astype(int))
                
                current_vector = sim_rows[feature_cols].iloc[[step_idx]]
                raw_probs = base_model.predict_proba(current_vector)[0]
                
                calibrated_long = platt_long.predict_proba([[raw_probs[2]]])[0][1]
                calibrated_short = platt_short.predict_proba([[raw_probs[0]]])[0][1]
                
                # Validation Logic Strategy Trigger Checks
                triggered = False; direction = ""; final_conf = 0.0
                if calibrated_long >= current_conf:
                    triggered = True; direction = "LONG"; final_conf = calibrated_long
                elif calibrated_short >= current_conf:
                    triggered = True; direction = "SHORT"; final_conf = calibrated_short
                    
                if triggered and active_position_expiry is None:
                    current_atr = sim_rows["atr_pct"].values[step_idx]
                    entry_slippage = current_atr * 0.10
                    raw_entry = sim_rows["Close"].values[step_idx]
                    
                    real_entry = raw_entry * (1.0 + entry_slippage) if direction == "LONG" else raw_entry * (1.0 - entry_slippage)
                    active_position_expiry = current_time + timedelta(hours=24)
                    active_position_data = {"Type": direction, "Entry_Px": real_entry}
            
            # Log metrics for this configuration step
            net_yield_pct = ((portfolio_balance - starting_capital) / starting_capital) * 100
            precision_pct = (total_wins / total_trades) * 100 if total_trades > 0 else 0.0
            
            heatmap_yields[current_mem][current_conf] = net_yield_pct
            heatmap_precisions[current_mem][current_conf] = precision_pct
            heatmap_trades[current_mem][current_conf] = total_trades

    # --- RENDER STRATEGY ROBUSTNESS REPORT CARDS ---
    print("\n" + "="*85)
    print(f"                   DYNAMIC PARAMETER ROBUSTNESS GRID ({target_coin})                   ")
    print("="*85)
    
    print("\n MATRIX 1: ABSOLUTE NET STRATEGY YIELD (%)")
    print("-" * 65)
    print(f"{'Mem Window':<12} | {'Conf 68%':<10} | {'Conf 70%':<10} | {'Conf 72%':<10} | {'Conf 75%':<10}")
    print("-" * 65)
    for mem in memory_options:
        print(f"{str(mem)+' Days':<12} | {heatmap_yields[mem][0.68]:+8.2f}% | {heatmap_yields[mem][0.70]:+8.2f}% | {heatmap_yields[mem][0.72]:+8.2f}% | {heatmap_yields[mem][0.75]:+8.2f}%")
        
    print("\n MATRIX 2: OUT-OF-SAMPLE STRATEGY PRECISION (%)")
    print("-" * 65)
    print(f"{'Mem Window':<12} | {'Conf 68%':<10} | {'Conf 70%':<10} | {'Conf 72%':<10} | {'Conf 75%':<10}")
    print("-" * 65)
    for mem in memory_options:
        print(f"{str(mem)+' Days':<12} | {heatmap_precisions[mem][0.68]:6.2f}%    | {heatmap_precisions[mem][0.70]:6.2f}%    | {heatmap_precisions[mem][0.72]:6.2f}%    | {heatmap_precisions[mem][0.75]:6.2f}%")

    print("\n MATRIX 3: EXECUTION VOLUME (TOTAL TRADES)")
    print("-" * 65)
    print(f"{'Mem Window':<12} | {'Conf 68%':<10} | {'Conf 70%':<10} | {'Conf 72%':<10} | {'Conf 75%':<10}")
    print("-" * 65)
    for mem in memory_options:
        print(f"{str(mem)+' Days':<12} | {heatmap_trades[mem][0.68]:<10} | {heatmap_trades[mem][0.70]:<10} | {heatmap_trades[mem][0.72]:<10} | {heatmap_trades[mem][0.75]:<10}")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_robustness_matrix_engine()
