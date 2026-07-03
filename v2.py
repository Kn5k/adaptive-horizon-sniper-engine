import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from datetime import datetime, timedelta, timezone

def run_calibrated_sniper_engine():
    print("\n" + "="*95)
    print("   QUANT ALPHA ENGINE: VERSION 3 (MODULE A - PROBABILITY CALIBRATION & SLIPPAGE)  ")
    print("="*95)
    
    target_coin = input(" Enter Single Ticker to Hunt (e.g., BTC-USD, ETH-USD, SOL-USD): ").strip().upper()
    starting_capital = float(input(" Total Starting Portfolio Capital ($USD): "))
    allocation_per_trade = float(input(" Allocation Cash Size Per Trade ($USD): "))
    backtest_days = int(input(" Enter Live Simulation Window Depth (e.g., 30, 45 days): "))
    
    training_days = 45   
    calibration_days = 15  # Out-of-sample block dedicated exclusively to training the Platt Scaler
    
    current_time_anchor = datetime.now(timezone.utc)
    sim_end_dt = current_time_anchor - timedelta(hours=6)
    sim_start_dt = sim_end_dt - timedelta(days=backtest_days)
    calib_start_dt = sim_start_dt - timedelta(days=calibration_days)
    train_start_dt = calib_start_dt - timedelta(days=training_days)
    
    start_date_pull = train_start_dt.strftime('%Y-%m-%d')
    end_date_pull = (sim_end_dt + timedelta(days=2)).strftime('%Y-%m-%d')
    
    chosen_interval = "1h"
    window_scale = 24  
    BASE_FEE_RATE = 0.0005  # 0.05% Exchange Fee Standard
    
    print(f"\n[1/3] Streaming market matrices starting from {start_date_pull}...")
    df_raw = yf.download(tickers=target_coin, start=start_date_pull, end=end_date_pull, interval=chosen_interval, multi_level_index=False, progress=False)
    
    if df_raw.empty or len(df_raw) < 100:
        print(" Operational Failure: Empty historical stream packet.")
        return
        
    df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    
    # --- FEATURES ---
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
    
    # Dynamic Volatility Metric (ATR Proxy) for Slippage Penalties
    high_low_range = df["High"] - df["Low"]
    high_close_shift = abs(df["High"] - df["Close"].shift(1))
    low_close_shift = abs(df["Low"] - df["Close"].shift(1))
    true_range = pd.concat([high_low_range, high_close_shift, low_close_shift], axis=1).max(axis=1)
    df["atr_pct"] = true_range.rolling(window=14).mean() / df["Close"]
    
    # Multi-Class Classifications Target (24H Shift)
    df["real_future_close"] = df["Close"].shift(-window_scale)
    df["future_target"] = 1  
    df.loc[df["real_future_close"] > df["Close"], "future_target"] = 2  
    df.loc[df["real_future_close"] < df["Close"], "future_target"] = 0  
    
    feature_cols = ["candle_return", "rsi", "macd_histogram", "distance_from_mean"]
    df_clean = df.dropna(subset=feature_cols + ["future_target", "atr_pct"]).copy()
    
    # Account & Metric Trackers
    portfolio_account_balance = starting_capital
    portfolio_peak_balance = starting_capital
    portfolio_max_drawdown = 0.0
    
    long_trades, long_wins, long_losses = 0, 0, 0
    short_trades, short_wins, short_losses = 0, 0, 0
    
    active_position_expiry = None  
    active_position_data = {}
    
    # Calibration Assessment Arrays
    all_calibrated_forecasts = []
    all_actual_outcomes = []
    
    sim_rows = df_clean.loc[(df_clean.index >= sim_start_dt) & (df_clean.index <= sim_end_dt)]
    
    print(f"[2/3] Executing Calibrated Simulation Loop over {len(sim_rows)} steps...")
    print("\n" + "="*135)
    print(f"              VERSION 3 (MODULE A): CALIBRATED TRANSACTION LEDGER ({target_coin})               ")
    print("="*135)
    print(f"{'Asset':<8} | {'Date/Time':<16} | {'Type':<5} | {'Raw Tree':<10} | {'Calib Conf':<11} | {'Slippage':<9} | {'Net PnL ($)':<12} | {'Balance ($)':<15}")
    print("-"*135)

    for step_idx in range(len(sim_rows)):
        current_time = sim_rows.index[step_idx]
        
        # Position Settle Engine
        if active_position_expiry is not None and current_time >= active_position_expiry:
            p_data = active_position_data
            exit_price_raw = df_clean.loc[current_time, "Close"]
            current_atr = df_clean.loc[current_time, "atr_pct"]
            
            # Slippage Applied to Exit Market Orders
            exit_slippage_pct = (current_atr * 0.10)  # Slippage is penalized at 10% of standard active ATR
            
            if p_data["Type"] == "LONG":
                exit_price = exit_price_raw * (1.0 - exit_slippage_pct)
                gross_pnl_usd = allocation_per_trade * ((exit_price - p_data["Entry_Px"]) / p_data["Entry_Px"])
                long_trades += 1
            else:
                exit_price = exit_price_raw * (1.0 + exit_slippage_pct)
                gross_pnl_usd = allocation_per_trade * ((p_data["Entry_Px"] - exit_price) / p_data["Entry_Px"])
                short_trades += 1
                
            round_trip_fee = (allocation_per_trade * BASE_FEE_RATE) * 2
            net_pnl_usd = gross_pnl_usd - round_trip_fee
            portfolio_account_balance += net_pnl_usd
            
            if net_pnl_usd > 0:
                if p_data["Type"] == "LONG": long_wins += 1
                else: short_wins += 1
            else:
                if p_data["Type"] == "LONG": long_losses += 1
                else: short_losses += 1
                
            print(f"{target_coin:<8} | {str(p_data['Timestamp'])[:16]:<16} | {p_data['Type']:<5} | {p_data['Raw_Conf']*100:6.1f}%    | {p_data['Calib_Conf']*100:7.1f}%    | {p_data['Slippage_Paid']*100:6.3f}%   | ${net_pnl_usd:+=10.2f} | ${portfolio_account_balance:<14,.2f}")
            
            if portfolio_account_balance > portfolio_peak_balance: portfolio_peak_balance = portfolio_account_balance
            dd = (portfolio_peak_balance - portfolio_account_balance) / portfolio_peak_balance * 100
            if dd > portfolio_max_drawdown: portfolio_max_drawdown = dd
                
            active_position_expiry = None
            active_position_data = {}

        # --- THE PLATT CALIBRATION PIPELINE ---
        # Isolate history relative to the exact hour row step
        historical_pool = df_clean.loc[df_clean.index < current_time]
        
        train_block = historical_pool.iloc[-int((training_days + calibration_days) * 24): -int(calibration_days * 24)]
        calib_block = historical_pool.iloc[-int(calibration_days * 24):]
        
        if len(train_block) < 200 or len(calib_block) < 100: continue
        
        # Step 1: Fit Baseline LightGBM on Train Block
        base_model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, class_weight="balanced",
            n_estimators=45, learning_rate=0.03, num_leaves=15, max_depth=4, random_state=42, verbosity=-1
        )
        base_model.fit(train_block[feature_cols], train_block["future_target"])
        
        # Step 2: Extract out-of-sample probabilities over Calibration Block
        calib_raw_probs = base_model.predict_proba(calib_block[feature_cols])
        
        # Step 3: Train Independent Platt Scalers (Logistic Regressions) for Long/Short Vectors
        platt_long = LogisticRegression(C=1e5, solver='lbfgs')
        platt_short = LogisticRegression(C=1e5, solver='lbfgs')
        
        # Binary conversions for targeted classes
        binary_target_long = (calib_block["future_target"] == 2).astype(int)
        binary_target_short = (calib_block["future_target"] == 0).astype(int)
        
        platt_long.fit(calib_raw_probs[:, [2]], binary_target_long)
        platt_short.fit(calib_raw_probs[:, [0]], binary_target_short)
        
        # Step 4: Inference on Current Active Vector
        current_vector = sim_rows[feature_cols].iloc[[step_idx]]
        raw_probs = base_model.predict_proba(current_vector)[0]
        
        # Pass raw predictions through the trained Platt Scaling sigmoids
        calibrated_prob_up = platt_long.predict_proba([[raw_probs[2]]])[0][1]
        calibrated_prob_down = platt_short.predict_proba([[raw_probs[0]]])[0][1]
        
        # Track Calibration Stats for Brier Evaluation Matrix
        all_calibrated_forecasts.append(calibrated_prob_up if calibrated_prob_up > calibrated_prob_down else calibrated_prob_down)
        real_outcome_class = sim_rows["future_target"].values[step_idx]
        if calibrated_prob_up > calibrated_prob_down:
            all_actual_outcomes.append(1 if real_outcome_class == 2 else 0)
        else:
            all_actual_outcomes.append(1 if real_outcome_class == 0 else 0)
            
        # Enforce execution gates strictly over Calibrated Probabilities (72% Threshold Hurdle)
        trade_triggered = False; trade_direction = ""; active_calib_conf = 0.0; active_raw_conf = 0.0
        
        if calibrated_prob_up >= 0.72:
            trade_triggered = True; trade_direction = "LONG"
            active_calib_conf = calibrated_prob_up; active_raw_conf = raw_probs[2]
        elif calibrated_prob_down >= 0.72:
            trade_triggered = True; trade_direction = "SHORT"
            active_calib_conf = calibrated_prob_down; active_raw_conf = raw_probs[0]
            
        if trade_triggered and active_position_expiry is None:
            current_atr = sim_rows["atr_pct"].values[step_idx]
            entry_slippage_pct = (current_atr * 0.10)
            raw_entry_px = sim_rows["Close"].values[step_idx]
            
            # Penalize entry execution pricing
            real_entry_px = raw_entry_px * (1.0 + entry_slippage_pct) if trade_direction == "LONG" else raw_entry_px * (1.0 - entry_slippage_pct)
            
            active_position_expiry = current_time + timedelta(hours=window_scale)
            active_position_data = {
                "Timestamp": current_time, "Type": trade_direction, "Entry_Px": real_entry_px,
                "Raw_Conf": active_raw_conf, "Calib_Conf": active_calib_conf, "Slippage_Paid": entry_slippage_pct
            }

    # --- ADVANCED PROBABILITY CALIBRATION ANALYSIS ---
    calculated_brier_score = brier_score_loss(all_actual_outcomes, all_calibrated_forecasts) if all_actual_outcomes else 1.0
    total_trades = long_trades + short_trades
    total_wins = long_wins + short_wins
    
    print("="*135)
    print(f"                             MODULE A PROBABILITY CALIBRATION & FRICTION AUDIT                               ")
    print("="*135)
    print(f"   SYSTEM BRIER CALIBRATION SCORE : {calculated_brier_score:.5f}  (Lower Bound Approaching 0.000 = Elite Calibration)")
    print(f"   CALIBRATED SIGNAL EFFICIENCY : Total Executed: {total_trades} | Real Out-of-Sample Precision: {(total_wins/total_trades)*100:.2f}%" if total_trades > 0 else " 📊 CALIBRATED SIGNAL EFFICIENCY: 0 Trades Triggers.")
    print(f"   LONG METRICS                 | Executed: {long_trades:<2} | Wins: {long_wins:<2} | Losses: {long_losses:<2}")
    print(f"   SHORT METRICS                | Executed: {short_trades:<2} | Wins: {short_wins:<2} | Losses: {short_losses:<2}")
    print(f"   REALIZED FINANCIAL PROFILE   | Net Yield: {((portfolio_account_balance - starting_capital)/starting_capital)*100:+.2f}% | Max Drawdown: {portfolio_max_drawdown:.2f}%")
    print("="*135 + "\n")

if __name__ == "__main__":
    run_calibrated_sniper_engine()

    ##having the formulas
