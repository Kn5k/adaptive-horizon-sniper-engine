import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from datetime import datetime, timedelta, timezone

def run_70pct_adaptive_sniper():
    print("\n" + "="*85)
    print("     QUANT ALPHA ENGINE: MODEL 2 ADAPTIVE STEP RETRAINING ENGINE (70%+ TARGET)   ")
    print("="*85)
    
    target_coin = input(" Enter Single Ticker to Hunt (e.g., BTC-USD, ETH-USD, SOL-USD): ").strip().upper()
    starting_capital = float(input("Total Starting Portfolio Capital ($USD): "))
    allocation_per_trade = float(input(" Allocation Cash Size Per Trade ($USD): "))
    backtest_days = int(input(" Enter Live Simulation Window Depth (e.g., 30, 45 days): "))
    
    training_days = 45   # Length of rolling memory the engine keeps active
    current_time_anchor = datetime.now(timezone.utc)
    
    sim_end_dt = current_time_anchor - timedelta(hours=6)
    sim_start_dt = sim_end_dt - timedelta(days=backtest_days)
    train_start_dt = sim_start_dt - timedelta(days=training_days)
    
    start_date_pull = train_start_dt.strftime('%Y-%m-%d')
    end_date_pull = (sim_end_dt + timedelta(days=2)).strftime('%Y-%m-%d')
    
    chosen_interval = "1h"
    window_scale = 24  
    ESTIMATED_FEE_RATE = 0.0005  
    
    print(f"\n[1/3] Ingesting dynamic rolling matrix from {start_date_pull}...")
    df_raw = yf.download(tickers=target_coin, start=start_date_pull, end=end_date_pull, interval=chosen_interval, multi_level_index=False, progress=False)
    
    if df_raw.empty or len(df_raw) < 100:
        print(" Operational Failure: Core pipeline data stream empty.")
        return
        
    df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    
    # --- VOLATILITY-INFORMED REAL FEATURES ---
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
    df["volume_z_score"] = (df["Volume"] - df["Volume"].rolling(window=window_scale).mean()) / (df["Volume"].rolling(window=window_scale).std() + 1e-8)
    
    df["real_future_close"] = df["Close"].shift(-window_scale)
    df["future_target"] = 1  
    df.loc[df["real_future_close"] > df["Close"], "future_target"] = 2  
    df.loc[df["real_future_close"] < df["Close"], "future_target"] = 0  
    
    df_clean = df.dropna(subset=["candle_return", "rsi", "macd_histogram", "distance_from_mean", "volume_z_score", "future_target"]).copy()
    
    # Trackers
    portfolio_account_balance = starting_capital
    portfolio_peak_balance = starting_capital
    portfolio_max_drawdown = 0.0
    
    long_trades, long_wins, long_losses, long_loss_dollars = 0, 0, 0, 0.0
    short_trades, short_wins, short_losses, short_loss_dollars = 0, 0, 0, 0.0
    total_fees_paid = 0.0
    
    daily_trade_tracker = {}
    active_position_expiry = None  
    active_position_data = {}
    
    feature_cols = ["candle_return", "rsi", "macd_histogram", "distance_from_mean", "volume_z_score"]
    
    # Isolate initial simulation pool rows
    sim_rows = df_clean.loc[(df_clean.index >= sim_start_dt) & (df_clean.index <= sim_end_dt)]
    
    print(f"[2/3] Initiating live simulation. Entering Hour-by-Hour Retraining Core Matrix...")
    print("\n" + "="*125)
    print(f"              ADAPTIVE STEP-RETRAIN SNIPER TRANSACTION LEDGER ({target_coin})               ")
    print("="*125)
    print(f"{'Asset':<8} | {'Date/Time':<16} | {'Type':<5} | {'Model Conf':<10} | {'Entry Px':<10} | {'Exit Px':<10} | {'Real Move%':<10} | {'Net PnL ($)':<12} | {'Balance ($)':<15}")
    print("-"*125)

    # --- THE CHRONOLOGICAL LIVE STEP RETRAIN LOOP ---
    for step_idx in range(len(sim_rows)):
        current_time = sim_rows.index[step_idx]
        timestamp_day_str = str(current_time)[:10]
        
        # Position Termination Settlement Check
        if active_position_expiry is not None and current_time >= active_position_expiry:
            p_data = active_position_data
            # Extract real settlement price at expiration tick
            exit_price = df_clean.loc[current_time, "Close"]
            raw_pct_change = (exit_price - p_data["Entry_Px"]) / p_data["Entry_Px"]
            
            if p_data["Type"] == "LONG":
                gross_pnl_usd = allocation_per_trade * raw_pct_change
                move_display = raw_pct_change * 100
                long_trades += 1
            else:
                gross_pnl_usd = allocation_per_trade * (-raw_pct_change)
                move_display = -raw_pct_change * 100
                short_trades += 1
                
            round_trip_fee = (allocation_per_trade * ESTIMATED_FEE_RATE) * 2
            net_pnl_usd = gross_pnl_usd - round_trip_fee
            
            portfolio_account_balance += net_pnl_usd
            total_fees_paid += round_trip_fee
            
            if net_pnl_usd > 0:
                if p_data["Type"] == "LONG": long_wins += 1
                else: short_wins += 1
            else:
                if p_data["Type"] == "LONG":
                    long_losses += 1
                    long_loss_dollars += abs(net_pnl_usd)
                else:
                    short_losses += 1
                    short_loss_dollars += abs(net_pnl_usd)
                    
            print(f"{target_coin:<8} | {str(p_data['Timestamp'])[:16]:<16} | {p_data['Type']:<5} | {p_data['Conf']*100:6.2f}%    | ${p_data['Entry_Px']:<9.2f} | ${exit_price:<9.2f} | {move_display:+=9.2f}% | ${net_pnl_usd:+=10.2f} | ${portfolio_account_balance:<14,.2f}")
            
            if portfolio_account_balance > portfolio_peak_balance:
                portfolio_peak_balance = portfolio_account_balance
            dd = (portfolio_peak_balance - portfolio_account_balance) / portfolio_peak_balance * 100
            if dd > portfolio_max_drawdown:
                portfolio_max_drawdown = dd
                
            active_position_expiry = None
            active_position_data = {}

        #  THE BRAIN RESET: Extract rolling history ending EXACTLY at this active hour
        rolling_train = df_clean.loc[df_clean.index < current_time].iloc[-int(training_days * 24):]
        
        # Enforce probability calibration constraints on the tree structures
        step_model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, class_weight="balanced",
            n_estimators=45, learning_rate=0.03, num_leaves=15, max_depth=4, 
            min_child_samples=15, random_state=42, verbosity=-1
        )
        step_model.fit(rolling_train[feature_cols], rolling_train["future_target"])
        
        # Pull single active vector row
        current_vector = sim_rows[feature_cols].iloc[[step_idx]]
        probs = step_model.predict_proba(current_vector)[0]
        prob_down, prob_up = probs[0], probs[2]
        
        # Enforce strict 72% Confidence Handoff Filter
        trade_triggered = False; trade_direction = ""; setup_confidence = 0.0
        if prob_up >= 0.72:
            trade_triggered = True; trade_direction = "LONG"; setup_confidence = prob_up
        elif prob_down >= 0.72:
            trade_triggered = True; trade_direction = "SHORT"; setup_confidence = prob_down
            
        if trade_triggered:
            if timestamp_day_str not in daily_trade_tracker: daily_trade_tracker[timestamp_day_str] = 0
            if active_position_expiry is not None or daily_trade_tracker[timestamp_day_str] >= 1: continue
                
            daily_trade_tracker[timestamp_day_str] += 1
            active_position_expiry = current_time + timedelta(hours=window_scale)
            active_position_data = {
                "Timestamp": current_time, 
                "Type": trade_direction, 
                "Entry_Px": sim_rows["Close"].values[step_idx], 
                "Conf": setup_confidence
            }

    # Final Auditing Outputs
    total_trades = long_trades + short_trades
    total_wins = long_wins + short_wins
    print("="*125)
    print(f"                                DYNAMIC DIRECTIONAL DIAGNOSTICS REPORT                                 ")
    print("="*125)
    print(f"  LONG POSITION METRICS  | Total Executed: {long_trades:<2} | Wins: {long_wins:<2} | Losses: {long_losses:<2} | Capital Bleed: ${long_loss_dollars:,.2f}")
    print(f"  SHORT POSITION METRICS | Total Executed: {short_trades:<2} | Wins: {short_wins:<2} | Losses: {short_losses:<2} | Capital Bleed: ${short_loss_dollars:,.2f}")
    print("-" * 125)
    print(f" Real Out-of-Sample Strategy Precision: {(total_wins / total_trades)*100:.2f}%" if total_trades > 0 else " Real Out-of-Sample Strategy Precision: 0.00%")
    print(f" Final Portfolio Balance Realized     : ${portfolio_account_balance:,.2f} (Net Growth: {((portfolio_account_balance - starting_capital)/starting_capital)*100:+.2f}%)")
    print(f" Peak Macro Portfolio Drawdown        : {portfolio_max_drawdown:.2f}%")
    print("="*125 + "\n")

if __name__ == "__main__":
    run_70pct_adaptive_sniper()
