# Adaptive Horizon Sniper Engine (AHSE)

Adaptive Horizon Sniper Engine (AHSE) is a quantitative machine learning research project investigating short-horizon cryptocurrency prediction under realistic validation conditions.

The project began as a simple experiment exploring whether machine learning could identify recurring directional patterns in cryptocurrency markets. Over five successive versions, the framework evolved from a basic predictive model into a regime-aware research system incorporating walk-forward validation, probability calibration, execution-cost modeling, cross-asset testing, and market-regime analysis.

Rather than focusing exclusively on maximizing historical returns, the project emphasizes methodological rigor and seeks to address common weaknesses found in financial machine learning research, including overfitting, look-ahead bias, parameter sensitivity, and unrealistic backtesting assumptions.

The complete research paper documenting the development of the framework is included in this repository as **AHSE (cl).pdf**.

---

## Repository Structure

```text
ADAPTIVE-HORIZON-SNIPER-ENGINE/

AHSE (cl).pdf
README.md
requirements.txt

versions/
    v1.py
    v2.py
    v3.py
    v4.py
    v5.py
```

---

## Framework Evolution

### Version 1 — Baseline Framework

Established the initial machine learning pipeline using Random Forest models and basic technical features. The objective was to determine whether cryptocurrency market data contained exploitable short-term structure.

### Version 2 — Walk-Forward Learning

Introduced rolling walk-forward validation and LightGBM-based prediction. This version focused on addressing market non-stationarity and reducing dependence on static train-test splits.

### Version 3 — Calibration & Realism

Added probability calibration, transaction-cost modeling, and robustness testing. This stage shifted the project from maximizing performance toward evaluating the reliability of reported results.

### Version 4 — Generalization

Expanded testing across multiple cryptocurrency assets and introduced regime-based analysis. The objective was to determine whether observed predictive behavior generalized beyond a single market.

### Version 5 — Regime-Aware Expert Ensemble

Implemented a regime-aware architecture designed to assign specialized predictive models to different market environments. This version represents the final validated cryptocurrency framework and serves as the primary focus of the accompanying paper.

---

## Methodology

The framework follows a rolling walk-forward learning process in which models are trained only on information that would have been historically available at the time of prediction.

Across successive versions, the project incorporated:

* Walk-forward validation
* Probability calibration
* Execution-cost modeling
* Cross-asset testing
* Regime analysis
* Multi-stage out-of-sample evaluation

The final Version 5 framework was evaluated across approximately 630 days of historical data using a fixed parameter configuration deployed across sixteen independent cryptocurrency markets.

---

## Validation Universe

Version 5 was evaluated across:

BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT, LTC, NEAR, OP, INJ, ATOM, and THETA.

The objective was to assess whether predictive performance could persist across assets with different liquidity profiles, volatility characteristics, and market structures.

---

## Example Results

| Asset | Validation Return | Precision |
| ----- | ----------------- | --------- |
| BTC   | +72.65%           | 63.3%     |
| ETH   | +102.18%          | 63.9%     |
| AVAX  | +106.74%          | 64.6%     |
| ADA   | +82.37%           | 60.0%     |
| DOT   | +77.02%           | 55.6%     |

Across the full validation universe, the framework demonstrated positive returns on all tested assets while maintaining precision levels generally ranging between 55% and 65%.

A detailed discussion of methodology, experiments, and results is provided in the accompanying research paper.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Disclaimer

This repository is intended for educational and research purposes only. Historical results do not guarantee future performance, and nothing contained within this project should be interpreted as financial or investment advice.

