# Adaptive Horizon Sniper Engine

*A walk-forward machine learning research framework for studying short-horizon directional prediction, probability calibration, and cross-sectional robustness in cryptocurrency markets.*

---

## Overview

The Adaptive Horizon Sniper Engine is an independent quantitative research project exploring whether short-term market behavior can be modeled through localized machine learning, strict walk-forward validation, probability calibration, and cross-sectional testing.

The project began as a simple experiment investigating whether a small set of interpretable market features could identify recurring momentum and liquidation patterns in cryptocurrency markets. Over five consecutive development iterations, the framework evolved into a structured research platform featuring rolling retraining, probability calibration, execution-cost modeling, parameter robustness testing, multi-asset validation, and long-horizon out-of-sample evaluation.

Rather than pursuing increasingly complex models, the project focused on building a framework that prioritizes validation quality, reproducibility, and resistance to common quantitative research failures such as overfitting, data leakage, look-ahead bias, and unrealistic execution assumptions.

The final Version 5 framework was evaluated across sixteen independent digital assets over approximately 630 days while incorporating transaction costs, purged validation windows, and frozen parameters across the entire universe.

---

## Research Motivation

Machine learning is widely applied in financial markets, yet many published trading systems fail when moved outside their original testing environment. Common causes include:

* Overfitting to a single asset
* Parameter optimization bias
* Look-ahead leakage
* Unrealistic execution assumptions
* Lack of cross-sectional validation
* Probability miscalibration

The Adaptive Horizon Sniper Engine was developed as an attempt to address these issues systematically.

The central hypothesis behind the project is not that markets are fully predictable, but that localized momentum acceleration, panic-driven liquidations, and short-term volatility expansions may create recurring structures that can be identified through a disciplined walk-forward learning framework.

---

## Repository Structure

```text
Adaptive-Horizon-Sniper-Engine/

├── README.md
│
├── versions/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   ├── v4/
│   └── v5/
│
├── papers/
│   ├── Evolution_of_the_Calibrated_Sniper_Engine.pdf
│   └── Institutional_Grade_Asset_Generalization.pdf
│
├── results/
   ├── tables/
   ├── charts/
   └── performance_reports/
```

---

## Evolution of the Project

### Version 1

The first version established the core walk-forward architecture.

Key goals:

* Hourly cryptocurrency prediction
* Rolling retraining framework
* Initial feature engineering
* Fixed prediction horizon

This stage served primarily as a proof of concept and demonstrated that directional classification could achieve performance above random chance under controlled conditions.

### Version 2

The second version expanded the feature engineering process and introduced a more structured validation workflow.

Improvements included:

* Refined momentum indicators
* Enhanced training procedures
* Expanded historical testing
* Preliminary overfitting audits

This stage produced promising results but revealed vulnerability to parameter sensitivity and idealized execution assumptions.

### Version 3

Version 3 focused on realism and calibration.

Major additions:

* Platt Scaling probability calibration
* Volatility-adjusted execution friction
* Parameter sensitivity mapping
* Monte Carlo validation

This phase represented a major shift from maximizing returns toward understanding model reliability.

### Version 4

Version 4 expanded the framework beyond single-asset testing.

New capabilities included:

* Cross-sectional asset validation
* Regime profiling
* Feature importance tracking
* Market-environment diagnostics

The objective was to determine whether observed performance represented a localized anomaly or a broader market phenomenon.

### Version 5

Version 5 serves as the final validated cryptocurrency framework.

The architecture introduced:

* 630-day evaluation horizon
* 16-asset validation universe
* 10-day purged validation buffer
* Fixed transaction-cost assumptions
* Frozen parameter deployment
* Strict out-of-sample processing

No asset-specific tuning was performed during Version 5 validation.

---

## Methodology

The framework follows a continuous walk-forward learning process.

For every prediction step, the system trains only on information that would have been historically available at that point in time.

The Version 5 workflow follows:

```text
Training Window (50 Days)
          ↓
Purged Buffer (10 Days)
          ↓
Calibration Window (15 Days)
          ↓
Out-of-Sample Prediction
```

This design attempts to reduce information leakage between training, calibration, and execution phases.

Models are retrained repeatedly as new information becomes available, allowing the framework to adapt to changing market conditions without relying on static historical assumptions.

---

## Feature Engineering

The final Version 5 framework intentionally uses a compact feature set.

### Logarithmic Returns

Measures localized directional velocity:

log(Ct / Ct-1)

### RSI (14)

Tracks relative strength and short-term market boundaries.

### MACD Histogram

Captures changes in momentum acceleration and trend velocity.

### Distance From EMA(36)

Measures deviation from a short-term equilibrium baseline.

These features were selected because they remain interpretable, computationally efficient, and relatively robust across different assets.

---

## Machine Learning Framework

The predictive layer uses LightGBM, a gradient-boosted decision tree model.

The prediction problem is formulated as a three-class classification task:

* Down Move
* Neutral Move
* Up Move

Rather than predicting exact prices, the model attempts to classify the likely directional outcome over the next 24-hour horizon.

This formulation reduces sensitivity to extreme price values and focuses the learning process on directional structure.

---

## Probability Calibration

Raw probabilities generated by gradient-boosted trees are frequently overconfident.

To address this issue, Version 3 introduced Platt Scaling.

A logistic regression calibration layer is trained on a dedicated calibration window and transforms raw model outputs into calibrated probability estimates.

The objective is that a prediction reported as 70% confidence should correspond more closely to an observed historical frequency near 70%.

Version 5 only executes trades when calibrated probabilities exceed a fixed 70% threshold.

---

## Execution Assumptions

Many backtests assume perfect market execution.

To improve realism, Version 5 applies a fixed round-trip transaction penalty of 0.12% to every trade.

This penalty represents:

* Exchange fees
* Slippage
* Market impact
* Execution latency

By applying friction uniformly across all assets and trades, the framework attempts to provide a more conservative estimate of performance.

---

## Validation Framework

The final validation phase was designed around three principles:

### Cross-Sectional Robustness

The same parameters are deployed across all assets without retuning.

### Temporal Robustness

Testing spans approximately 630 days across multiple market environments.

### Leakage Prevention

A dedicated 10-day purge window separates training and calibration periods.

Together these measures aim to reduce common sources of false performance.

---

## Validation Universe

The Version 5 framework was evaluated across sixteen cryptocurrencies:

BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD, DOGE-USD, AVAX-USD, LINK-USD, DOT-USD, LTC-USD, NEAR-USD, OP-USD, INJ-USD, ATOM-USD, and THETA-USD.

The universe includes:

* Large-cap assets
* Infrastructure protocols
* High-beta assets
* Ecosystem tokens

This diversity provides a broader assessment of model behavior than single-asset testing.

---

## Summary Results

Version 5 produced positive historical returns across all sixteen tested assets during the evaluation period.

Headline metrics:

* Assets Tested: 16
* Profitable Asset Runs: 16 / 16
* Validation Horizon: ~630 Days
* Precision Range: 48.6% – 64.6%
* Trade Count Range: 55 – 126 Trades
* Average Alpha vs Buy-and-Hold: +108.33%

Detailed performance tables, reports, and charts are available in the `results/` directory.

---

## Research Papers

The repository includes two accompanying papers documenting the development process and final validation framework.

### Evolution of the Calibrated Sniper Engine

Documents the progression from Version 1 through Version 4, including:

* Calibration development
* Friction modeling
* Parameter robustness studies
* Multi-asset validation

### Institutional Grade Asset Generalization

Documents the complete Version 5 framework, including:

* Purged walk-forward validation
* Cross-sectional testing
* Multi-year evaluation
* Performance analysis
* Methodological discussion

---

## Limitations

Although the reported results are encouraging, several limitations remain.

The framework has only been evaluated on cryptocurrency markets and has not yet been validated on:

* Equities
* Futures
* Commodities
* Options
* Fixed income markets

Historical performance does not guarantee future performance.

Market structure can change, and patterns observed during one period may weaken or disappear during another.

The framework should therefore be viewed as a research platform rather than a finished trading product.

---

## Future Research Directions

Potential future directions include:

* Commodity market validation
* Futures market validation
* Options market research
* Multi-brain regime architectures
* Dynamic position sizing
* Portfolio-level optimization
* Advanced calibration methods
* Explainable machine learning diagnostics

The cryptocurrency validation timeline is currently considered complete, and future development will focus on testing whether the framework generalizes to entirely different market structures.

---

## Reproducibility

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Navigate to the desired version directory and execute the corresponding script.

Version 5 represents the final validated cryptocurrency framework.

---

## Author

**Kanishk Sharma**

Independent Quantitative Research Project

2026

---

## Disclaimer

This repository is intended exclusively for educational and research purposes.

Nothing contained within this project should be interpreted as investment advice, financial advice, trading advice, or a recommendation to buy or sell any asset.

All results presented are historical simulations and validation studies. Real-world trading involves substantial risk and may produce outcomes significantly different from historical backtests.
