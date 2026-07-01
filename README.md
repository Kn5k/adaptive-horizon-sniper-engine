# Adaptive Horizon Sniper Engine (AHSE)

Adaptive Horizon Sniper Engine (AHSE) is a quantitative machine learning research project investigating short-term cryptocurrency prediction under increasingly realistic validation conditions.

Rather than presenting a single trading model, the project documents the evolution of five successive frameworks, each designed to address weaknesses identified in earlier versions. The research focuses on walk-forward validation, probability calibration, execution realism, cross-asset robustness, and regime-aware prediction.

The complete research paper is included in this repository as:

**AHSE (cl).pdf**

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

### Version 1

Baseline machine learning framework using Random Forest models and simple feature engineering.

### Version 2

Introduced rolling walk-forward validation and LightGBM-based prediction.

### Version 3

Added probability calibration, transaction-cost modeling, and robustness testing.

### Version 4

Expanded evaluation across multiple cryptocurrency assets and market environments.

### Version 5

Implemented a regime-aware expert architecture and large-scale cross-sectional validation.

---

## Key Findings

* Walk-forward validation produces more reliable estimates than static train-test splits.
* Probability calibration improves decision quality despite reducing apparent performance.
* Execution costs materially affect strategy viability.
* Predictive performance varies significantly across market regimes.
* Regime-aware architectures appear more stable than monolithic forecasting systems.

---

## Validation Universe

Version 5 was evaluated across sixteen cryptocurrency markets:

BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, DOT, LTC, NEAR, OP, INJ, ATOM, and THETA.

---

## Example Version 5 Results

| Asset | Return   | Precision |
| ----- | -------- | --------- |
| BTC   | +72.65%  | 63.3%     |
| ETH   | +102.18% | 63.9%     |
| AVAX  | +106.74% | 64.6%     |
| ADA   | +82.37%  | 60.0%     |
| DOT   | +77.02%  | 55.6%     |

These results are discussed in detail within the accompanying paper.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Research Paper

The complete methodology, experiments, results, limitations, and discussion are documented in:

**AHSE (cl).pdf**

---

## Disclaimer

This repository is intended for educational and research purposes only. Historical results do not guarantee future performance.

