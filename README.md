# CausCrypto

## Causal Inference Analysis for Crypto Coins

**CausCrypto**

## Table of Contents

* Description
* Data
* Running the Analysis
* How the code works
* Installation
* File Structure
* License

## Description

* What is CausCrypto? It is an end-to-end Python framework that implements & normalizes daily closing prices for nine major crypto-assets (BTC, ETH, etc.).
* It performs time series tests **Augmented Dickey-Fuller** to assess stationarity. Visualizes trends, correlation matrices and a hand-crafted **Causal Graph**.
* It builds a **Bayesian Network** to compute conditional probability that an alt-coin moves with Bitcoin.
* The constructed **Propensity Score** via **Logistic Regression** matches the treatment (momentun trade) to control the observations. Finally, the **Average Treatment Effect** is estimated for the return and the trend value.
* Why? Many traders believe that strong correlation between Bitcoin / Ethereum and alt-coins is a weak hint of causality. This script lets you move from a correlation to a causal inference pipeline - giving you a quantified probability of a joint up / down move and letting you test a simple momentum strategy.

## Data

The script expects daily CSV downloads from **https://www.cryptodatadownload.com** in the following format (columns show in the original files):

```
'Unix', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume <symbol>', 'tradecount'
```
Place the files in the root folder (or point to another directory by editing the òpen()` path).

> Files included in the repo:
> `Binance_ADAUSDT_d.csv`, `Binance_BCHUSDT_d.csv`, ..., `Binance_SOLUSDT_d.csv`

---

## Running the Analysis
 
```bash
python CausCrypto.py
```

The script prints a quick report to the console:

```
Latest Unix time entry = 1597104000000
Correlation Matrix of Cryptocoins
Estimated ATE for Return: -4.3162 (+/- 20.9837)
Estimated ATE for Trend: 0.5410 (+/- 4.5521)
Sample Propensity Scores:
   Treatment  Propensity Score
0          1            0.689294
1          1            0.545454
...
```

---

## How the code works (high-level walk-through)

* *Data Preparation* | Reads CSVs, convert dates to 'datetime', aligns them to common timestanp (`latest_unix`).
* *Normalization* | Scales each coin's `Close`by its maximum value (result `_s`series).
* *ADF* | For each coin, prints the ADF test statistics to gauge stationarity.
* *Visualizations* | Sub-stacked daily closing trend of all nine coins, Heat map of Pearson correlations, and a hand-craft causal graph (BTC → SOL, BTC → TRX, ...)

![plot](Trends.png)

![plot](CorrMatrix.png)

![plot](ACG.png)

* *Bayesian Network* | Uses `pomegranate` to specify acyclic graph of seven coins with conditional probability tables derived from the Pearson correlations.

![plot](ACG_final.png)

* *Momentum strategy* | Generate a binary "treatment" if Solana's trend matches Bitcoin's trend over a 30-day SMA.
* *Propensity Score* | *Logistic Regression* on standardized features to compute `Propensity Score`.
* *Nearest-Neighbor* matching | Matches each treated point with the nearest control within a caliper of 0.05.
* *ATE Calculation* | Substract matched pair Return & Trend to quantify effect sizes. 
Histogram of the return and trend balance of the matched dataset

![plot](Histogram.png)

> **Why a Bayesian network for altcoins?**  
> The network encodes assumed causal drivers (BTC→TRX, BTC→BNB, etc.). After training, it outputs the probability that a “down” move in BTC leads to a “down” in each alt‑coin. 
In the current setting the result is roughly 80 % for BNB, TRX, SOL and XRP, which is the main story used to justify momentum‑based forecasts.

---

## Installation

### Prerequisites

* python 3.12.11
* numpy 1.18
* pandas 1.2
* matplotlib 3.5
* seaborn 0.11
* networkx 2.6
* statmodels 0.12
* pomegranate 0.14
* scikit-learn 0.24
* torch | optional for BN conversion
* pip (Python package manager)
 
### Clone the repo:
```
git clone https://github.com/mschindler779/CausCrypto.git
cd CausCrypto
```

### Create a virtual environment (optional but recommended)
```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```

### Install dependencies
```
python -m pip install -r requirements.txt
```

### Usage
```
python CausCrypto.py
```    

## File Structure

├── CausCrypto.py **Main Python application**<br/>
├── Trends.png **Graphic for Moving Averages**<br/>
├── CorrMatrix.png **Graphic for the Heatmap**<br/>
├── Heatmap.png **Graphic for Heatmap**<br/>
├── ACG.png **Graphic for the Causal Graph**<br/>
├── ACG_final.png **Graphic for the final DAG**<br/>
├── Histogram.png **Graphic for Balance**<br/>
├── Binance_...USDT_d.csv **Data CSV Files**<br/>
├── requirements.txt **Python dependencies**<br/>
├── README.md **This file**<br/>
└── LICENSE **Unlicense**

## License

This project is licensed under the Unlicense - see the LICENSE file for details

© 2026 Markus Schindler
