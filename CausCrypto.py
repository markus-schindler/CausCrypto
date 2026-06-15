#"/usr/bin/python
# -*- coding: utf-8 -*-

"""CausCrypto.py: End-to-end causal inference pipeline for cryptocurrency
price data.

The script loads daily OHCLV data for a set of cryptocurrencies, performs
exploratory analysis (stationarity, correlations, visualization), builds a
preliminary causal Bayesian network, estimates propensity scores, matches
treated and control observations, and computes average treatment effects
(ATE) for returns and price-trend direction.
"""

__author__ = "Markus Schindler"
__copyright__ = "Copyright 2026"

__license__ = "Unlicense"
__version__ = "0.1.1"
__maintainer__ = "Markus Schindler"
__email__ = "schindlerdrmarkus@gmail.com"
__status__ = "Education"

# -------------------------- #
# Built-in / Generic Imports #
# -------------------------- #

import argparse
import logging
import sys
from pathlib import Path
# from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import statsmodels.api as sm

# pomegranate imports
from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork

# sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# --------------------- #
# Logging configuration #
# --------------------- #

log = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
log.addHandler(handler)
log.setLevel(logging.INFO)

# ---------------------------- #
# Data loading & preprocessing #
# ---------------------------- #

def load_coin_data(coin: str,
        data_dir: Path,
        sep: str = ","
    ) -> pd.DataFrame:
    log.info("Loading data from %s", data_dir)
    try:
        df = pd.read_csv(data_dir, sep = sep, header = 1)
    except Exception as exc:
        log.error(f"Failed to read {data_dir}: {exc}")
        raise
    required = {"Date", "Close", "Unix"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        log.error(f"Missing required columns in {csv_path}: {missing}")
        raise ValueError(f"Missing columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Close", "Date", "Unix"]]

def load_full_coin_data(coin: str,
        data_dir: Path,
        sep: str = ","
    ) -> pd.DataFrame:
    csv_path = data_dir / f"Binance_{coin}USDT_d.csv"
    try:
        raw = pd.read_csv(csv_path, sep = sep, header = 1)
    except Exception as exc:
        log.error(f"Failed to read {data_dir}: {exc}")
    latest_unix = find_latest_unix(data_dir)
    raw["Date"] = pd.to_datetime(raw["Date"])
    df = raw[raw["Unix"] > latest_unix]
    return df

def normalize_coin(df: pd.DataFrame
    ) -> pd.DataFrame:
    log.info("Normalizing price data")
    series = df["Close"].copy()
    maximum = series.max()
    if maximum == 0:
        return series
    return series / maximum

def find_latest_unix(data_dir: Path
    ) -> int:
    latest_unix = 0
    for csv_path in data_dir.glob("*_d.csv"):
        try:
            unix_min = int(pd.read_csv(csv_path, sep = ",", header = 1)["Unix"].min())
            if unix_min > latest_unix:
                latest_unix = unix_min
        except Exception as exc:
            log.warning(f"Could not read {csv_path} for Unix check: {exc}")
    log.info(f"Latest Unix time entry = {latest_unix}")
    return latest_unix

def date_list(data_dir: Path
         ) -> pd.DataFrame:
    time = {}
    latest_unix = find_latest_unix(data_dir)
    csv_path = data_dir / f"Binance_BTCUSDT_d.csv"
    raw = load_coin_data("BTC", csv_path)
    raw = raw[raw["Unix"] > latest_unix]
    time["Date"] = raw["Date"]
    return pd.DataFrame(time)

def prepare_coin_frames(coin_list,
        data_dir: Path
    ):
    coin_frames = {}
    latest_unix = find_latest_unix(data_dir)

    for coin in coin_list:
        csv_path = data_dir / f"Binance_{coin}USDT_d.csv"
        try:
            raw = load_coin_data(coin, csv_path)
        except Exception:
            log.error(f"Skipping {coin} due to loading error.")
            continue
        
        # Filter to the common time range
        raw = raw[raw["Unix"] > latest_unix]
        
        # Normalization
        norm = normalize_coin(raw)
        norm.name = coin
        coin_frames[f"{coin}_s"] = norm
        log.info(f"Prepared normalized data for {coin}")
    return pd.DataFrame(coin_frames)

# -------------------- #
# Exploratory analysis #
# -------------------- #

def test_stationarity(coin_frames: dict
    ) -> pd.DataFrame:
    # Run Augmented Dickey-Fuller test on each normalized series
    results = []
    for coin in coin_frames.items():
        adf_stat = sm.tsa.stattools.adfuller(coin[1])[0]
        adf_p = sm.tsa.stattools.adfuller(coin[1])[1]
        adf_lags = sm.tsa.stattools.adfuller(coin[1])[2]
        results.append(
            {
                "Coin": coin[0].replace("_s",""),
                "Test Statistics": adf_stat,
                "P-Value": adf_p,
                "Used Lags": adf_lags
            }
        )
    df_res = pd.DataFrame(results)
    log.info("Stationarity test results:\n%s", df_res)
    return df_res

def plot_trends(coin_frames: dict,
        data_dir: Path,
        output_path: Path):
    """Create a multi‑panel time‑series plot for all coins."""
    # Order coins by market‑cap (hard‑coded list)
    coin_cap = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "DOGE", "BCH", "ADA"]
    plot_df = pd.concat([coin_frames[f"{c}_s"] for c in coin_cap], axis = 1)
    plot_df.columns = coin_cap
    plot_df["Date"] = date_list(data_dir)

    colors = sns.color_palette("winter_r", len(coin_cap))
    n_rows = len(coin_cap)
    fig, axes = plt.subplots(
        nrows = n_rows, ncols = 1, figsize = (10, 1.5 * n_rows), sharex = True
    )
    if n_rows == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, coin_cap)):
        ax.plot(plot_df["Date"], plot_df[col], label = col, linewidth = 1.0, color = colors[i])
        ax.set_ylabel(col, fontsize = 10)
        ax.grid(True, linestyle = "--", alpha = 0.4)
        ax.tick_params(axis = "both", which = "both", direction = "in")
    axes[-1].set_xlabel("Date", fontsize = 12)
    plt.xticks(rotation = 45)
    fig.suptitle("Various Crypto Coin Trends", fontsize = 16, weight = "bold")
    plt.tight_layout(rect = [0, 0.03, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)
    log.info(f"Trend plot saved to {output_path}")

def compute_correlation_matrix(coin_frames: dict,
        coin_order: list) -> pd.DataFrame:
    """Calculate Pearson correlation matrix for the ordered series."""
    corr_data = []
    for coin in coin_order:
        df = coin_frames.get(f"{coin}_s")
        if df is None:
            log.warning(f"{coin}_s not found, skipping.")
            continue
        corr_data.append(df)
    corr_df = pd.concat(corr_data, axis = 1)
    corr_df.columns = coin_order
    corr_matrix = corr_df.corr()
    log.info("Correlation matrix computed.")
    return corr_matrix

def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
        output_path: Path):
    """Heatmap visualisation of the correlation matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot = True,
        cmap = "coolwarm",
        fmt = ".2f",
        linewidths = 0.5,
    )
    plt.title("Correlation Matrix of Cryptocoins", weight = "bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    log.info(f"Correlation heatmap saved to {output_path}")

def build_preliminary_dag(corr_matrix: pd.DataFrame,
        threshold: float = 0.8) -> nx.DiGraph:
    """Create a simple DiGraph based on correlation > threshold."""
    edges = []
    n = len(corr_matrix)
    for i in range(n):
        for j in range(i, n):
            if i != j and corr_matrix.iloc[i, j] > threshold:
                edges.append((list(corr_matrix.columns)[i], list(corr_matrix.columns)[j]))
    log.info(f"Preliminary DAG constructed with {len(edges)} edges.")
    return nx.DiGraph(edges)

def plot_dag(dag: nx.DiGraph,
        output_path: Path):
    """Draw a directed acyclic graph."""
    fig, ax = plt.subplots(figsize = (8, 6))
    pos = nx.shell_layout(dag)
    nx.draw(
        dag,
        pos,
        with_labels = True,
        node_color = "lightseagreen",
        edge_color = "dimgray",
        node_size = 3000,
        font_size = 10,
        font_weight = "bold",
        arrowsize = 20,
        width = 2
    )
    plt.title("Causal Graph - Preliminary Model", weight = "bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    log.info(f"DAG plot saved to {output_path}")

# ----------------------------------------------------------------------
# Binary state preparation (up/down)
# ----------------------------------------------------------------------

def prepare_binary_states(coin_frames: dict) -> dict:
    """Create up/down binary columns for each coin."""
    binary_frames = {}
    for coin, df in coin_frames.items():
        # Shift and compare directly, filling the first row NaN with 0
        binary_frames[f"{coin}_b"] = (df.diff() < 0).astype(int).fillna(0).values
        log.debug(f"{coin} binary state created.")
    return binary_frames

# ----------------------------------------------------------------------
# Bayesian Network fitting
# ----------------------------------------------------------------------

def define_bayesian_network():
    """Manually define the conditional categorical distributions used in the BN."""
    # --- Definition of Distributions ---
    # BTC (Index 0) & ETH (Index 1)
    # The initial probability was set to 0.5
    p = 0.5 

    d_BTC = Categorical([[p, p]])
    d_ETH = Categorical([[p, p]])

    # BNB (Index 2) depends from BTC
    d_BNB = ConditionalCategorical([[[p, p], [p, p]]])

    # TRX (Index 3) depends from BTC
    d_TRX = ConditionalCategorical([[[p, p], [p, p]]])

    # SOL (Index 4) depends both from BTC and ETH
    # Important additional brackets for this child, due to the existence of two parental nodes
    d_SOL = ConditionalCategorical([[[[p, p], [p, p]], [[p, p], [p, p]]]])

    # XRP (Index 5) depends from TRX
    d_XRP = ConditionalCategorical([[[p, p], [p, p]]])

    # Creation of Bayesian Network with list of distributions and edges
    # Edges are selected from the preliminary model; see directed acyclic graph
    crypto_bn = BayesianNetwork([d_BTC, d_ETH, d_BNB, d_TRX, d_SOL, d_XRP],
                                [(d_BTC, d_BNB), (d_BTC, d_TRX), (d_BTC, d_SOL),
                                 (d_ETH, d_SOL), (d_TRX, d_XRP)])
    return crypto_bn
    
def fit_bayesian_network(coin_binary_dict: dict
        ) -> BayesianNetwork:
    """Fit the Bayesian Network on the binary data."""
    
    # Convert binary columns to a 2‑D int array
    X_list = []
    for coin in ["BTC", "ETH", "BNB", "TRX", "SOL", "XRP"]:
        arr = coin_binary_dict[f"{coin}_s_b"]
        X_list.append(arr)
    X_np = np.column_stack(X_list)
    X_tensor = torch.tensor(X_np, dtype = torch.int32)

    # Build the network
    bn = define_bayesian_network()
    bn.fit(X_tensor)
    log.info("Bayesian Network fitted.")
    return bn

def probability_from_tensor(tensor):
    """Convert a Pomegranate CPD tensor to a readable percentage string."""
    value = round(100 * float(tensor), 1)
    return f"{value} %"

def extract_node_probabilities(bn: BayesianNetwork,
        coin_binary_dict: dict):
    """Return a dict mapping node names to probability strings."""
    probs = {}
    # BTC (index 0) – probability of Down = 1 (second element)
    p_ETH = probability_from_tensor(bn.distributions[1].probs[0][1])
    p_BNB = probability_from_tensor(bn.distributions[2].probs[0][1][1])
    p_TRX = probability_from_tensor(bn.distributions[3].probs[0][1][1])
    p_SOL = probability_from_tensor(bn.distributions[4].probs[0][1][1][1])
    p_XRP = probability_from_tensor(bn.distributions[5].probs[0][1][1])

    probs["BTC"] = f"BTC\n100 %" # BTC is forced to Down in this analysis
    probs["ETH"] = f"ETH\n{p_ETH}"
    probs["BNB"] = f"BNB\n{p_BNB}"
    probs["TRX"] = f"TRX\n{p_TRX}"
    probs["SOL"] = f"SOL\n{p_SOL}"
    probs["XRP"] = f"XRP\n{p_XRP}"
    log.info("Node probabilities extracted.")
    return probs

def plot_final_dag(node_probs: dict,
        output_path: Path):
    """Create a final DAG that includes the probability labels."""
    # Hard‑coded edges based on the original analysis
    edges = [
        (node_probs["BTC"], node_probs["TRX"]),
        (node_probs["BTC"], node_probs["BNB"]),
        (node_probs["TRX"], node_probs["XRP"]),
        (node_probs["BTC"], node_probs["SOL"]),
        (node_probs["ETH"], node_probs["SOL"]),
    ]

    G = nx.DiGraph(edges)
    fig, ax = plt.subplots(figsize = (8, 6))
    pos = nx.shell_layout(G)
    nx.draw(
        G,
        pos,
        with_labels = True,
        node_color = "lightseagreen",
        edge_color = "dimgray",
        node_size = 3000,
        font_size = 10,
        font_weight = "bold",
        arrowsize = 20,
        width = 2,
    )
    plt.title("Causal Graph with Probabilities for Down in BTC", weight = "bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    log.info(f"Final DAG plot saved to {output_path}")

# ----------------------------- #
# Volatility & RSI calculations #
# ----------------------------- #

def compute_volatility(
        coin: str,
        series: pd.Series) -> float:
    """Annualized volatility (standard deviation of daily returns)."""
    returns = series.pct_change().dropna()
    vol = returns.std()
    log.info(f"Volatility of {coin}: {vol:.6f}")
    return vol

def calculate_rsi(
        coin: str,
        series: pd.Series,
        window: int = 14) -> pd.Series:
    """Classic Wilder RSI implemenration."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window = window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window = window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    log.info(f"RSI calculated for {coin} (window = {window}).")
    return rsi

def add_moving_average(
        coin: str,
        series: pd.Series,
        window: int = 30) -> pd.Series:
    """30-day simple moving average."""
    sma = series.rolling(window = window).mean()
    log.info(f"30-day SMA calculated for {coin}.")
    return sma

# -------------------------- #
# Momentum‑strategy sampling #
# ---------------------------#

# --------------------------------------------------------------------- #
# We use BTC as the source dataframe for the sampling.                  #
# For the case that both signs of the observed trends of the momentum   #
# strategy are equal, the treatment could be applied, otherwise not.    #
# Ensure the indices are accessible via iloc for lookback calculations. #
# --------------------------------------------------------------------- #

def momentum_strategy(
        test_df: pd.DataFrame,
        ref_df: pd.DataFrame,
        n_samples: int = 1000,
        seed: int = 42):
    """Select rows where price trend aligns with BTC’s 30‑day SMA."""
    random.seed(seed)
    np.random.seed(seed)
  
    # Align indices via iloc
    available_indices = list(range(len(test_df)))
    sampled_indices = random.sample(available_indices, n_samples)
  
    sampled_rows = []
    for idx in sampled_indices:
        test_row = test_df.iloc[idx]
        ref_row = ref_df.iloc[idx]

        # Trend difference
        test_trend_diff = test_row["Close"] - test_row["SMA"]
        ref_trend_diff = ref_row["Close"] - ref_row["SMA"]
  
        # Same sign => treat as 1, else 0
        treatment = 0 if (test_trend_diff / ref_trend_diff) < 0 else 1
        
        # Look‑back for return calculation
        lookback_idx = max(0, idx - 30)
        price_then = test_df.iloc[lookback_idx]["Close"]
        price_return = test_row["Close"] - price_then

        if price_return and treatment >= 0:
            sampled_rows.append(
                {
                    "Date": test_row["Date"],
                    "Open": test_row["Open"],
                    "Close": test_row["Close"],
                    "Low": test_row["Low"],
                    "High": test_row["High"],
                    "Volume": test_row["Volume SOL"],
                    "Tradecount": test_row["tradecount"],
                    "RSI": test_row["RSI"],
                    "Return": price_return,
                    "Trend": test_trend_diff,
                    "Treatment": treatment,
                }
            )
    df_sampled = pd.DataFrame(sampled_rows)
    df_sampled.dropna(inplace = True)
    log.info(f"Momentum strategy produced {len(df_sampled)} samples.")
    return df_sampled

# --------------------------- #
# Propensity Score & Matching #
# --------------------------- #

def compute_propensity_scores(
        df: pd.DataFrame,
        features: list,
        seed: int = 42):
    """Fit logistic regression to estimate propensity scores."""
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df["Treatment"]

    model = LogisticRegression(random_state = seed)
    model.fit(X_scaled, y)
    df["Propensity Score"] = model.predict_proba(X_scaled)[:, 1]
    log.info("Propensity scores computed.")
    return df, model, scaler

def match_nearest_neighbors(
        df_matched: pd.DataFrame,
        caliper: float = 0.05,
        n_neighbors: int = 1):
    """Nearest‑neighbour matching on the propensity score."""
    treated = df_matched[df_matched["Treatment"] == 1]
    control = df_matched[df_matched["Treatment"] == 0]

    nn = NearestNeighbors(n_neighbors = n_neighbors, radius = caliper)
    nn.fit(control[["Propensity Score"]])
    distances, indices = nn.kneighbors(treated[["Propensity Score"]])
    matched_control = control.iloc[indices.flatten()]
    matched_data = pd.concat([treated.reset_index(drop = True), matched_control.reset_index(drop = True)], axis = 1)
    # rename columns to indicate source
    treated_cols = [f"treated_{c}" for c in treated.columns]
    control_cols = [f"control_{c}" for c in matched_control.columns]
    matched_data.columns = treated_cols + control_cols
    log.info("Nearest‑neighbour matching completed.")
    return matched_data

def estimate_ate(
        df: pd.DataFrame):
    """Calculate ATE for Return and Trend with non‑parametric bootstrapping."""
    ate_return = df["treated_Return"] - df["control_Return"]
    ate_trend = df["treated_Trend"] - df["control_Trend"]

    # Bootstrap for confidence intervals (10,000 resamples)
    n_boot = 10_000
    boot_return = np.random.choice(ate_return, size = n_boot, replace = True)
    boot_trend = np.random.choice(ate_trend, size = n_boot, replace = True)

    ci_return = np.percentile(boot_return, [2.5, 97.5])
    ci_trend = np.percentile(boot_trend, [2.5, 97.5])

    log.info(f"ATE Return: {ate_return.mean():.4f} (+/- {ate_return.std():.4f}) [CI {ci_return[0]:.4f}, {ci_return[1]:.4f}]")
    log.info(f"ATE Trend: {ate_trend.mean():.4f} (+/- {ate_trend.std():.4f}) [CI {ci_trend[0]:.4f}, {ci_trend[1]:.4f}]")
    return {
        "ate_return_mean": ate_return.mean(),
        "ate_return_sd": ate_return.std(),
        "ate_return_ci_low": ci_return[0],
        "ate_return_ci_high": ci_return[1],
        "ate_trend_mean": ate_trend.mean(),
        "ate_trend_sd": ate_trend.std(),
        "ate_trend_ci_low": ci_trend[0],
        "ate_trend_ci_high": ci_trend[1],
    }

def plot_balance_histograms(
        df_original: pd.DataFrame,
        df_matched: pd.DataFrame,
        output_path: Path):
    """Histogram of key metrics before and after matching."""
    fig, axes = plt.subplots(1, 2, figsize = (10, 4))
    for i, feat in enumerate(["Return", "Trend"]):
        ax = axes[i]
        ax.hist(
            df_original[df_original["Treatment"] == 1][feat],
            color = "orange",
            alpha = 0.5,
            label = "Treated",
            density = True,
        )
        ax.hist(
            df_original[df_original["Treatment"] == 0][feat],
            color = "dodgerblue",
            alpha = 0.5,
            label = "Control",
            density = True,
        )
        ax.set_title(f"Pre‑Match: {feat}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    log.info(f"Balance histograms saved to {output_path}")   

# ------------------------------- #
# Argument parsing & main routine #
# ------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Causal inference analysis of cryptocurrencies trends.")
    parser.add_argument(
        "--data-dir",
        type = str,
        default = "data",
        help = "Directory containing the cryptocurrency CSV files (*_d.csv)."
    )
    parser.add_argument(
        "--output-dir",
        type = Path,
        default = Path("output"),
        help = "Directory where plots and intermediate files will be saved."
    )
    parser.add_argument(
        "--correlation-threshold",
        type = float,
        default = 0.8,
        help = "Correlation threshold for building the preliminary DAG."
    )
    parser.add_argument(
        "--sample-size",
        type = int,
        default = 1000,
        help = "Number of random rows to include in the momentum‑strategy sample."
    )
    parser.add_argument(
        "--random-seed",
        type = int,
        default = 42,
        help = "Random seed for reproducibility."
    )
    parser.add_argument(
        "--caliper",
        type=float,
        default = 0.05,
        help = "Caliper (maximum distance) for nearest‑neighbour matching."
    )
    parser.add_argument(
        "--n-neighbors",
        type = int,
        default = 1,
        help = "Number of neighbours for nearest‑neighbour matching."
    )        
    return parser.parse_args()

# ------------ #
# Main Routine #
# ------------ #

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents = True, exist_ok = True)

    # -------------------- #
    # 1. Load & clean data #
    # -------------------- #
    
    coin_list = [
            "ADA",
            "BCH",
            "BNB",
            "BTC",
            "DOGE",
            "ETH",
            "SOL",
            "TRX",
            "XRP"
    ]
    data_dir = Path(args.data_dir)
    coin_frames = prepare_coin_frames(coin_list, data_dir)

    # ----------------------------------------- #
    # 2. Exploratory plots & correlation matrix #
    # ----------------------------------------- #
    
    test_stationarity(coin_frames)
    plot_trends(coin_frames, data_dir, args.output_dir / "Trends.png")
    coin_cap_list = [
            "BTC",
            "ETH",
            "BNB",
            "XRP",
            "SOL",
            "TRX",
            "DOGE",
            "BCH",
            "ADA"
    ]
    corr_matrix = compute_correlation_matrix(coin_frames, coin_cap_list)
    plot_correlation_heatmap(corr_matrix, args.output_dir / "CorrMatrix.png")

    # -------------------------------------- #
    # 3. Preliminary DAG (correlation based) #
    # -------------------------------------- #
    
    dag = build_preliminary_dag(corr_matrix, threshold = args.correlation_threshold)
    plot_dag(dag, args.output_dir / "ACG.png")

    # ------------------------------------------------ #
    # 4. Binary state preparation for Bayesian Network #
    # -------------------------------------------------#
    
    binary_frames = prepare_binary_states(coin_frames)

    # ------------------------------------------------------------------ #
    # 5. Fit Bayesian Network on a subset (BTC, ETH, BNB, TRX, SOL, XRP) #
    # ------------------------------------------------------------------ #
    
    bn = fit_bayesian_network(binary_frames)

    # Extract probabilities for final plot
    node_probs = extract_node_probabilities(bn, binary_frames)
    plot_final_dag(node_probs, args.output_dir / "ACG_final.png")

    # --------------------------------------------------------------------------------------- #
    # Based on the analysis, the price trajectories of most cryptocurrencies closely track    #
    # those of Bitcoin (BTC) or Ethereum (ETH). Consequently, altcoin trends generally follow #
    # similar cycles, albeit with varying intensities. In particular, four alternative coins  #
    # exhibited roughly an 80 % probability of moving in the same direction (up or down) as   #
    # BTC’s daily closing price.                                                              #
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Building on the earlier analysis, it is worthwhile to forecast the price movements of   #
    # altcoins by leveraging Bitcoin’s trend.  When Bitcoin is on a positive trajectory, a    #
    # momentum strategy can be useful: the closing price is compared against a 30‑day simple  #
    # moving average (SMA). If the current price exceeds the SMA, a high probability of a     #
    # continued up‑move is implied. In such a scenario there is a strong likelihood that the  #
    # same pattern would hold for other coins, as the earlier study suggested an ~ 80 %       #
    # alignment of sign with BTC.  To illustrate, I applied this methodology to Solana (SOL), #
    # observing that its daily closing price similarly trended above its 30‑day SMA during    #
    # the same period.                                                                        #
    # --------------------------------------------------------------------------------------- #

    # ------------------- #
    # 6. Volatility & RSI #
    # ------------------- #

    BTC = load_full_coin_data("BTC", data_dir)
    SOL = load_full_coin_data("SOL", data_dir)

    volatility = {
        "BTC": compute_volatility("BTC", BTC["Close"]),
        "SOL": compute_volatility("SOL", SOL["Close"])
    }
   
    # RSI
    BTC["RSI"] = calculate_rsi("BTC", BTC["Close"])
    SOL["RSI"] = calculate_rsi("SOL", SOL["Close"])

    # 30-day SMA
    BTC["SMA"] = add_moving_average("BTC", BTC["Close"], window = 30)
    SOL["SMA"] = add_moving_average("SOL", SOL["Close"], window = 30)

    # ----------------------------------- #
    # 7. Momentum-strategy sample for SOL #
    # ----------------------------------- #

    SOL_sampled = momentum_strategy(
        test_df = SOL,
        ref_df = BTC,
        n_samples = args.sample_size,
        seed = args.random_seed,
    )

    # ------------------------------ #
    # 8. Propensity score & matching #
    # -------------------------------#

    # ---------------------------------------------------------------------------------------------- #
    # The propensity score is the probability that a given time‑point of a cryptocurrency will be    #
    # chosen for a momentum trade, conditioned on the observed covariates—such as trading volume,    #
    # closing prices, volatility, etc.. All of those features can be assembled into a vector X.      #
    # When we split the data into “treated” (momentum‑traded) and “control” (non‑traded) groups,     #
    # the two groups can differ systematically in X, which biases any estimate of the treatment      #
    # effect. Conditioning on the propensity score remedies this: if the distribution of key         #
    # outcomes (e.g., trend direction or return) is balanced across the groups and there is          #
    # substantial overlap in the score values, confounding is largely eliminated.                    #
    # ---------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------- #
    # A logistic regression is fitted to all covariates in order to estimate the probability of      #
    # receiving the treatment. In the model the outcome is coded as "Treated" = 1 and "Control" = 0. #
    # The resulting fitted probability for each observation is the propensity score. Once the scores #
    # are estimated, a nearest‑neighbor algorithm can be applied: each treated unit is paired with   #
    # one or more control units whose propensity scores are closest. This matching reduces bias      #
    # because the paired observations are balanced on the covariates that influenced treatment       #
    # assignment. After the match, any single-parameter outcome—such as the asset return or trend    #
    # can be compared across the two groups.                                                         #
    # ---------------------------------------------------------------------------------------------- #
    
    features = [
        "Open",
        "Close",
        "Low",
        "High",
        "Volume",
        "Tradecount",
        "RSI",
        "Return",
        "Trend",
    ]

    # ---------------------------------------------------------------------------------------------- #
    # Nearest‑Neighbor matching on the propensity score is a practical, data‑driven shortcut to      #
    # emulate a randomized experiment: it takes a single‑number approximation of covariate           #
    # similarity the propensity score and pulls the most similar control for each treated case.      #
    # Because the propensity score is theoretically the minimal sufficient summary of covariates     #
    # for balancing, the nearest‑match algorithm keeps bias low while remaining computationally      #
    # simple. That’s why it is the default go‑to method in most propensity‑score workflows.          #
    # ---------------------------------------------------------------------------------------------- #
    
    SOL_sampled, lr_model, scaler = compute_propensity_scores(
        SOL_sampled, features, seed = args.random_seed
    )

    # Create treated / control split
    treated = SOL_sampled[SOL_sampled["Treatment"] == 1].copy()
    control = SOL_sampled[SOL_sampled["Treatment"] == 0].copy()

    # Nearest‑neighbour matching
    matched_data = match_nearest_neighbors(
        pd.concat([treated, control], ignore_index = True),
        caliper = args.caliper,
        n_neighbors = args.n_neighbors,
    )

    # ----------------- #
    # 9. ATE estimation #
    # ------------------#
    
    ate_results = estimate_ate(matched_data)

    # ------------------------- #
    # 10. Balance visualisation #
    # ------------------------- #
    
    plot_balance_histograms(
        df_original = SOL_sampled,
        df_matched = matched_data,
        output_path = args.output_dir / "Histogram.png",
    )

    log.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
