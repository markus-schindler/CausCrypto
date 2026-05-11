#"/usr/bin/python
# -*- coding: utf-8 -*-

"""CausCrypto.py: Python script for causal inference on cryptocoins"""

__author__ = "Markus Schindler"
__copyright__ = "Copyright 2026"

__license__ = "Unlicense"
__version__ = "0.1.0"
__maintainer__ = "Markus Schindler"
__email__ = "schindlerdrmarkus@gmail.com"
__status__ = "Education"

# Built-in / Generic Imports
import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import statsmodels.api as sm
from pomegranate.distributions import Categorical
from pomegranate.distributions import ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

#################
# Load the Data #
#################

# Data set contains following crypto coins from https://www.cryptodatadownload.com
cryptocurrencies = ['ADA', 'BCH', 'BNB', 'BTC', 'DOGE',
                    'ETH', 'SOL', 'TRX', 'XRP']

for coin in cryptocurrencies:
    globals()[coin] = pd.read_csv(f'Binance_{coin}USDT_d.csv', sep = ',', header = 1)
    globals()[coin]['Date'] = pd.to_datetime(globals()[coin]['Date'])

####################
# Data Preparation #
####################

# Data is inconsistent with overlapping time ranges
# Identification of the latest unix time covering all coin trends
latest_unix = 0
for coin in cryptocurrencies:
    _ = int(globals()[coin]['Unix'].min())
    if _ > latest_unix:
        latest_unix = _
print('Latest Unix time entry =', latest_unix)

# Most coins have different USDT values
# Normalization creating the coin dataframes ending with _s

for coin in cryptocurrencies:
    coin_s = coin + '_s'
    globals()[coin_s] = globals()[coin]['Close'][globals()[coin]['Unix'] > latest_unix]
    maximum = globals()[coin_s].max()
    globals()[coin_s] = globals()[coin_s] / maximum
    globals()[coin_s].name = coin

# Some coins show a positive trend value over a longer time period
# The augmented Dickey-Fuller test can show whether a trend is stationary or not

stationary_data = []
for coin in cryptocurrencies:
    coin_s = coin + '_s'
    # Performing augmented Dickey-Fuller test for stationarity
    Test = sm.tsa.stattools.adfuller(globals()[coin_s])[0]
    p_value = sm.tsa.stattools.adfuller(globals()[coin_s])[1]
    used_lags = sm.tsa.stattools.adfuller(globals()[coin_s])[2]
    stationary_data.append({'Coin': coin,
                            'Test Statistics': Test,
                            'P-Value': p_value,
                            'Used Lags': used_lags})
stationarity = pd.DataFrame(stationary_data)
print(stationarity)

# Creation of date data series for all data
date = BTC['Date'][BTC['Unix'] > latest_unix]
date = pd.DataFrame(date)

# Coins were sorted according to their market capitalization
coin_cap = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL',
            'TRX', 'DOGE', 'BCH', 'ADA']
result = pd.concat([date, BTC_s, ETH_s, BNB_s, XRP_s, SOL_s,
                    TRX_s, DOGE_s, BCH_s, ADA_s], axis = 1)

##############
# Data Plots #
##############

#######################
# Plotting the Trends #
#######################

# Plotting all trends in a substacked chart
# Some known coins show a similarity in time behavior

plot_df = result
currency_cols = [col for col in plot_df.columns if col != 'Date']

# Define a distinct color palette
colors = sns.color_palette("winter_r", len(currency_cols))

fig, axes = plt.subplots(nrows = len(currency_cols), ncols = 1, figsize = (10, 1.5 * len(currency_cols)),
                        sharex = True)

# Ensure axes is iterable
if len(currency_cols) == 1:
    axes = [axes]

# Plot each currency with modified parameters
for i, (ax, col) in enumerate(zip(axes, currency_cols)):
    ax.plot(plot_df['Date'], plot_df[col], label = col, linewidth = 1.0, color = colors[i])
    ax.set_ylabel(col, fontsize = 10)
    ax.grid(True, linestyle = '--', alpha = 0.4)
    # Set ticks to be inline (pointing inwards)
    ax.tick_params(axis = 'both', which = 'both', direction = 'in')

# Set the common X-label
axes[-1].set_xlabel('Date', fontsize = 12)
plt.xticks(rotation = 45)

# Add a main title on the top of the entire figure
fig.suptitle('Various Crypto Coin Trends', fontsize = 16, weight = 'bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('Trends.png')

######################
# Correlation Matrix #
######################

# Based on the trends Pearson correlation could be helpful
# to identify similar behavior

# Create a DataFrame containing all the normalized series (_s) in the order of market cap
corr_df = pd.concat([globals()[coin + '_s'] for coin in coin_cap], axis = 1)
corr_df.columns = coin_cap

# Calculate the correlation matrix
corr_matrix = corr_df.corr()

# Visualise the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title("Correlation Matrix of Cryptocoins", weight = 'bold')
plt.savefig('CorrMatrix.png')

# Listing up the major contributions for each crypto coin
corr = corr_matrix.to_numpy()
corr = corr.copy()
number_coins = len(coin_cap)

for i in range(number_coins):
    corr[i][i] = 0

# Here the number was set to 20 to cover all
# impacts above 80 percent
flat_corr = corr.flatten()
top_indices = np.argsort(flat_corr)[-20:][::-1]

for idx in top_indices:
    row, col = divmod(idx, len(coin_cap))
    print(f"{coin_cap[row]} -> {coin_cap[col]} {flat_corr[idx]}")

######################################
# Directed Acyclic Graph Preparation #
######################################

# Based on the made observations edges in a directed acyclic graph
# could be defined. Correlations larger than 0.8 were
# selected for a preliminary causel inference model
crypto_graph = nx.DiGraph([('BTC', 'TRX'),
                           ('BTC', 'BNB'),
                           ('TRX', 'XRP'),
                           ('BTC', 'SOL'),
                           ('ETH', 'SOL'),
                        ])

# Plotting the DAG
plt.figure(figsize=(8, 6))
pos = nx.shell_layout(crypto_graph)
nx.draw(crypto_graph, pos, 
        with_labels = True, 
        node_color = 'lightseagreen', 
        edge_color = 'dimgray', 
        node_size = 3000, 
        font_size = 10, 
        font_weight = 'bold', 
        arrowsize = 20, 
        width = 2)
plt.title("Causal Graph - Preliminary Model", weight = 'bold')
plt.savefig('ACG.png')

############################################
# Sensitivity Analsysis | Bayesian Network #
############################################

# Preparing the data with binary states
for coin in cryptocurrencies:
    coin_s = coin + '_s'
    coin_b = coin + '_b'
    temp_array = np.array(globals()[coin_s])
    globals()[coin_b] = np.zeros_like(temp_array)
    for i in range(1, len(temp_array)):
        if temp_array[i] - temp_array[i-1] < 0:
            # Down change stated as one
            globals()[coin_b][i] = 1
        else:
            # Up change stated as zero
            globals()[coin_b][i] = 0
    globals()[coin_b][0] = 0

# Create trainings data
X0 = np.column_stack((BTC_b, ETH_b, BNB_b, TRX_b, SOL_b, XRP_b))

# Performing selectivity analysis changing all BTC values to "Down" = 1
X = []
for i in range(len(BTC_b)):
    if BTC_b[i] == 1:
        X.append(X0[i])
X_np = np.array(X)
X_tensor = torch.tensor(X_np, dtype = torch.int32)

####################
# Bayesian Network #
####################

# --- Definition of Distributions ---
# BTC (Index 0) & ETH (Index 1)
# The initial probability was set to 0.5

d_BTC = Categorical([[0.5, 0.5]])
d_ETH = Categorical([[0.5, 0.5]])

# BNB (Index 2) depends from BTC
d_BNB = ConditionalCategorical([[[0.5, 0.5], [0.5, 0.5]]])

# TRX (Index 3) depends from BTC
d_TRX = ConditionalCategorical([[[0.5, 0.5], [0.5, 0.5]]])

# SOL (Index 4) depends both from BTC and ETH
# Important additional brackets for this child, due to the existence of two parental nodes
d_SOL = ConditionalCategorical([[[
    [0.5, 0.5], # BTC = 0, ETH = 0
    [0.5, 0.5]  # BTC = 0, ETH = 1
], [
    [0.5, 0.5], # BTC = 1, ETH = 0
    [0.5, 0.5]  # BTC = 1, ETH = 1
]]])

# XRP (Index 5) depends from TRX
d_XRP = ConditionalCategorical([[[0.5, 0.5], [0.5, 0.5]]])

# Creation of Bayesian Network with list of distributions and edges
# Edges are selected from the preliminary model; see directed acyclic graph
crypto_bn = BayesianNetwork([d_BTC, d_ETH, d_BNB, d_TRX, d_SOL, d_XRP],
                            [(d_BTC, d_BNB), (d_BTC, d_TRX), (d_BTC, d_SOL),
                             (d_ETH, d_SOL), (d_TRX, d_XRP)]) 
                                                                         
# Initiate fitting
crypto_bn.fit(X_tensor)

# Function for delivering the probability as a string output in percent
def probability(tensor):
    # Creates string with percentage
    value = round(100 * float(tensor), 1)
    output = str(value) + ' %'
    return output

# All conditional probabilities for the crypto coins having BTC as 100 % Down
p_ETH = probability(crypto_bn.distributions[1].probs[0][1])
p_BNB = probability(crypto_bn.distributions[2].probs[0][1][1])
p_TRX = probability(crypto_bn.distributions[3].probs[0][1][1])
p_SOL = probability(crypto_bn.distributions[4].probs[0][1][1][1])
p_XRP = probability(crypto_bn.distributions[5].probs[0][1][1])

# Creating the labeling for the updated DAG
node_0 = 'BTC\n100 %'
node_1 = 'ETH\n' + p_ETH
node_2 = 'BNB\n' + p_BNB
node_3 = 'TRX\n' + p_TRX
node_4 = 'SOL\n' + p_SOL
node_5 = 'XRP\n' + p_XRP

################################
# Final Directed Acyclic Graph #
################################

crypto_graph = nx.DiGraph([(node_0, node_3),
                           (node_0, node_2),
                           (node_3, node_5),
                           (node_0, node_4),
                           (node_1, node_4),
                        ])

plt.figure(figsize=(8, 6))
pos = nx.shell_layout(crypto_graph)
nx.draw(crypto_graph, pos, 
        with_labels=True, 
        node_color='lightseagreen', 
        edge_color='dimgray', 
        node_size=3000, 
        font_size=10, 
        font_weight='bold', 
        arrowsize=20, 
        width=2)
plt.title('Causal Graph with Probabilities for "Down" in BTC', weight='bold')
plt.savefig('ACG_final.png')

# Based on the analysis, the price trajectories of most cryptocurrencies closely track
# those of Bitcoin (BTC) or Ethereum (ETH). Consequently, altcoin trends generally follow
# similar cycles, albeit with varying intensities. In particular, four alternative coins
# exhibited roughly an 80 % probability of moving in the same direction (up or down) as
# BTC’s daily closing price.

#########################################################################################

# Building on the earlier analysis, it is worthwhile to forecast the price movements of
# altcoins by leveraging Bitcoin’s trend.  When Bitcoin is on a positive trajectory, a 
# momentum strategy can be useful: the closing price is compared against a 30‑day simple
# moving average (SMA). If the current price exceeds the SMA, a high probability of a 
# continued up‑move is implied. In such a scenario there is a strong likelihood that the
# same pattern would hold for other coins, as the earlier study suggested an ~ 80 % 
# alignment of sign with BTC.  To illustrate, I applied this methodology to Solana (SOL),
# observing that its daily closing price similarly trended above its 30‑day SMA during
# the same period.

#################################
# Calculation of the Volatility #
#################################

BTC_volatility = BTC['Close'].pct_change().std()
SOL_volatility = SOL['Close'].pct_change().std()
print('Volatility of BTC =', BTC_volatility, '\nVolatility of SOL =', SOL_volatility)

##############################################
# Calculation of the Relative Strength Index #
##############################################

# Definition of the RSI function
def calculation_rsi(price, window = 14):
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window = window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
    relative_strength = gain / loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi

# Adding RSI values
BTC['RSI'] = calculation_rsi(BTC['Close'])
SOL['RSI'] = calculation_rsi(SOL['Close'])

#########################################
# Calculation of Single Moving Averages #
#########################################

# Calculation of SMA
BTC['30d_MA'] = BTC['Close'].rolling(window = 30).mean()
SOL['30d_MA'] = SOL['Close'].rolling(window = 30).mean()

# We use BTC as the source dataframe for the sampling
# For the case that both signs of the observed trends of the momentum
# strategy are equal, the treatment could be applied, otherwise not.
# Ensure the indices are accessible via iloc for lookback calculations
available_indices = list(range(len(SOL)))
random_indices = random.sample(available_indices, 1000)

sampled_data = []

for idx in random_indices:
    # Get current row data
    row = SOL.iloc[idx]
    row2 = BTC.iloc[idx]
    current_close = row['Close']
    reference_close = row2['Close']
    current_ma = row['30d_MA']
    reference_ma = row2['30d_MA']
    current_date = row['Date']
    current_open = row['Open']
    current_high = row['High']
    current_low = row['Low']
    current_volume = row['Volume SOL']
    current_tradecount = row['tradecount']
    current_RSI = row['RSI']
    # Calculate trends
    current_trend = current_close - current_ma
    reference_trend = reference_close - reference_ma
    # For the case that both trends have the same sign this case can be considered as treatment
    if current_trend / reference_trend < 0:
        treatment = 0
    else:
        treatment = 1
    
    # Calculate Return: (Price_t - Price_{t - 30}) / Price_{t - 30}
    # Using max(0, idx - 30) to handle entries near the start of the series
    lookback_idx = max(0, idx - 30)
    price_then = SOL.iloc[lookback_idx]['Close']
    price_return = current_close - price_then
    
    # Considering only positive return and trends
    if price_return and current_trend >= 0:
        sampled_data.append({
            'Date': current_date,
            'Open': current_open,
            'Close': current_close,
            'Low': current_low,
            'High': current_high,
            'Volume': current_volume,
            'Tradecount': current_tradecount,
            'RSI': current_RSI,
            'Return': price_return,
            'Trend': current_trend,
            'Treatment': treatment
        })

# Create the final pandas dataframe
SOL_sampled = pd.DataFrame(sampled_data)

# Remove the nan entries
SOL_sampled = SOL_sampled.dropna()

####################
# Basic Statistics #
####################

# Calculate descriptive statistics
descriptive_stats = SOL_sampled.describe().T  # Transpose for better readability
descriptive_stats = descriptive_stats[['mean', 'std', 'min', 'max']]
descriptive_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
print(descriptive_stats)

####################
# Propensity Score #
####################

# The propensity score is the probability that a given time‑point of a cryptocurrency will be
# chosen for a momentum trade, conditioned on the observed covariates—such as trading volume,
# closing prices, volatility, etc.. All of those features can be assembled into a vector X.
# When we split the data into “treated” (momentum‑traded) and “control” (non‑traded) groups,
# the two groups can differ systematically in X, which biases any estimate of the treatment
# effect. Conditioning on the propensity score remedies this: if the distribution of key
# outcomes (e.g., trend direction or return) is balanced across the groups and there is#
# substantial overlap in the score values, confounding is largely eliminated.

#######################
# Logistic Regression #
#######################

# A logistic regression is fitted to all covariates in order to estimate the probability of
# receiving the treatment. In the model the outcome is coded as "Treated" = 1 and "Control" = 0.
# The resulting fitted probability for each observation is the propensity score. Once the scores
# are estimated, a nearest‑neighbor algorithm can be applied: each treated unit is paired with
# one or more control units whose propensity scores are closest. This matching reduces bias
# because the paired observations are balanced on the covariates that influenced treatment
# assignment. After the match, any single­parameter outcome—such as the asset return or trend
# can be compared across the two groups.

# Estimate Propensity Scores using logistic regression
# Propensity Score ranges from 0 to 1 and is used for matching
features = ['Open', 'Close', 'Low', 'High', 'Volume', 'Tradecount', 'RSI', 'Return', 'Trend']
X = SOL_sampled[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = SOL_sampled['Treatment']

model = LogisticRegression(random_state = 42)
model.fit(X_scaled, y)
SOL_sampled['Propensity Score'] = model.predict_proba(X_scaled)[:,1]

print("Sample Propensity Scores:")
print(SOL_sampled[['Treatment', 'Propensity Score']].head())

# Preparation of the treated and control group
treated = SOL_sampled[SOL_sampled['Treatment'] == 1]
control = SOL_sampled[SOL_sampled['Treatment'] == 0]

#####################
# Nearest Neighbors #
#####################

# Nearest‑Neighbor matching on the propensity score is a practical, data‑driven shortcut to
# emulate a randomized experiment: it takes a single‑number approximation of covariate
# similarity the propensity score and pulls the most similar control for each treated case.
# Because the propensity score is theoretically the minimal sufficient summary of covariates
# for balancing, the nearest‑match algorithm keeps bias low while remaining computationally
# simple. That’s why it is the default go‑to method in most propensity‑score workflows.

nn = NearestNeighbors(n_neighbors = 1, radius = 0.05) # Caliper = 0.05
nn.fit(control[['Propensity Score']])

distances, indices = nn.kneighbors(treated[['Propensity Score']])
matched_control = control.iloc[indices.flatten()]

# Combine matched pairs
matched_data = pd.concat([treated.reset_index(drop = True), 
                          matched_control.reset_index(drop = True)], axis = 1)
matched_data.columns = ['treated_' + col if i < len(treated.columns) else 'control_' + SOL_sampled.columns[i - len(treated.columns)] 
                       for i, col in enumerate(matched_data.columns)]

############################
# Average Treatment Effect #
############################

ate_r = matched_data['treated_Return'] - matched_data['control_Return']
ate_t = matched_data['treated_Trend'] - matched_data['control_Trend']
print(f"Estimated ATE for Return: {ate_r.mean():.4f} (+/- {ate_r.std():.4f})")
print(f"Estimated ATE for Trend: {ate_t.mean():.4f} (+/- {ate_t.std():.4f})")

#######################################
# Balance Visualization by Histograms #
#######################################

# Visualize balance
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, feat in enumerate(['Return', 'Trend']):
    ax[i].hist(SOL_sampled[SOL_sampled['Treatment'] == 1][feat], color = 'orange', alpha = 0.5, label='Treated')
    ax[i].hist(SOL_sampled[SOL_sampled['Treatment'] == 0][feat], color = 'dodgerblue', alpha = 0.5, label='Control')
    ax[i].set_title(f'Pre-Match: {feat}')
    ax[i].legend()
plt.tight_layout()
plt.savefig('Histogram.png')
