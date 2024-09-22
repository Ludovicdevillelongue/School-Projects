# -*- coding: utf-8 -*-
"""
@authors: Amine Mounazil, Ludovic de Villelongue
"""
import investpy
import pandas as pd
from functools import reduce
from pandas import *
from time import sleep
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import warnings
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

###########
## Utils ##
###########

# Functions to get data from investing.com
def get_stock(stock_name, start, end):
    stock = investpy.get_stock_historical_data(stock_name,
                                               country='United States',
                                               from_date=start,
                                               to_date=end)
    return stock

def get_crypto(crypto_name, start, end):
    crypto = investpy.get_crypto_historical_data(crypto_name,
                                                 from_date=start,
                                                 to_date=end)
    return crypto

def get_currency(currencies_name, start, end):
    currency = investpy.get_currency_cross_historical_data(currencies_name,
                                                           from_date=start,
                                                           to_date=end)
    return currency

def get_bond(bond_name, start, end):
    bond = investpy.get_bond_historical_data(bond_name, from_date=start,
                                             to_date=end)
    return bond

def get_commodity(commodity_name, start, end):
    commodity = investpy.get_commodity_historical_data(commodity_name, from_date=start,
                                                       to_date=end)

    return commodity

def get_indices(index_name, country, start, end):
    index = investpy.get_index_historical_data(index_name, country=country, from_date=start,
                                               to_date=end)
    return index

# Function that returns a dataframe containing close prices of financial products by family
def product_family(dict_product: dict, List_Product):
    df = pd.concat(dict_product, axis=1)
    df = df.droplevel(level=0, axis=1)
    df_close = df["Close"]
    df_close.columns = List_Product
    return df_close

# Perform window calculation
def get_lagged_data(df, period):
    data = df.copy(deep=True)
    data = data.shift(-period)
    return df

# Function that identifies and plots missing data
def identify_data(df):
    plt.figure(figsize=(16, 4))
    (len(df.index) - df.count()).plot.bar()
    print("Data frame shape: {0}".format(df.shape))
    
# Function that removes missing data
def clean_data(df):
    # df = df.drop(df.columns[df.apply(lambda col: col.isnull().sum() > 50)], axis=1)
    data = df.copy(deep=True)
    data = data.dropna()
    return df

# Compute returns
def compute_returns(df):
    data = df.copy(deep=True)
    slice = len(data.columns)
    for index in data.columns:
        data['Returns_{0}'.format(index)] = np.log(data[index] / data[index].shift(1)).dropna()
    return data.iloc[:, -slice:]

# Compute volatility
def compute_vol(df):
    data = df.copy(deep=True)
    slice = len(data.columns)
    for index in data.columns:
        data['Volatility_{0}'.format(index)] = (np.log(data[index] / data[index].shift(1)) ** 2).dropna()
    return data.iloc[: , -slice:]

# Calculate the empirical distribution of returns and detect crisis events (1D)
def identify_crisis_event(df):
    data = df.copy(deep=True)
    for index in df.columns:
        data['Returns_{0}'.format(index)] = np.log(data[index] / data[index].shift(1)).dropna()
        data['Volatility_{0}'.format(index)] = (np.log(data[index] / data[index].shift(1)) ** 2).dropna()
        # Compute the first percentile of each return series
        q = data['Returns_{0}'.format(index)].quantile(q=0.05)
        # Assign binary values if our 1% threshold was or wasn't exceeded
        data['Classification_1D_{0}'.format(index)] = np.where(data['Returns_{0}'.format(index)] < q, 1, 0)
    return data

# Compute crisis events for bonds and currencies from identify_crisi_event() output (binary variable)
def compute_crisis_events_bc(df):
    data = df.copy(deep=True)
    classification_columns = list([col for col in data.columns if 'Classification' in col])
    binary_predictor_variables = pd.DataFrame(data, columns=classification_columns)
    return binary_predictor_variables

# Compute indices crisis events by region (binary variable)
def compute_crisis_events_region(data:DataFrame, region):
    classification_columns = list([col for col in data.columns if 'Classification' in col])
    binary_predictor_variables = pd.DataFrame(data, columns=classification_columns)
    binary_predictor_variables["Event count"] = binary_predictor_variables.sum(axis=1)
    binary_predictor_variables["Classification_1D_{0}".format(region)] = 0
    if region == "Asia":
        binary_predictor_variables.loc[binary_predictor_variables["Event count"] >= 3, "Classification_1D_{0}".format(region)] = 1
    elif region == "Americas":
        binary_predictor_variables.loc[binary_predictor_variables["Event count"] >= 3, "Classification_1D_{0}".format(region)] = 1
    elif region == "Europe":
        binary_predictor_variables.loc[binary_predictor_variables["Event count"] >= 8, "Classification_1D_{0}".format(region)] = 1
    else:
        raise Exception("Region provided is not considered, possible region values are : Asia, Americas, Europe")
    return binary_predictor_variables.iloc[:,-1:]

# Calculate indices global crisis events (binary variable)
def compute_crisis_events_global(df_asia, df_americas, df_europe):
    binary_predictor_variables = pd.concat([df_americas, df_europe, df_asia], axis=1).reindex(df_americas.index)
    binary_predictor_variables["Event count"] = binary_predictor_variables.sum(axis=1)
    binary_predictor_variables["Global crisis event 1D"] = 0
    binary_predictor_variables.loc[binary_predictor_variables["Event count"] >= 2, "Global crisis event 1D"] = 1
    return binary_predictor_variables.iloc[:,-1:]

# Count crisis events for any product
def count_crisis_events(df):
    data = df.copy(deep=True)
    data["Total Events"] = data.sum(axis=1)
    return data.iloc[:,-1:]

# Average number of significant events during the last n days based on previous values of
# the binary predictor variable (regional/global) which identifies whether the number of
# events exceeded a threshold someday
def compute_average_sig_events(df, period=5):
    df_events = df.copy(deep=True)
    global_slice=df_events.iloc[:,-1]
    region_slice=df_events.iloc[:,-4:-2]
    length = len(global_slice)
    for i in range(length):
        N = global_slice[0:period].cumsum()
        total = region_slice[0:period].cumsum()
        average = N/total
        df_events["Average sig events"] = average
    return df_events

# Average number of events during the last n working days based on the total number of 
# events on a daily basis.
def compute_average_events(df, period=5):
    df_events = df.copy(deep=True)
    global_slice = df_events.iloc[:,-2:-1]
    data_slice = df_events.iloc[:,:-2]
    length = len(data_slice)
    for i in range(length):
        N = data_slice[0:period].cumsum()
        total = global_slice[0:period].cumsum()
        average = N/total
        df_events["Average events"] = average
    return df_events

def generate_datasets_bonds(df):
    data = df.copy()
    data['Classification_1D_Global'] = np.where(data['Classification_1D_U.S. 10Y'] == 1, 1, 0)
    return data.iloc[:,-1:]


def generate_datasets_currencies(df):
    data = df.copy()
    data['Classification_1D_Global'] = np.where(data['Classification_1D_EUR/USD'] & data['Classification_1D_GBP/USD'] == 1, 1, 0)
    return data.iloc[:,-1:]


def generate_datasets_crypto(df):
    data = df.copy()
    data['Classification_1D_Global'] = np.where(data['Classification_1D_Bitcoin'] == 1, 1, 0)
    return data.iloc[:,-1:]

####################
## Data selection ##
####################

# List of financial products we work with
List_Crypto = ["Bitcoin", "Ethereum", "Litecoin", "Bitcoin Cash", "Ethereum Classic", "Monero", "Dash"]
List_Currencies = ['EUR/USD', 'GBP/USD', 'CAD/USD', 'AUD/USD', 'NZD/USD', 'CNY/USD', 'DKK/USD', 'HKD/USD', 'INR/USD', 'JPY/USD', 'KRW/USD', 'MYR/USD', 'MXN/USD', 'NOK/USD', 'SEK/USD', 'CHF/USD', 'TWD/USD', 'THB/USD','XAU/USD']
# Unavailable on Investing.com : Sweden 10Y, Denmark 10Y
List_Bonds = ['France 10Y', 'U.S. 10Y', 'Japan 10Y', 'Germany 10Y', 'U.K. 10Y', 'India 10Y', 'South Korea 10Y', 'Russia 10Y', 'Spain 10Y', 'Mexico 10Y', 'Indonesia 10Y', 'Netherlands 10Y', 'Switzerland 10Y', 'Taiwan 10Y', 'Poland 10Y', 'Belgium 10Y', 'Thailand 10Y', 'Austria 10Y', 'Norway 10Y', 'Hong Kong 10Y', 'Israel 10Y', 'Philippines 10Y', 'Malaysia 10Y', 'Ireland 10Y', 'Greece 10Y', 'Czech Republic 10Y', 'Hungary 10Y']
List_Commodity = ["", "Crude Oil WTI"]

# index : "CBOE Volatility Index", 
# currency : "XAU/USD"

# Need : Libor rate, TED Spread, Effective Federal Funds Rate, High yield bond returns

# Dictionary of financial products we work with
Dict_Indices_Americas = {'united states': 'Nasdaq', 'canada': 'S&P/TSX', 'argentina': 'S&P Merval',
                         'peru': 'S&P Lima General', 'brazil': 'Bovespa', 'chile': 'S&P CLX IPSA',
                         'mexico': 'FTSE BIVA Real Time Price'}
Dict_Indices_Europe = {'germany': 'DAX', 'united kingdom': 'FTSE 100', 'france': 'CAC 40', 'russia': 'RTSI',
                       'spain': 'IBEX 35', 'netherlands': 'AEX', 'switzerland': 'SMI', 'sweden': 'OMXS30',
                       'poland': 'WIG20', 'belgium': 'BEL 20', 'austria': 'ATX', 'norway': 'Oslo OBX',
                       'israel': 'TA 35', 'denmark': 'OMXC20', 'ireland': 'ISEQ Overall', 'greece': '	FTSE/Athex 20',
                       'czech republic': 'FTSE Czech Republic', 'hungary': 'Budapest SE', 'slovakia': 'SAX'}
Dict_Indices_Asia = {'china': 'SZSE Component', 'japan': 'Nikkei 1000', 'india': 'BSE Sensex', 'south korea': 'KOSPI',
                     'indonesia': 'IDX Composite', 'taiwan': 'Taiwan Weighted', 'thailand': 'SET',
                     'hong kong': 'Hang Seng', 'philippines': 'PSEi Composite', 'malaysia': 'KLCI',
                     'pakistan': 'Karachi 100', 'australia': 'S&P/ASX 200', 'new zealand': 'NZX 50'}

# Selected period
start_date = "01/01/2001"
end_date = "01/01/2022"
lag_range = [1,2,3,4,5,20,40,60]

# Data dictionaries
dict_crypto = {crypto: get_crypto(crypto, start = start_date, end = end_date) for crypto in List_Crypto}
dict_currencies = {currencies: get_currency(currencies, start = start_date, end = end_date) for currencies in List_Currencies}
dict_bonds = {bond: get_bond(bond, start = start_date, end = end_date) for bond in List_Bonds}
# dict_commodity = {commodity: get_commodity(commodity, start_date = start, end_date = end) for commodity in List_Commodity}
# data_event = investpy.economic_calendar(importances=["high", ""], from_date=start, to_date=end).dropna()
dict_indices_americas = {index : get_indices(index, country=country, start = start_date, end = end_date) for country, index in Dict_Indices_Americas.items()}
dict_indices_europe = {index : get_indices(index, country=country, start = start_date, end = end_date) for country, index in Dict_Indices_Europe.items()}
dict_indices_asia = {index: get_indices(index, country=country, start=start_date, end=end_date) for country, index in
                     Dict_Indices_Asia.items()}


#####################
## Data processing ##
#####################

# What needs to be done :
# 1. Consider Stock Indices, Bond Yields, and Currency Exchange Rates

# Get currencies data
df_currencies = product_family(dict_currencies, List_Currencies)

# Get bonds data
df_crypto = product_family(dict_crypto, List_Crypto)

# Get bonds data
df_bonds = product_family(dict_bonds, List_Bonds)

# Get indices data
df_indices_americas = product_family(dict_indices_americas, Dict_Indices_Americas.values())
df_indices_europe = product_family(dict_indices_europe, Dict_Indices_Europe.values())
df_indices_asia = product_family(dict_indices_asia, Dict_Indices_Asia.values())

# Identify missing data
# identify_data(df_currencies)
# identify_data(df_bonds)
# identify_data(df_indices_americas)
# identify_data(df_indices_europe)
# identify_data(df_indices_asia)

# Clean data
df_currencies = clean_data(df_currencies).dropna()
df_bonds = clean_data(df_bonds).dropna()
df_crypto = clean_data(df_crypto).dropna()
df_indices_americas = clean_data(df_indices_americas).dropna()
df_indices_europe = clean_data(df_indices_europe).dropna()
df_indices_asia = clean_data(df_indices_asia).dropna()


# In[16]:


df_currencies.to_csv(r'~/path/df_currencies.csv', index = True)
df_bonds.to_csv(r'~/path/df_bonds.csv', index = True)
df_crypto.to_csv(r'~/path/df_crypto.csv', index = True)
df_indices_americas.to_csv(r'~/path/df_indices_americas.csv', index = True)
df_indices_europe.to_csv(r'~/path/df_indices_europe.csv', index = True)
df_indices_asia.to_csv(r'~/path/df_indices_asia.csv', index = True)


# 2. Derive the continuously compounded rate of daily returns, by calculating the log-returns [ln]
#Calculate returns
df_currencies_ret = compute_returns(df_currencies)
df_bonds_ret = compute_returns(df_bonds)
df_crypto_ret = compute_returnse(df_crypto)
df_indices_americas_ret = compute_returns(df_indices_americas)
df_indices_europe_ret = compute_returns(df_indices_europe)
df_indices_asia_ret = compute_returns(df_indices_asia)

# 3. Calculate the daily volatility of return [ln2], by squaring the log-returns.
#Calculate volatilities
df_currencies_vol = compute_vol(df_currencies)
df_bonds_vol = compute_vol(df_bonds)
df_crypto_vol = compute_returnse(df_vol)
df_indices_americas_vol = compute_vol(df_indices_americas)
df_indices_europe_vol = compute_vol(df_indices_europe)
df_indices_asia_vol = compute_vol(df_indices_asia)


# 4. Calculate lagged variables on a daily basis for each crisis indicator (regional/global),
#    starting from 1 up to 5 days, 20 days, 40 days, and 60 days [lag1, lag2,..., lag5, lag20, lag40, lag60].

# Get lagged values
# Horizons : 1 up to 5, 20, 40, 60 days
df_currencies_L5D = get_lagged_data(df_currencies, period=5)
df_bonds_L5D = get_lagged_data(df_bonds, period=5)
df_crypto_L5D = get_lagged_data(df_crypto, period=5)
df_indices_americas_L5D = get_lagged_data(df_indices_americas, period=5)
df_indices_europe_L5D = get_lagged_data(df_indices_europe, period=5)
df_indices_asia_L5D = get_lagged_data(df_indices_asia, period=5)

df_currencies_L20D = get_lagged_data(df_currencies, period=20)
df_bonds_L20D = get_lagged_data(df_bonds, period=20)
df_crypto_L20D = get_lagged_data(df_crypto, period=20)
df_indices_americas_L20D = get_lagged_data(df_indices_americas, period=20)
df_indices_europe_L20D = get_lagged_data(df_indices_europe, period=20)
df_indices_asia_L20D = get_lagged_data(df_indices_asia, period=20)

# Clean lagged data
df_currencies_L5D = clean_data(df_currencies_L5D)
df_bonds_L5D = clean_data(df_bonds_L5D)
df_crypto_L5D = clean_data(df_crypto_L5D)
df_indices_americas_L5D = clean_data(df_indices_americas_L5D)
df_indices_europe_L5D = clean_data(df_indices_europe_L5D)
df_indices_asia_L5D = clean_data(df_indices_asia_L5D)

df_currencies_L20D = clean_data(df_currencies_L20D)
df_bonds_L20D = clean_data(df_bonds_L20D)
df_crypto_L20D = clean_data(df_crypto_L20D)
df_indices_americas_L20D = clean_data(df_indices_americas_L20D)
df_indices_europe_L20D = clean_data(df_indices_europe_L20D)
df_indices_asia_L20D = clean_data(df_indices_asia_L20D)

# Identify crisis events
currencies_events = identify_crisis_event(df_currencies)
bonds_events = identify_crisis_event(df_bonds)
crypto_events = identify_crisis_event(df_crypto)
indices_americas_events = identify_crisis_event(df_indices_americas)
indices_europe_events = identify_crisis_event(df_indices_europe)
indices_asia_events = identify_crisis_event(df_indices_asia)
# Identify crisis events (5 days lag)
currencies_events_L5D = identify_crisis_event(df_currencies_L5D)
bonds_events_L5D = identify_crisis_event(df_bonds_L5D)
crypto_events_L5D = identify_crisis_event(df_crypto_L5D)
indices_americas_events_L5D = identify_crisis_event(df_indices_americas_L5D)
indices_europe_events_L5D = identify_crisis_event(df_indices_europe_L5D)
indices_asia_events_L5D = identify_crisis_event(df_indices_asia_L5D)
# Identify crisis events (20 days lag)
currencies_events_L20D = identify_crisis_event(df_currencies_L20D)
bonds_events_L20D = identify_crisis_event(df_bonds_L20D)
crypto_events_L20D = identify_crisis_event(df_crypto_L20D)
indices_americas_events_L20D = identify_crisis_event(df_indices_americas_L20D)
indices_europe_events_L20D = identify_crisis_event(df_indices_europe_L20D)
indices_asia_events_L20D = identify_crisis_event(df_indices_asia_L20D)

# Compute crisis events for bonds and currencies and crypto
currencies_events = compute_crisis_events_bc(currencies_events)
bonds_events = compute_crisis_events_bc(bonds_events)
crypto_events = compute_crisis_events_bc(crypto_events)

# Compute crisis events for bonds and currencies (5 days lag)
currencies_events_L5D = compute_crisis_events_bc(currencies_events_L5D)
bonds_events_L5D = compute_crisis_events_bc(bonds_events_L5D)
crypto_events_L5D = compute_crisis_events_bc(crypto_events_L5D)

# Compute crisis events for bonds and currencies (20 days lag)
currencies_events_L20D = compute_crisis_events_bc(currencies_events_L20D)
bonds_events_L20D = compute_crisis_events_bc(bonds_events_L20D)
crypto_events_L20D = compute_crisis_events_bc(crypto_events_L20D)

# Compute crisis events per region
americas_events = compute_crisis_events_region(indices_americas_events, "Americas")
europe_events = compute_crisis_events_region(indices_europe_events, "Europe")
asia_events = compute_crisis_events_region(indices_asia_events, "Asia")

# Compute crisis events per region (5 days lag)
americas_events_L5D = compute_crisis_events_region(indices_americas_events_L5D, "Americas")
europe_events_L5D = compute_crisis_events_region(indices_europe_events_L5D, "Europe")
asia_events_L5D = compute_crisis_events_region(indices_asia_events_L5D, "Asia")

# Compute crisis events per region (20 days lag)
americas_events_L20D = compute_crisis_events_region(indices_americas_events_L20D, "Americas")
europe_events_L20D = compute_crisis_events_region(indices_europe_events_L20D, "Europe")
asia_events_L20D = compute_crisis_events_region(indices_asia_events_L20D, "Asia")

# Compute global crisis events
global_events = compute_crisis_events_global(americas_events, europe_events, asia_events)

# Compute global crisis events (5 days lag)
global_events_L5D = compute_crisis_events_global(americas_events_L5D, europe_events_L5D, asia_events_L5D)

# Compute global crisis events (20 days lag)
global_events_L20D = compute_crisis_events_global(americas_events_L20D, europe_events_L20D, asia_events_L20D)

# Concat all classification events
cross_product_event = pd.concat([currencies_events, bonds_events, crypto_events, americas_events, europe_events, asia_events, global_events], axis=1)
cross_product_event_L5D = pd.concat([currencies_events_L5D, bonds_events_L5D, crypto_events_L5D, americas_events_L5D, europe_events_L5D, asia_events_L5D, global_events_L5D], axis=1)
cross_product_event_L20D = pd.concat([currencies_events_L20D, bonds_events_L20D, crypto_events_L20D, americas_events_L20D, europe_events_L20D, asia_events_L20D, global_events_L20D], axis=1)

# Average number of events during the last 5 working days and the last 20 days [L5D, L20D],
# based on the total number of events on a daily basis.
average_sig_events = compute_average_sig_events(cross_product_event).dropna()
average_sig_events_L20D = compute_average_sig_events(cross_product_event_L20D, 20).dropna()

# 5. Compute the average number of significant events during the last 5 working days and the last 20 days
#    [L5D, L20D], based on previous values of the binary predictor variable (regional/global) which identifies
#    whether the number of events exceeded a threshold someday (as described previously).
average_events = compute_average_events(cross_product_event).dropna()
average_events_L20D = compute_average_events(cross_product_event_L20D).dropna()
final_cross_product_event = pd.concat([cross_product_event, average_events, average_sig_events], axis=1).dropna()


## Datasets for ML models
# Create various datatframes by relevant metric
# Bonds Dataset + Classification
b_ev = bonds_events.dropna()
bonds_dataset = pd.concat([compute_returns(df_bonds), compute_vol(df_bonds)], axis = 1).dropna()

# Currencies Dataset + Classification
cu_ev = currencies_events.dropna()
currencies_dataset = pd.concat([compute_returns(df_currencies), compute_vol(df_currencies)], axis = 1).dropna()

# Crypto Dataset + Classification
cr_ev = crypto_events.dropna()
crypto_dataset = pd.concat([compute_returns(df_crypto), compute_vol(df_crypto)], axis = 1).dropna()

# Indices Americas Dataset + Classification
am_idx_ev = americas_events.dropna()
americas_indices_dataset = pd.concat([compute_returns(df_indices_americas), compute_vol(df_indices_americas), am_idx_ev], axis = 1).dropna()
americas_indices_dataset.to_csv(r'~/path/americas_indices_dataset.csv', index = True)

# Indices Europe Dataset + Classification
eu_idx_ev = europe_events.dropna()
europe_indices_dataset = pd.concat([compute_returns(df_indices_europe), compute_vol(df_indices_europe), eu_idx_ev], axis = 1).dropna()
europe_indices_dataset.to_csv(r'~/path/europe_indices_dataset.csv', index = True)

# Indices Asia Dataset + Classification
as_idx_ev = asia_events.dropna()
asia_indices_dataset = pd.concat([compute_returns(df_indices_asia), compute_vol(df_indices_asia), as_idx_ev], axis = 1).dropna()
asia_indices_dataset.to_csv(r'~/path/asia_indices_dataset.csv', index = True)

# Indices Dataset + Global Classification
gl_idx_ev = global_events.dropna()
indices_dataset = pd.concat([compute_returns(df_indices_americas), compute_vol(df_indices_americas), compute_returns(df_indices_europe), compute_vol(df_indices_europe),  compute_returns(df_indices_asia), compute_vol(df_indices_asia), gl_idx_ev], axis = 1).dropna()
indices_dataset.to_csv(r'~/path/indices_dataset.csv', index = True)

# Bonds dataset
b_ev = generate_datasets_bonds(b_ev).iloc[:,-1].dropna()
bonds_dataset = pd.concat([bonds_dataset, b_ev], axis = 1).dropna()
bonds_dataset.to_csv(r'~/path/bonds_dataset.csv', index = True)

# Currencies dataset
cu_ev = generate_datasets_currencies(cu_ev).iloc[:,-1].dropna()
currencies_dataset = pd.concat([currencies_dataset, cu_ev], axis = 1).dropna()
currencies_dataset.to_csv(r'~/path/currencies_dataset.csv', index = True)


# Crypto dataset
cr_ev = generate_datasets_crypto(cr_ev).iloc[:,-1].dropna()
crypto_dataset = pd.concat([crypto_dataset, cr_ev], axis = 1).dropna()
crypto_dataset.to_csv(r'~/path/crypto_dataset.csv', index = True)
