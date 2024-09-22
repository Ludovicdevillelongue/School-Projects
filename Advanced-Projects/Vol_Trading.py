# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:15:36 2022

@author: ludov
"""
from math import *
from ProbaTools import *
from datetime import *
import pandas as pd
import matplotlib.pyplot as plt
from FinancialObjects import *
from Applications import *
import numpy as np
import math
from pathlib import Path  

name_file = 'implied_vol.xlsx'

#Liste des stocks et indices souhaités
liste = ['NDX Index', 'AMZN US Equity', 'CAC Index', 'SPX Index']
#Liste des indices souhaités
liste_index=['NDX Index', 'SPX Index']
#Liste de stocks souhaités
liste_stock=['AMZN US Equity']
#Fenêtre de volatilité
in_sample_histo = 40
in_sample_z = 60
#Trigger z-score
trigger_long_z_score = -0.75
trigger_short_z_score = 1.5
#Trigger spread
trigger_long_spread = 0
trigger_short_spread =5




'''
-----------------------------------------
---Z-Scores Volatility Option Strategy---
-----------------------------------------
'''    
    

def Z_score_option_strategy():

    f2, axes2 = plt.subplots(figsize=(10,3.75))
    
    #Collection des données par indice et ou stock
    dataframe_collection = {} 
    for idx in liste:

        xls = pd.ExcelFile(name_file)
        market_data = pd.read_excel(xls, idx)

        in_sample_z = 60
        in_sample_histo = 40
        trigger_long_z_score = -0.75
        trigger_short_z_score = 1.5

        
        #Calcul du close, des log ret et de la vol
        market_data['Close'] = market_data['BLOOMBERG_CLOSE_PRICE']
        market_data['logret'] = np.log(market_data.Close/market_data.Close.shift(1))
        market_data['vol_hist' + str(in_sample_histo)] = market_data.logret.rolling(in_sample_histo).std() * np.sqrt(252)

        #Calcul des z-scores
        mean_vec_impli = market_data['12MO_PUT_IMP_VOL'].rolling(in_sample_z).mean()
        std_vec_impli = market_data['12MO_PUT_IMP_VOL'].rolling(in_sample_z).std()
        market_data['z_score_impli'] = (market_data['12MO_PUT_IMP_VOL'] - mean_vec_impli) / std_vec_impli

        mean_vec_histo = market_data['vol_hist' + str(in_sample_histo)].rolling(in_sample_z).mean()
        std_vec_histo =  market_data['vol_hist' + str(in_sample_histo)].rolling(in_sample_z).std()
        market_data['z_score_histo'] = (market_data['vol_hist' + str(in_sample_histo)] - mean_vec_histo) / std_vec_histo

        #Trigger en fonction des z-scores
        for col in market_data.columns:
            market_data = market_data[~pd.isnull(market_data[col])]

        market_data['bool_long_impli'] = market_data['z_score_impli'] <= trigger_long_z_score
        market_data['bool_short_impli'] = market_data['z_score_impli'] >= trigger_short_z_score
        market_data['bool_long_histo'] = market_data['z_score_histo'] <= trigger_long_z_score
        market_data['bool_short_histo'] = market_data['z_score_histo'] >= trigger_short_z_score
        
        #Création des positions
        bool_long = []
        bool_short =[]
        for i in range(0,len(market_data['bool_long_impli'])):
            if market_data['bool_long_impli'].iloc[i] == True and market_data['bool_long_histo'].iloc[i] == True:
                bool_long.append(True)
            else:
                bool_long.append(False)
            if market_data['bool_short_impli'].iloc[i] == True and market_data['bool_short_histo'].iloc[i] == True:
                bool_short.append(True)
            else:
                bool_short.append(False)

        market_data['bool_long'] = bool_long
        market_data['bool_short'] = bool_short

        f1, axes1 = plt.subplots(2, figsize=(10,7))
        #Représentation des volatilités
        axes1[0].plot(market_data['Dates'], market_data['vol_hist' + str(in_sample_histo)]*100, label = f'vol histo40 {idx}')
        axes1[0].plot(market_data['Dates'],market_data['12MO_PUT_IMP_VOL'], label = f'vol impli252 {idx}')
        axes1[0].set_title('Volatilities')
        axes1[0].legend(loc='upper left')
        #Représentation des z-scores
        axes1[1].plot(market_data['Dates'],market_data['z_score_impli'], label = f'z_score_impli {idx}')
        axes1[1].plot(market_data['Dates'],market_data['z_score_histo'], label = f'z_score_histo {idx}')
        axes1[1].plot(market_data['Dates'],market_data['bool_long'], label = f'bool_long {idx}')
        axes1[1].legend(loc='upper left')


        #Paramètres de trading à initialiser
        maturity = 1
        stop_loss = -0.75
        pnl_Long= 0.0
        #pnl_Short = 0.0
        bool_long = False
        #bool_short = False
        strat_days = 0
        pnL_Strat = 0.0
        bool_long_liste = []
        pnl_Long_liste = []
        #bool_short_liste = []
        #pnl_Short_liste = []
        pay_off_T = []
        pnl_date = []
        pnl_temp = []

        #Création de la stratégie
        for i in range(len(market_data)-2):
            #Prise de position long
            if market_data['bool_long'].iloc[i] and bool_long == False:
                bool_long = True
                bool_short = False
                strat_days = 0
                pnL_Strat = 0
                strike = market_data.Close.iloc[i]
                pricing_vol = market_data['12MO_PUT_IMP_VOL'].iloc[i] / 100
                my_udl = Underlying(market_data.Close.iloc[i], idx)
                my_call = EuropeanCall(strike, my_udl, maturity)
                delta = my_call.bs_delta(pricing_vol, 0)
                vega = my_call.bs_vega(pricing_vol, 0)
                call_price = my_call.bs_price(pricing_vol, 0)
                
                old_price = call_price
            #Calcul du P&L
            if bool_long and strat_days + 1 <= (round(maturity * 252)): 
                pnL_Strat = 0       
                my_udl = Underlying(market_data.Close.iloc[i+1], "BNP")
                my_call = EuropeanCall(strike, my_udl, maturity - (strat_days + 1)/252)
                call_price = my_call.bs_price(market_data['12MO_PUT_IMP_VOL'].iloc[i+1]/100, 0)
                pnL_Strat = pnL_Strat + (call_price - old_price - delta * (market_data.Close.iloc[i+1] - market_data.Close.iloc[i])) / vega
                pnl_Long = pnl_Long + (call_price - old_price - delta * (market_data.Close.iloc[i+1] - market_data.Close.iloc[i])) / vega
                old_price = call_price
                delta = my_call.bs_delta(market_data['12MO_PUT_IMP_VOL'].iloc[i] / 100, 0)
                vega = my_call.bs_vega(market_data['12MO_PUT_IMP_VOL'].iloc[i] / 100, 0)
                strat_days = strat_days + 1

                #Payoff de l'option ajouté au P&L
                if strat_days == round(maturity * 252):
                    pnL_Strat = pnL_Strat + (np.max(market_data.Close.iloc[i+1] - strike,0)) / vega
                    pnl_Long = pnl_Long + (np.max(market_data.Close.iloc[i+1] - strike,0)) / vega
                

                pnl_temp.append(pnL_Strat)

            #Conditions de sortie de position
            if  strat_days + 1 == (round(maturity * 252) or pnL_Strat <= stop_loss or market_data['bool_short'].iloc[i]) and (bool_long):
                bool_long = False
                #bool_short = True
            
            bool_long_liste.append(bool_long)
            pnl_Long_liste.append(pnl_Long)
            #bool_short_liste.append(bool_short)
            #pnl_Short_liste.append(pnl_Short)
            pnl_date.append(market_data.Dates.iloc[i+1])

        #Recapitulatif des prises de position et du P&L cumulatif
        a = pd.DataFrame({'bool_long':bool_long_liste, 'pnl_long':pnl_Long_liste, 'pnl_dates':pnl_date})

        axes2.plot(a.pnl_dates,a.pnl_long, label = idx)
        axes2.legend(loc='upper left')
        axes2.set_title('P&L Z-SCORE STRATEGY')
        dataframe_collection[idx]=a
    plt.show()
    return dataframe_collection

#Dictionnaire des p&L par stock et ou indice
strategy_1=Z_score_option_strategy()


'''
-----------------------------------------
---------Spread Option Strategy----------
-----------------------------------------
'''    



def Spread_option_strategy():

    f2, axes2 = plt.subplots(figsize=(10,3.75))
    
    #Collection des données par indice et ou stock
    dataframe_collection={}
    stk=liste_stock[0]
    for idx in liste_index:

        xls = pd.ExcelFile(name_file)
        market_data_bench = pd.read_excel(xls, idx)
        market_data_stock=pd.read_excel(xls, stk)
        #Fixation des fenêtres et triggers
        in_sample_z = 60
        in_sample_histo = 40
        trigger_long_spread = 0
        trigger_short_spread =5


        market_data_stock['Close'] = market_data_stock['BLOOMBERG_CLOSE_PRICE']
        market_data_stock['logret'] = np.log(market_data_stock.Close / market_data_stock.Close.shift(1))
        
        market_data_bench['Close'] = market_data_bench['BLOOMBERG_CLOSE_PRICE']
        market_data_bench['logret'] = np.log(market_data_bench.Close / market_data_bench.Close.shift(1))
        
        #Calcul du spread des volatilités implicites
        market_data_spread=pd.DataFrame()
        market_data_spread['Dates']=market_data_bench.Dates
        market_data_spread['vol_impli'] = market_data_stock['12MO_PUT_IMP_VOL']-market_data_bench['12MO_PUT_IMP_VOL']
        
        for col in market_data_spread.columns:
            market_data_spread = market_data_spread[~pd.isnull(market_data_spread[col])]

        #Trigger en fonction des spreads
        market_data_spread['bool_long_impli'] = market_data_spread['vol_impli']<= trigger_long_spread
        market_data_spread['bool_short_impli'] = market_data_spread['vol_impli'] >= trigger_short_spread



        #Création des positions
        bool_long = []
        bool_short = []
        for i in range(0, len(market_data_spread['bool_long_impli'])):
            if market_data_spread['bool_long_impli'].iloc[i] == True :
                bool_long.append(True)
            else:
                bool_long.append(False)
            if market_data_spread['bool_short_impli'].iloc[i] == True :
                bool_short.append(True)
            else:
                bool_short.append(False)


        
        market_data_spread['bool_long'] = bool_long
        market_data_spread['bool_short'] = bool_short


        f1, axes = plt.subplots(2, figsize=(12.5, 15))
        #Représentation du spread de volatilité
        axes[0].plot(market_data_spread['Dates'], market_data_spread['vol_impli'], label=f'Spread_vol_impli{stk}-{idx}')
        axes[0].set_title('Volatilities')
        axes[0].legend(loc='upper left')
        #Représentation des positions longues sur le spread
        axes[1].plot(market_data_spread['Dates'], market_data_spread['bool_long'], label='bool_long')
        axes[1].legend(loc='upper left')
        
        #Paramètres de trading à initialiser
        maturity = 1
        stop_loss = -1
        pnl_Long = 0.0
        bool_long = False
        strat_days = 0
        pnL_Strat = 0
        pnL_Strat_stock=0
        pnL_Strat_index=0
        pnl_Long_stock=0
        pnl_Long_index=0
        bool_long_liste = []
        pnl_Long_liste = []
        pay_off_T = []
        pnl_date = []
        pnl_temp = []

        #Création de la stratégie
        for i in range(len(market_data_spread) - 1):
            #Prise de position long
            if market_data_spread['bool_long'].iloc[i] and bool_long == False:
                bool_long = True
                bool_short = False
                strat_days = 0
                pnL_Strat = 0
                pnL_Strat_stock=0
                pnL_Strat_index=0
                strike_stock = market_data_stock.Close.iloc[i]
                strike_index = market_data_bench.Close.iloc[i]
                pricing_vol_stock = market_data_stock['12MO_PUT_IMP_VOL'].iloc[i] / 100
                pricing_vol_index = market_data_bench['12MO_PUT_IMP_VOL'].iloc[i] / 100
                my_stock = Underlying(market_data_stock.Close.iloc[i], stk)
                my_index = Underlying(market_data_bench.Close.iloc[i], idx)
                my_call_stock = EuropeanCall(strike_stock, my_stock, maturity)
                my_call_index = EuropeanCall(strike_index, my_index, maturity)
                delta_stock = my_call_stock.bs_delta(pricing_vol_stock, 0)
                delta_index = my_call_index.bs_delta(pricing_vol_index, 0)
                vega_stock = my_call_stock.bs_vega(pricing_vol_stock, 0)
                vega_index = my_call_index.bs_vega(pricing_vol_index, 0)
                call_price_stock = my_call_stock.bs_price(pricing_vol_stock, 0)
                call_price_index = my_call_index.bs_price(pricing_vol_index, 0)
                old_price_stock = call_price_stock
                old_price_index = call_price_index

                
            #Calcul du P&L
            if bool_long and strat_days + 1 <= (round(maturity * 252)): 
                pnL_Strat = 0
                pnL_Strat_stock=0
                pnL_Strat_index=0
                my_stock = Underlying(market_data_stock.Close.iloc[i + 1], stk)
                my_index = Underlying(market_data_bench.Close.iloc[i + 1], idx)
                my_call_stock = EuropeanCall(strike_stock, my_stock, maturity - (strat_days + 1) / 252)
                my_call_index = EuropeanCall(strike_index, my_index, maturity - (strat_days + 1) / 252)
                call_price_stock = my_call_stock.bs_price(market_data_stock['12MO_PUT_IMP_VOL'].iloc[i + 1] / 100, 0)
                call_price_index = my_call_index.bs_price(market_data_bench['12MO_PUT_IMP_VOL'].iloc[i + 1] / 100, 0)
                pnl_stock =(call_price_stock - old_price_stock - delta_stock * (market_data_stock.Close.iloc[i + 1] - market_data_stock.Close.iloc[i])) / vega_stock
                pnl_index = (call_price_index - old_price_index - delta_index * ( market_data_bench.Close.iloc[i + 1] - market_data_bench.Close.iloc[i])) / vega_index
                pnL_Strat = pnL_Strat + pnl_stock - pnl_index
                pnl_Long = pnl_Long + pnl_stock - pnl_index
                old_price_stock = call_price_stock
                old_price_index = call_price_index
                delta_stock = my_call_stock.bs_delta(market_data_stock ['12MO_PUT_IMP_VOL'].iloc[i] / 100, 0)
                delta_index = my_call_index.bs_delta(market_data_bench['12MO_PUT_IMP_VOL'].iloc[i] / 100, 0)
                vega_stock = my_call_stock.bs_vega(market_data_stock ['12MO_PUT_IMP_VOL'].iloc[i] / 100, 0)
                vega_index = my_call_index.bs_vega(market_data_bench['12MO_PUT_IMP_VOL'].iloc[i] / 100, 0)
                strat_days = strat_days + 1
        
        
                #Payoff de l'option ajouté au P&L
                if strat_days == round(maturity * 252):
                    pnL_Strat_stock = pnL_Strat_stock + (np.max(market_data_stock.Close.iloc[i + 1] - strike_stock, 0)) / vega_stock
                    pnL_Strat_index = pnL_Strat_index + (np.max(market_data_bench.Close.iloc[i + 1] - strike_index, 0)) / vega_index
                    pnl_Long_stock = pnl_Long_stock + (np.max(market_data_stock.Close.iloc[i + 1] - strike_stock, 0)) / vega_stock
                    pnl_Long_index = pnl_Long + (np.max(market_data_bench.Close.iloc[i + 1] - strike_index, 0)) / vega_index
                    pnL_Strat =pnL_Strat+ pnL_Strat_stock - pnL_Strat_index
                    pnl_Long = pnl_Long+pnl_Long_stock - pnl_Long_index
                pnl_temp.append(pnL_Strat)
        
        
            #Conditions de sortie de position
            if strat_days + 1 == (round(maturity * 252) or pnL_Strat <= stop_loss or market_data_spread['bool_short'].iloc[i]) and (bool_long):
                bool_long = False
                # bool_short = True
        
            bool_long_liste.append(bool_long)
            pnl_Long_liste.append(pnl_Long)
            # bool_short_liste.append(bool_short)
            # pnl_Short_liste.append(pnl_Short)
            pnl_date.append(market_data_spread.Dates.iloc[i + 1])
        
        
        #Recapitulatif des prises de position et du P&L cumulatif
        recap = pd.DataFrame({'bool_long': bool_long_liste, 'pnl_long': pnl_Long_liste, 'pnl_dates': pnl_date})
        
        axes2.plot(recap.pnl_dates, recap["pnl_long"], label=f'{stk}-{idx}')
        axes2.legend(loc='upper left')
        axes2.set_title('P&L SPREAD STRATEGY')
        dataframe_collection[idx]=recap
    plt.show()
    return dataframe_collection
    
#Dictionnaire des p&L par stock et ou indice
strategy_2=Spread_option_strategy()



'''
-----------------------------------------
-------Bollinger Band Strategy-----------
-----------------------------------------
'''

#Création des bandes de Bollinger en utilisant la volatilité historique comme référence
    # Etre long d'une position est généralement profitable lorsque la volatilité historique
    # est supérieure à la volatilité implicite.La volatilité historique est pour cela 
    # associée à la bande basse. (si vol impli passe en dessous de vol histo, signal d'achat')
    # La volatilité implicite a tendance à être supérieure à la volatilité historique.
    # Afin de lutter contre ce biais, la bande haute est égale à 2 fois la volatilité historique


def bb(data):
    data['logret'] = np.log(data.close/data.close.shift(1))
    data["vol_histo"]=data.logret.rolling(in_sample_histo).std() * np.sqrt(252)*100
    data["upper_bb"] = 2*data['vol_histo']
    data["lower_bb"] = data['vol_histo']
    return data

#Achat ou vente si la volatilité implicite est respectivement au dessus ou en dessous des bandes
def implement_strategy(data, lower_bb, upper_bb):
    buy_price = np.empty(len(data))*np.nan
    sell_price = np.empty(len(data))*np.nan
    bb_signal = np.zeros(len(data))
    signal = 0
    for i in range(len(data)):
        if data.iloc[i-1] > lower_bb.iloc[i-1] and data.iloc[i] < lower_bb.iloc[i]:
            if signal != 1:
                buy_price[i]=data[i]
                signal = 1
                bb_signal[i]=signal
        elif data.iloc[i-1] < upper_bb.iloc[i-1] and data.iloc[i] > upper_bb.iloc[i]:
            if signal != -1:
                sell_price[i]=data[i]
                signal = -1
                bb_signal[i]=signal
    return buy_price, sell_price, bb_signal



def position(data, signal, close,indicator, upper_band, lower_band):
    #Création des positions
    position = np.zeros(len(close))
    
    for i in range(len(close)):
        if signal[i] == 1:
            position[i] = 1
        elif signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
    
    #Réunion en une dataframe pour vérification
    signal = pd.DataFrame(signal).rename(columns = {0:'signal'}).set_index(data.index)
    position = pd.DataFrame(position).rename(columns = {0:'position'}).set_index(data.index)
    strategy = pd.concat([indicator, upper_band,lower_band , signal, position]\
                         ,join = 'inner', axis = 1).set_index(data.index)
    
    return strategy


def backtest(data,strategy, name):
    #Backtest de la stratégie en utilisant les log return des stocks à la place des options
    #Calcul des rendements de la stratégie
    bb_strategy_ret = []
    for i in range(len(data)):
            returns = data["logret"].iloc[i]*strategy['position'].iloc[i]
            bb_strategy_ret.append(returns)
    
    strategy["returns"] = bb_strategy_ret


    #Calcul des rendements de la stratégie pour un investissement initial de 100k
    investment_value = 100000
    number_of_stocks = math.floor(investment_value/data['close'].iloc[-1])
    bb_investment_ret = []
    pnl_list=[]
    pnl=0
    #Calcul des rendements journaliers
    for i in range(len(strategy)):
        returns = number_of_stocks*strategy["returns"].iloc[i]
        bb_investment_ret.append(returns)
        pnl=  pnl+ returns
        pnl_list.append(pnl)
    strategy["investment_returns"] = bb_investment_ret
    strategy["P&L_"+name]=pnl_list
    return strategy


def bollinger_band_strategy():
    
    #Collection des données par indice et ou stock
    dataframe_collection={}
    f2, axes2 = plt.subplots(figsize=(10,3.75))
    for idx in liste:
        
        #Import des données
        xls = pd.ExcelFile(name_file)
        data = pd.read_excel(xls, idx)
        data = data[['Dates','BLOOMBERG_CLOSE_PRICE', '12MO_PUT_IMP_VOL']]
        data = data.rename(columns={'Dates': 'date','BLOOMBERG_CLOSE_PRICE':'close',\
                                          '12MO_PUT_IMP_VOL':'vol_impli'})
        
        #Récupération des signaux
        data = bb(data)
        data=data.dropna().set_index("date")
        buy_price, sell_price, bb_signal = \
            implement_strategy(data['vol_impli'], data['lower_bb'], data['upper_bb'])
        
        f1, axes1 = plt.subplots(figsize=(10,7))
        #Création d'un graphique pour visualiser les sorties des bandes
        axes1.plot(data.index,data['vol_impli'], label = 'IMPLIED VOLATILITY LEVEL', alpha = 0.3)
        axes1.plot(data.index,data['upper_bb'],label = 'UPPER BB', linestyle = '--', linewidth = 1, color = 'black')
        axes1.plot(data.index,data['vol_histo'], label = 'MIDDLE BB', linestyle = '--', linewidth = 1.2, color = 'grey')
        axes1.plot(data.index,data['lower_bb'],label = 'HISTORICAL VOLATILITY LEVEL / LOWER BB', linestyle = '--', linewidth = 1, color = 'black')
        axes1.scatter(data.index, buy_price, marker = '^', color = 'green', label = 'BUY', s = 200)
        axes1.scatter(data.index, sell_price, marker = 'v', color = 'red', label = 'SELL', s = 200)
        axes1.set_title(f'Asset BB STRATEGY TRADING SIGNALS {idx}')
        plt.legend(loc = 'upper left')

        #Recapitulatif des positions
        strategy_bb=position(data,bb_signal, data['close'],data['vol_impli'],data['upper_bb'],data['lower_bb'])
        #P&L de la strategie
        recap_strat_bb=backtest(data,strategy_bb, "BB")
        #Graphique du P&L cumulatif
        axes2.plot(recap_strat_bb.index,recap_strat_bb["P&L_BB"], label =idx)
        axes2.legend(loc='upper left')
        axes2.set_title('P&L BB STRATEGY')
        dataframe_collection[idx]=recap_strat_bb
    plt.show()
    return dataframe_collection

#Dictionnaire des p&L par stock et ou indice
strategy_3=bollinger_band_strategy()

'''
-----------------------------------------
-------------VAMA Strategy---------------
-----------------------------------------
'''

def vama_bands(data, lookback_volatility_short, lookback_volatility_long,lookback_volatility,std_multiplier):
    #Calcul des volatilités short et long
    std_short = list(data["close"].rolling(window = lookback_volatility_short).std())
    std_long = list(data["close"].rolling(window = lookback_volatility_long).std())
    #Calcul de l'alpha
    alpha=[]
    for i in range(len(data)):
        alpha_daily= 0.2 * (std_short[i] /std_long[i])
        alpha.append(alpha_daily)
    data["alpha"] = alpha
    data=data.dropna()
    
    vama=[]
    #Calcul du premier VAMA
    vama_init=(data["alpha"].iloc[1] * data["close"].iloc[1]) + ((1 -  data["alpha"].iloc[1])*data["close"].iloc[0]) 
    vama.append(vama_init)
    #Calcul des VAMA
    for i in range(1,len(data)):
        vama_daily = (data["alpha"].iloc[i] * data["close"].iloc[i]) + ((1 - data["alpha"].iloc[i]) * vama[i-1]) 
        vama.append(vama_daily)
    data.insert(4,"vama", vama)
    data=data.dropna()
    
    #Calcul de la volatilité utilisée comme référence
    std_ref=list(data["close"].rolling(window = lookback_volatility).std())
    data["std"]=std_ref
    data=data.dropna()
    data["upper_vama_band"] = data["vama"] + std_multiplier * data["std"]
    data["lower_vama_band"] = data["vama"] - std_multiplier * data["std"]
    return data

def vama_strategy():
    
    #Collection des données par indice et ou stock
    dataframe_collection={}
    f2, axes2 = plt.subplots(figsize=(10,3.75))
    for idx in liste:
        
        #Import des données
        xls = pd.ExcelFile(name_file)
        data = pd.read_excel(xls, idx)
        data = data[['Dates','BLOOMBERG_CLOSE_PRICE', '12MO_PUT_IMP_VOL']]
        data = data.rename(columns={'Dates': 'date','BLOOMBERG_CLOSE_PRICE':'close',\
                                          '12MO_PUT_IMP_VOL':'vol_impli'})
        data['logret'] = np.log(data.close/data.close.shift(1))
        
        #Calcul du vama 
        lookback_volatility = 40
        lookback_volatility_short = 3
        lookback_volatility_long  = 60
        std_multiplier=1.5
        data=vama_bands(data, lookback_volatility_short,lookback_volatility_long,lookback_volatility,std_multiplier)
        data=data.dropna().set_index("date")
        #Achat ou vente si le close est respectivement au dessus ou en dessous des bandes
        buy_price_vama, sell_price_vama, signal_vama\
            = implement_strategy(data['close'], data['lower_vama_band'], data['upper_vama_band'])
        
        f1, axes1 = plt.subplots(figsize=(10,7))
        #Création d'un graphique pour visualiser les sorties des bandes
        axes1.plot(data.index,data['close'], label = 'CLOSE LEVEL', alpha = 0.3)
        axes1.plot(data.index,data['upper_vama_band'],label = 'UPPER VAMA', linestyle = '--', linewidth = 1, color = 'black')
        axes1.plot(data.index,data['lower_vama_band'],label = ' LOWER VAMA', linestyle = '--', linewidth = 1, color = 'black')
        axes1.plot(data.index,data['vama'],label = ' VAMA', linestyle = '--', linewidth = 1, color = 'grey')
        axes1.scatter(data.index, buy_price_vama, marker = '^', color = 'green', label = 'BUY', s = 200)
        axes1.scatter(data.index, sell_price_vama, marker = 'v', color = 'red', label = 'SELL', s = 200)
        axes1.set_title(f'Asset VAMA STRATEGY TRADING SIGNALS {idx}')
        axes1.legend(loc = 'upper left')
        
        #Recapitulatif des positions
        strategy_vama=position(data, signal_vama, data['close'],data['close'],data['upper_vama_band'],data['lower_vama_band'])
        #P&L de la strategie
        recap_strat_vama=backtest(data, strategy_vama, "VAMA")
        #Graphique du P&L cumulatif
        axes2.plot(recap_strat_vama.index,recap_strat_vama["P&L_VAMA"], label =idx)
        axes2.legend(loc='upper left')
        axes2.set_title('P&L VAMA STRATEGY')
        dataframe_collection[idx]=recap_strat_vama
    plt.show()
    return dataframe_collection
#Dictionnaire des p&L par stock et ou indice
strategy_4=vama_strategy()


#La stratégie 1 prenant le moins de dates possibles, on réindexe les dictionnaires sur celles-ci
def resize_dict(strat):
    strat_resize={}
    for key in strategy_1:
        if key in strat:
            strat_resize[key]=strat[key].iloc[len(strat[key])-len(strategy_1[key]):len(strat[key]),:]
    return strat_resize

strategy_2=resize_dict(strategy_2)
strategy_3=resize_dict(strategy_3)
strategy_4=resize_dict(strategy_4)

#Cumul des P&L des stratégies pour un même stock ou un même indice
strat_tot={}
f3, axes3 = plt.subplots(figsize=(10,7))
for key in strategy_2:
    strat_tot[key]= pd.DataFrame(strategy_1[key]["pnl_long"].values +strategy_2[key]["pnl_long"].values+\
        strategy_3[key]["P&L_BB"].values+strategy_4[key]["P&L_VAMA"].values)
    strat_tot[key]["strat_total"]=strat_tot[key][0]
    strat_tot[key]["strat_1"]=strategy_1[key]["pnl_long"].values
    strat_tot[key]["strat_2"]=strategy_2[key]["pnl_long"].values
    strat_tot[key]["strat_3"]=strategy_3[key]["P&L_BB"].values
    strat_tot[key]["strat_4"]=strategy_4[key]["P&L_VAMA"].values
    strat_tot[key]=strat_tot[key].set_index(strategy_1[key]["pnl_dates"].values)
    #Export excel des prises de positions et P&L
    strat_tot[key].to_excel(f'P&L construction {key}.xlsx')
    #Calcul des statistiques associées à l'évolution du P&L
    statistics=strat_tot[key].describe(include='all') 
    #Export excel des statistiques
    statistics.to_excel(f'P&L statistics {key}.xlsx')
    
    
    axes3.plot(strat_tot[key].index,strat_tot[key]["strat_total"].values,label=key)
    axes3.set_title('P&L cumulé des stratégies')
    axes3.legend(loc='upper left')
    
#Affichage de l'ensemble des P&L cumulés
plt.show()
    


   
    