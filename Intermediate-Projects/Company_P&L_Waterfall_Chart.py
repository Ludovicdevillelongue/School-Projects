# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:33:49 2021

@author: ludov
"""

import plotly.graph_objects as go
from yahoo_fin import stock_info as si
import yfinance as yf



'''---------------------------// Get last financials from yahoo finance //-------------------------------'''

stock = si.get_income_statement('AAPL')
stock_name=yf.Ticker('AAPl')
name=stock_name.info['longName']
stock.reset_index(inplace=True)    
stock = stock.iloc[:,0:2]
stock=stock.fillna(0)



'''------------------------------// Pick financials needed for a P&L //-------------------------------'''

Revenue=stock[stock['Breakdown'] == 'totalRevenue'].iloc[0][1]
Cogs=stock[stock['Breakdown'] == 'costOfRevenue'].iloc[0][1]*-1
grossProfit = stock[stock['Breakdown'] == 'grossProfit'].iloc[0][1]
RandDev = stock[stock['Breakdown'] == 'researchDevelopment'].iloc[0][1]*-1
SGandA = stock[stock['Breakdown'] == 'sellingGeneralAdministrative'].iloc[0][1]*-1
totalexpenses=stock[stock['Breakdown'] == 'totalOperatingExpenses'].iloc[0][1]
amort=(totalexpenses+SGandA+RandDev+Cogs)*-1
Ebit= stock[stock['Breakdown'] == 'ebit'].iloc[0][1]
interest = stock[stock['Breakdown'] == 'totalOtherIncomeExpenseNet'].iloc[0][1]
Ebt = stock[stock['Breakdown'] == 'incomeBeforeTax'].iloc[0][1]
incTax = stock[stock['Breakdown'] == 'incomeTaxExpense'].iloc[0][1]*-1
netIncome =  stock[stock['Breakdown'] == 'netIncomeFromContinuingOps'].iloc[0][1]



'''---------------------------// Plot the waterfall chart in browser //-------------------------------'''

fig = go.Figure(go.Waterfall(
    name = "20", orientation = "v",
    measure = ["relative", "relative", "total", "relative", "relative","relative", "total","relative","total","relative","total"],
    x = ["Revenue", "COGS", "Gross Profit", "RD", "G&A","D&A", "EBIT","Interest Expense", "Earn Before Tax","Income Tax","Net Income"],
    textposition = "outside",
    text = [Revenue/100000, Cogs/100000, grossProfit/100000, RandDev/100000, SGandA/100000, amort/100000,
            Ebit/100000,interest/100000, Ebt/100000,incTax/100000, netIncome/100000],
    y = [Revenue, Cogs, grossProfit, RandDev, SGandA, amort, Ebit, interest, Ebt, incTax, netIncome], 
    connector = {"line":{"color":"rgb(63, 63, 63)"}}, )) 
fig.update_layout( 
    title = "Profit and loss statement, " + str(name),
    showlegend = True)
fig.show(renderer='browser')


