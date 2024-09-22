# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:15:34 2020

@author: ludov
"""
import pandas as pd
import numpy as np
from scipy import stats
from pandas import *
import matplotlib.pyplot as plt


'''---------------------------// Look for efficient frontier portfolio //-------------------------------'''





#dataframe creation
file=ExcelFile(r'African_Indexes_Portfolio.xlsx')
df=file.parse('Data')

#length of downloaded historical data
T=len(df)
df=df.set_index('TIME_PERIOD')

#returns calculation
returns = df[['NSE','EGX','JSE','MASI']].pct_change()

mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0178


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=df[['NSE','EGX','JSE','MASI']].columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=df[['NSE','EGX','JSE','MASI']].columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print ("-"*80)
    print ("Sharpe ratio maximization portfolio allocations \n") 
    print ("annualized return:", round(rp,2))
    print ("annualized volatility:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Volatility minimization portfolio allocations\n")
    print ("annualized return:", round(rp_min,2))
    print ("annualized volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Sharpe ratio maximum')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Volatility minimum')
    plt.title('Efficient frontier based portfolio optimization simulation')
    plt.xlabel('annualized volatility')
    plt.ylabel('annualized return')
    plt.legend(labelspacing=0.8)
    plt.show()
    
    
    
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)


'''---------------------------// VAR test according to weights //-------------------------------'''


#initial portfolio value
portfolio_value = 15000000

#risk threshold
alpha = 0.01

'''---------------------------/ Portfolio I allocation /-------------------------------'''


#weights for each index
weights1 = np.array([.4, .2, .2, .2])
 


#portfolio returns calculation by multiplying with weights
portfolio_returns_1 = returns.dot(weights1)

#scaling
portfolio_returns_1=portfolio_returns_1[1:T]

#sort portfolio returns
portfolio_returns_sorted_1=portfolio_returns_1.sort_values(ascending=True)




#calculation of absolute and relative VaR at 99%
VaR_historical_1day_relative_1=abs(portfolio_returns_sorted_1[int(alpha*T)])
VaR_historical_1day_absolue_1=(VaR_historical_1day_relative_1)*portfolio_value 

VaR_historical_10days_relative_1=VaR_historical_1day_relative_1*np.sqrt(10)
VaR_historical_10days_absolue_1=VaR_historical_10days_relative_1*portfolio_value


VaR_historical_1year_relative_1=VaR_historical_1day_relative_1 *np.sqrt(250)
VaR_historical_1year_absolue_1=VaR_historical_1year_relative_1*portfolio_value




#calculation of expected shortfall
expected_shortfall_historical_1day_relative_1=-np.array(portfolio_returns_sorted_1[:int(alpha*T)]).mean()
expected_shortfall_historical_1day_absolue_1=expected_shortfall_historical_1day_relative_1*portfolio_value

expected_shortfall_historical_10days_relative_1=expected_shortfall_historical_1day_relative_1*np.sqrt(10)
expected_shortfall_historical_10days_absolue_1=expected_shortfall_historical_10days_relative_1*portfolio_value



    

'''---------------------------/ Portfolio II allocation /-------------------------------'''


#weights for each index
weights2 = np.array([.2, .4, .2, .2])
 


#portfolio returns calculation by multiplying with weights
portfolio_returns_2 = returns.dot(weights2)

#scaling
portfolio_returns_2=portfolio_returns_2[1:T]

#sort portfolio returns
portfolio_returns_sorted_2=portfolio_returns_2.sort_values(ascending=True)




#calculation of absolute and relative VaR at 99%
VaR_historical_1day_relative_2=abs(portfolio_returns_sorted_2[int(alpha*T)])
VaR_historical_1day_absolue_2=(VaR_historical_1day_relative_2)*portfolio_value 

VaR_historical_10days_relative_2=VaR_historical_1day_relative_2*np.sqrt(10)
VaR_historical_10days_absolue_2=VaR_historical_10days_relative_2*portfolio_value


VaR_historical_1year_relative_2=VaR_historical_1day_relative_2 *np.sqrt(250)
VaR_historical_1year_absolue_2=VaR_historical_1year_relative_2*portfolio_value




#Calculation of expected shortfall
expected_shortfall_historical_1day_relative_2=-np.array(portfolio_returns_sorted_2[:int(alpha*T)]).mean()
expected_shortfall_historical_1day_absolue_2=expected_shortfall_historical_1day_relative_2*portfolio_value

expected_shortfall_historical_10days_relative_2=expected_shortfall_historical_1day_relative_2*np.sqrt(10)
expected_shortfall_historical_10days_absolue_2=expected_shortfall_historical_10days_relative_2*portfolio_value



    
    
    
'''---------------------------/ Portfolio III allocation /-------------------------------'''

#weights for each index
weights3 = np.array([.2, .2, .4, .2])
 


#portfolio returns calculation by multiplying with weights
portfolio_returns_3 = returns.dot(weights3)

#scaling
portfolio_returns_3=portfolio_returns_3[1:T]

#sort portfolio returns
portfolio_returns_sorted_3=portfolio_returns_3.sort_values(ascending=True)



#calculation of absolute and relative VaR at 99%
VaR_historical_1day_relative_3=abs(portfolio_returns_sorted_3[int(alpha*T)])
VaR_historical_1day_absolue_3=(VaR_historical_1day_relative_3)*portfolio_value 

VaR_historical_10days_relative_3=VaR_historical_1day_relative_3*np.sqrt(10)
VaR_historical_10days_absolue_3=VaR_historical_10days_relative_3*portfolio_value


VaR_historical_1year_relative_3=VaR_historical_1day_relative_3 *np.sqrt(250)
VaR_historical_1year_absolue_3=VaR_historical_1year_relative_3*portfolio_value




#calculation of expected shortfall
expected_shortfall_historical_1day_relative_3=-np.array(portfolio_returns_sorted_3[:int(alpha*T)]).mean()
expected_shortfall_historical_1day_absolue_3=expected_shortfall_historical_1day_relative_3*portfolio_value

expected_shortfall_historical_10days_relative_3=expected_shortfall_historical_1day_relative_3*np.sqrt(10)
expected_shortfall_historical_10days_absolue_3=expected_shortfall_historical_10days_relative_3*portfolio_value




    
    
    
'''---------------------------/ Portfolio IV allocation /-------------------------------'''

#weights for each index
weights4 = np.array([.2, .2, .2, .4])
 


#portfolio returns calculation by multiplying with weights
portfolio_returns_4 = returns.dot(weights4)

#scaling
portfolio_returns_4=portfolio_returns_4[1:T]

#sort portfolio returns
portfolio_returns_sorted_4=portfolio_returns_4.sort_values(ascending=True)




#calculation of absolute and relative VaR at 99%
VaR_historical_1day_relative_4=abs(portfolio_returns_sorted_4[int(alpha*T)])
VaR_historical_1day_absolue_4=(VaR_historical_1day_relative_4)*portfolio_value 

VaR_historical_10days_relative_4=VaR_historical_1day_relative_4*np.sqrt(10)
VaR_historical_10days_absolue_4=VaR_historical_10days_relative_4*portfolio_value


VaR_historical_1year_relative_4=VaR_historical_1day_relative_4 *np.sqrt(250)
VaR_historical_1year_absolue_4=VaR_historical_1year_relative_4*portfolio_value




#calculation of expected shortfall
expected_shortfall_historical_1day_relative_4=-np.array(portfolio_returns_sorted_4[:int(alpha*T)]).mean()
expected_shortfall_historical_1day_absolue_4=expected_shortfall_historical_1day_relative_4*portfolio_value

expected_shortfall_historical_10days_relative_4=expected_shortfall_historical_1day_relative_4*np.sqrt(10)
expected_shortfall_historical_10days_absolue_4=expected_shortfall_historical_10days_relative_4*portfolio_value



