# -*- coding: utf-8 -*-
"""
Created on Tue May 17 00:07:57 2022

@author: ludov
"""


"""
Packages
"""
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from functools import reduce
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

'''
-----------------------------------------------------
------------Data Retrieval & Processing--------------
-----------------------------------------------------
'''


"""
Part I : Kenneth French’s website, Risk free rate, market return, asset pricing factor data
"""
###
# Stock return data // Daily
###

FF3 = pd.read_csv('F-F_Research_Data_Factors_daily.CSV', sep=',')
#df.isnull().sum --> data clean
FF3['Mkt'] = FF3['Mkt-RF'] + FF3['Mkt-RF']
FF3['Date'] = pd.to_datetime(FF3['Date'].astype(str), format='%Y%m%d')
FF3 = FF3.set_index('Date')
data_ff3 = (FF3.index> '2010-01-01') & (FF3.index<'2011-01-01')
Final_FF3 = FF3.loc[data_ff3]

FF5 = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.CSV', sep=',')
FF5['Mkt'] = FF5['Mkt-RF'] + FF5['Mkt-RF']
FF5['Date'] = pd.to_datetime(FF5['Date'].astype(str), format='%Y%m%d')
FF5 = FF5.set_index('Date')
data_ff5 = (FF5.index> '2010-01-01') & (FF5.index<'2011-01-01')
Final_FF5 = FF5.loc[data_ff5]

Momemtum = pd.read_csv('F-F_Momentum_Factor_daily.CSV', sep=',')
Momemtum['Date'] = pd.to_datetime(Momemtum['Date'].astype(str), format='%Y%m%d')
Momemtum = Momemtum.set_index('Date')
data_Momemtum = (Momemtum.index>'2010-01-01') & (Momemtum.index<'2011-01-01')
Final_Momemtum = Momemtum .loc[data_Momemtum]





"""
Part II AQR, betting against beta (BAB factors)
Fichier excel avec beaucoup de data
"""




"""
Part III : Kent Daniel, DMRS factors 
"""
DMRS_factor= pd.read_csv('dmrs_factor_portfolios_daily.txt', sep="	")
DMRS_factor['date'] = pd.to_datetime(DMRS_factor['date'].astype(str), format='%Y%m%d')
DMRS_factor = DMRS_factor.set_index('date')
data_DMRS_factor = (DMRS_factor.index>'1925-11-1') & (DMRS_factor.index<'2017-1-1')
Final_DMRS_factor= DMRS_factor.loc[data_DMRS_factor]

DMRS_hedge= pd.read_csv('dmrs_hedge_portfolios_daily.txt', sep="	")
DMRS_hedge['date'] = pd.to_datetime(DMRS_hedge['date'].astype(str), format='%Y%m%d')
DMRS_hedge = DMRS_hedge.set_index('date')
data_DMRS_hedge = (DMRS_hedge.index>'1925-11-1') & (DMRS_hedge.index<'2017-1-1')
Final_DMRS_hedge= DMRS_hedge.loc[data_DMRS_hedge]




###
# FRED = Macro data // Monthly
###


"""
Part IV :	FRED   o	Moody’s BaaAaa spread (Bonds spread)
"""

BaaAaa = pd.read_csv('BaaAaa.csv', sep=',')
BaaAaa['DATE'] = pd.to_datetime(BaaAaa['DATE'].astype(str), format='%Y%m%')
BaaAaa = BaaAaa.set_index('DATE')
data_BaaAaa = (BaaAaa.index>'1925-11-1') & (BaaAaa.index<'2017-1-1')
Final_BaaAaa = BaaAaa.loc[data_BaaAaa]




"""
Part V  :	FRED   o	Industrial production 
"""

INDPRO = pd.read_csv('INDPRO.csv', sep=',')
INDPRO['DATE'] = pd.to_datetime(INDPRO['DATE'].astype(str), format='%Y%m%')
INDPRO = INDPRO.set_index('DATE')
data_INDPRO = (INDPRO.index>'1925-11-1') & (INDPRO.index<'2017-1-1')
Final_INDPRO = INDPRO.loc[data_INDPRO]
Final_INDPRO=Final_INDPRO.rename_axis(index={'DATE': 'Date'})
Final_INDPRO.columns=["Mkt"]




"""
Part VI :	FRED   o    Initial claims --> à mensualier
"""

Claims = pd.read_csv('Initial_claims_monthly.csv', sep=',')
Claims['DATE'] = pd.to_datetime(Claims['DATE'].astype(str), format='%Y%m%')
Claims = Claims.set_index('DATE')
data_Claims = (Claims.index>'1967-2-1') & (Claims.index<'2017-1-1')
Final_Claims = Claims.loc[data_Claims]
Final_Claims=Final_Claims.rename_axis(index={'DATE': 'Date'})
Final_Claims.columns=["Mkt"]




"""
Part VII :	FRED   o	5 years treasury yield minus 3month T bills
"""
"""
Five_Year_Yield = pd.read_csv('5Y_treasury_Monthly.csv', sep=',').set_index("DATE")
data_Five_Year_Yield = (Five_Year_Yield.index> '1953-04-01') & (Five_Year_Yield.index<'2017-01-01')
Final_Five_Year_Yield = Five_Year_Yield.loc[data_Five_Year_Yield]
"""

Five_Year_Yield = pd.read_csv('5Y_treasury_Monthly.csv', sep=',')
Five_Year_Yield['DATE'] = pd.to_datetime(Five_Year_Yield['DATE'].astype(str), format='%Y%m%')
Five_Year_Yield = Five_Year_Yield.set_index("DATE")
data_Five_Year_Yield = (Five_Year_Yield.index> '1953-04-01') & (Five_Year_Yield.index<'2017-01-01')
Final_Five_Year_Yield = Five_Year_Yield.loc[data_Five_Year_Yield]

Three_T_BILL = pd.read_csv('TBILLS_3M_Monthly.csv', sep=',')
Three_T_BILL['DATE'] = pd.to_datetime(Three_T_BILL['DATE'].astype(str), format='%Y%m%')
Three_T_BILL = Three_T_BILL.set_index("DATE")
data_Three_T_BILL = (Three_T_BILL.index>'1953-04-01') & (Three_T_BILL.index<'2017-01-01')
Final_Three_T_BILL = Three_T_BILL.loc[data_Three_T_BILL]

Final_Slope = Final_Five_Year_Yield  - Final_Three_T_BILL




"""
Part VIII :	FRED   o    NBER recession indicators 
"""
NBER = pd.read_csv('NBER_Monthly.csv', sep=',')
NBER['DATE'] = pd.to_datetime(NBER['DATE'].astype(str), format='%Y%m%')
NBER = NBER.set_index('DATE')
data_NBER = (NBER.index>'1967-1-1') & (NBER.index<'2017-1-1')
Final_NBER = NBER .loc[data_NBER]





"""
Part IX  :	FRED   o    Quarterly real per Capita GDP
"""
GDPQuart = pd.read_csv('GDP_PERCAPITA_QUART.csv', sep=',')
GDPQuart['DATE'] = pd.to_datetime(GDPQuart['DATE'].astype(str), format='%Y%m%')
GDPQuart = GDPQuart.set_index('DATE')
data_GDPQuart = (GDPQuart.index>'1967-1-1') & (GDPQuart.index<'2017-1-1')
Final_GDPQuart= GDPQuart .loc[data_GDPQuart]




"""
Part X :	FRED   o    Consumption  
"""
Consumption = pd.read_csv('PersoConsumptionEC.csv', sep=',')
Consumption['DATE'] = pd.to_datetime(Consumption['DATE'].astype(str), format='%Y%m%')
Consumption = Consumption.set_index('DATE')
data_Consumption = (Consumption.index>'1967-1-1') & (Consumption.index<'2017-1-1')
Final_Consumption = Consumption .loc[data_Consumption]




'''
-----------------------------------------------------
----------------Beta Sorted Portfolio----------------
-----------------------------------------------------
'''


"""
Part I : Factor Dataset Selection
"""
def choose_factor(factor):
    market_ptf=factor.Mkt.to_frame()
    return market_ptf

Final_INDPRO.Mkt=((Final_INDPRO.Mkt/Final_INDPRO.Mkt.shift(6))-1)
Final_INDPRO=Final_INDPRO.dropna()

Final_Claims.Mkt=((Final_Claims.Mkt/Final_Claims.Mkt.shift(6))-1)
Final_Claims=Final_Claims.dropna()

Final_Slope.columns=['Mkt']
Final_Slope.Mkt=((Final_Slope.Mkt/Final_Slope.Mkt.shift(3))-1)
Final_Slope=Final_Slope.dropna()
Final_Slope=Final_Slope.rename_axis(index={'DATE': 'Date'})

Final_BaaAaa.columns=['Mkt']
Final_BaaAaa.Mkt=((Final_BaaAaa.Mkt/Final_BaaAaa.Mkt.shift(3))-1)
Final_BaaAaa=Final_BaaAaa.dropna()
Final_BaaAaa=Final_BaaAaa.rename_axis(index={'DATE': 'Date'})




market_ptf= choose_factor(Final_INDPRO)
    
market_ptf=market_ptf.loc[market_ptf.index>'2005-12-30']
market_ptf = market_ptf.reset_index(level=0)
market_ptf.insert(0,'date',pd.to_datetime(market_ptf['Date'], format='%m/%d/%Y'))
market_ptf=market_ptf.drop(columns=['Date'])
# convert into monthly periods 
market_ptf['date']=market_ptf['date'].dt.to_period('M')
#calculate rolling standard deviation
# shift below is used to use PREVIOUS 12 month
market_ptf['std_est'] = market_ptf['Mkt'].rolling(12).std().shift(1)





"""
Part II : Stocks Dataset
"""


df=pd.read_pickle("crspm2005_2020_test.pkl")
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
## calculate market cap
df['mkt_cap']=df['prc'].abs()*df['shrout'] 
df = df[['permno', 'date', 'mkt_cap', 'ret']]
# Here inside the square brackets we try to convert 'ret' to numeric values
df = df[pd.to_numeric(df['ret'], errors='coerce').notnull()]
df['ret'] = df['ret'].astype(float)
# Just in case convert stock ids into text format
df['permno'] = df['permno'].astype(str)
# convert into monthly periods 
df['date'] = df['date'].dt.to_period('M')

# To calculate betas we need to merge the stocks dataset with mkt returns
df_merged = pd.merge(df, market_ptf, on='date', how='left')






"""
Part III : Pre Formation Beta
"""
# define function to estimate rolling 2 year corr
def roll_corr(x):
    return pd.DataFrame(x['ret'].rolling(24).corr(x['Mkt']))


# define function to estimate rolling 24 month std
def roll_std(x):
    return pd.DataFrame(x['ret'].rolling(24).std())


#Then we need to apply this functions to each stock, for this purpose we use groupby 'id'
# Shift is used so that the current datapoint is not included
df_merged['corr_est'] = df_merged.groupby('permno')[['ret','Mkt']].apply(roll_corr)
df_merged['corr_est'] = df_merged.groupby('permno')['corr_est'].shift(1)
df_merged['id_std_est'] = df_merged.groupby('permno')[['ret','Mkt']].apply(roll_std)
df_merged['id_std_est'] = df_merged.groupby('permno')[['id_std_est']].shift(1)


#drop all the rows where in ANY column there is a NAN value
df_merged = df_merged.dropna(how='any')
# Estimation betas 
df_merged['beta_est'] = df_merged['corr_est']*df_merged['id_std_est']







"""
Part IV : Beta-sorted portfolio bucket
"""

df_merged.sort_values(inplace=True, by="date",ascending=True)
df_merged["date"]=pd.to_datetime(df_merged["date"].astype(str))


test_1=pd.DataFrame(columns=['Bucket_mkt'])
test_2=pd.DataFrame(columns=['Bucket_hedge'])
for ele in df_merged["date"].drop_duplicates().to_list():
    df_merged.index=df_merged.date
    temp=df_merged.loc[ele]
    temp_1=pd.qcut(temp['beta_est'], q=[0, 0.2, 0.4, 0.6, 0.8, 1], \
                                  labels=False)
    temp_2=pd.qcut(temp['beta_est'], q=[0,0.5,1], \
                                  labels=False)
    temp_1=temp_1.to_frame()
    temp_2=temp_2.to_frame()
    temp_1.columns=['Bucket_mkt']
    temp_2.columns=['Bucket_hedge']
    test_1 = pd.concat([test_1,temp_1],ignore_index=True)
    test_2 = pd.concat([test_2,temp_2],ignore_index=True)
  
    
  
    
  
"""
Part V : Market Portfolio
"""

#buckets creation
df_merged['bucket_mkt']=test_1['Bucket_mkt'].to_list()
df_merged.reset_index(drop=True, inplace=True)
df_bucket_1=df_merged.loc[df_merged['bucket_mkt']==0]
df_bucket_2=df_merged.loc[df_merged['bucket_mkt']==1]
df_bucket_3=df_merged.loc[df_merged['bucket_mkt']==2]
df_bucket_4=df_merged.loc[df_merged['bucket_mkt']==3]
df_bucket_5=df_merged.loc[df_merged['bucket_mkt']==4]


#function to value weight each portfolio bucket
def bucket_portfolios(bucket):
    test_3=pd.DataFrame()
    bucket["date"]=pd.to_datetime(bucket["date"].astype(str))
    # bucket["weight"]=-10
    for ele_2 in bucket["date"].drop_duplicates().to_list():
        bucket.index=bucket.date
        temp_3=bucket.loc[ele_2]
        total_market_cap=temp_3.mkt_cap.sum()
        temp_4=temp_3.mkt_cap/total_market_cap
        temp_4=temp_4.to_frame()
        temp_4.columns=['weight']
        test_3=pd.concat([test_3,temp_4],ignore_index=True)
    test_3.index=bucket.date
    # calculating return
    test_3['ret'] = (bucket['ret']*test_3['weight'])
    test_3['beta_est']=bucket['beta_est']
    # print(test_3.columns)
        # for each date and group (H, L) calculate the aggregate betas and rets
    result_bucket = test_3.groupby(['date'])['ret','beta_est'].sum().reset_index()
    # result_bucket.weight=test_3.weight
    # print(test_3['ret'] )
    return result_bucket



# test_3=pd.DataFrame()
# df_bucket_1["date"]=pd.to_datetime(df_bucket_1["date"].astype(str))
# df_bucket_1["weight"]=-10
# for ele_2 in df_bucket_1["date"].drop_duplicates().to_list():
#     df_bucket_1.index=df_bucket_1.date
#     temp_3=df_bucket_1.loc[ele_2]
#     total_market_cap=temp_3.mkt_cap.sum()
#     temp_4=temp_3.mkt_cap/total_market_cap
#     temp_4=temp_4.to_frame()
#     temp_4.columns=['weight']
#     test_3=pd.concat([test_3,temp_4],ignore_index=True)
# test_3.index=df_bucket_1.date
# # calculating return
# test_3['ret'] = (df_bucket_1['ret']*test_3['weight'])
# test_3['beta_est']=df_bucket_1['beta_est']
#     # for each date and group (H, L) calculate the aggregate betas and rets
# #result_bucket = test_3.groupby(['date'])['ret','beta_est'].sum().reset_index()
# # return result_bucket
    
#results by bucket
result_bucket_1=bucket_portfolios(df_bucket_1)
result_bucket_2=bucket_portfolios(df_bucket_2)
result_bucket_3=bucket_portfolios(df_bucket_3)
result_bucket_4=bucket_portfolios(df_bucket_4)
result_bucket_5=bucket_portfolios(df_bucket_5)

dataframes_market = [result_bucket_1, result_bucket_2, result_bucket_3,result_bucket_4,result_bucket_5]

result_portfolio_market = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                            how='inner'), dataframes_market)
result_portfolio_market.columns=["date","ret_bucket_1","beta_bucket_1",\
                          "ret_bucket_2","beta_bucket_2",\
                          "ret_bucket_3","beta_bucket_3",
                          "ret_bucket_4","beta_bucket_4",
                          "ret_bucket_5","beta_bucket_5"]
    
ret_list_mkt=["ret_bucket_1","ret_bucket_2","ret_bucket_3","ret_bucket_4","ret_bucket_5"]
beta_list_mkt=["beta_bucket_1","beta_bucket_2","beta_bucket_3","beta_bucket_4","beta_bucket_5"]
result_portfolio_market["ret"]=(result_portfolio_market[ret_list_mkt].sum(axis=1))
result_portfolio_market["beta_est"]=result_portfolio_market[beta_list_mkt].sum(axis=1)

#regression function
def regression(result_port):
    X=Final_Claims.loc["2008-01-01 00:00:00":"2016-12-01 00:00:00"]

    y=(result_port["ret"]).shift(1).dropna()
    print(len(y))
    
    #X=np.array((((X-mean)/st_dev).shift(-1)).dropna())
    mean=X.mean()
    st_dev=X.std()
    X=(X-mean)/st_dev
    X=X.shift(-1)
    X=X.dropna()
    X=X*(-1)
    print(len(X))
    X=np.array(X)
    #X=sm.add_constant(X)
   
    result = sm.OLS(y, X).fit()
    # X = sm.add_constant(x)
    return print(result.summary())
    
#regression on market portfolio
regression_mkt=regression(result_portfolio_market)





"""
Part VI : Hedge Portfolio
"""
#creation of two buckets: beta low and beta high
df_merged['bucket_hedge']=test_2['Bucket_hedge'].to_list()
#df_merged.reset_index(drop=True, inplace=True)
df_bucket_low=df_merged.loc[df_merged['bucket_hedge']==0]
df_bucket_high=df_merged.loc[df_merged['bucket_hedge']==1]

result_bucket_low = bucket_portfolios(df_bucket_low)
result_bucket_high = bucket_portfolios(df_bucket_high)

dataframes_hedge=[result_bucket_low,result_bucket_high]
result_portfolio_hedge = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                            how='inner'), dataframes_hedge)

result_portfolio_hedge.columns=["date","ret_bucket_low","beta_bucket_low",\
                          "ret_bucket_high","beta_bucket_high"]
    
#results by bucket (inverse sign for short high beta)
result_portfolio_hedge["ret_bucket_high"]=result_portfolio_hedge["ret_bucket_high"]*(-1)
result_portfolio_hedge["beta_bucket_high"]=result_portfolio_hedge["beta_bucket_high"]*(-1)
ret_list_hedge=["ret_bucket_low","ret_bucket_high"]
beta_list_hedge=["beta_bucket_low","beta_bucket_high"]
result_portfolio_hedge["ret"]=(result_portfolio_hedge[ret_list_hedge].sum(axis=1))
result_portfolio_hedge["beta_est"]=result_portfolio_hedge[beta_list_hedge].sum(axis=1)

#regression on hedge portfolio
regression_hedge=regression(result_portfolio_hedge)






"""
Part VII : Market Hedge Portfolio
"""

#results for market+hedge portfolio
result_portfolio_mkt_hedge=result_portfolio_market[["date","ret","beta_est"]].merge(\
    result_portfolio_hedge[["date","ret","beta_est"]],on="date")
result_portfolio_mkt_hedge.columns=["date","total_ret_mkt","total_beta_mkt","total_ret_hedge",\
                                    "total_beta_hedge"]
result_portfolio_mkt_hedge["ret"]=result_portfolio_mkt_hedge["total_ret_mkt"]\
    +result_portfolio_mkt_hedge["total_ret_hedge"]
result_portfolio_mkt_hedge["beta_est"]=result_portfolio_mkt_hedge["total_beta_mkt"]\
    +result_portfolio_mkt_hedge["total_beta_hedge"]

#regression on market+hedge portfolio    
regression_hedge=regression(result_portfolio_mkt_hedge)




"""
Part VIII :Portfolios plot
"""

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name
    

list_portfolios=[result_portfolio_market,result_portfolio_mkt_hedge]
f_ptf, axes_ptf = plt.subplots(figsize=(10,3.75))
for ptf in list_portfolios:
    axes_ptf.plot(ptf['date'], ptf['ret'],label=get_df_name(ptf),\
                    linestyle='dashed', linewidth=2)
    axes_ptf.set_xlabel("Dates")
    axes_ptf.set_ylabel("Returns")
    axes_ptf.legend(loc='upper left')
    axes_ptf.set_title('Cumulative Return')
plt.show()



f_ptf_b, axes_ptf_b = plt.subplots(figsize=(10,3.75))
for ptf_2 in list_portfolios:
    axes_ptf_b.plot(ptf_2['date'], ptf_2['beta_est'],label=get_df_name(ptf_2),\
                    linestyle='dashed', linewidth=2)
    axes_ptf_b.set_xlabel("Dates")
    axes_ptf_b.set_ylabel("Betas")
    axes_ptf_b.legend(loc='upper left')
    axes_ptf_b.set_title('Cumulative Betas')
plt.show()
print(ptf['beta_est'])



"""
Part IX : Univariate Factors with low minus high portfolio
"""
#Function to simplify plot
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name
    
#Function to simplify plot
def get_df_port_name(df_port):
    name =[x for x in globals() if globals()[x] is df_port][0]
    return name

def choose_factor_LMH(factor):
    market_ptf=factor.to_frame()      
    market_ptf=market_ptf.loc[market_ptf.index>'2005-12-30']
    market_ptf = market_ptf.reset_index(level=0)
    market_ptf.insert(0,'date',pd.to_datetime(market_ptf['Date'], format='%m/%d/%Y'))
    market_ptf=market_ptf.drop(columns=['Date'])
    # convert into monthly periods 
    market_ptf['date']=market_ptf['date'].dt.to_period('M')
    #calculate rolling standard deviation
    # shift below is used to use PREVIOUS 12 month
    market_ptf['std_est'] = market_ptf.iloc[:,1].rolling(12).std().shift(1)
    
    
    
    #load data
    df=pd.read_pickle("C:/Users/ludov/Documents/Dauphine/M2/S2/outils quant/projet/crspm2005_2020.pkl")
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    ## calculate market cap
    df['mkt_cap']=df['prc'].abs()*df['shrout'] 
    df = df[['permno', 'date', 'mkt_cap', 'ret']]
    # Here inside the square brackets we try to convert 'ret' to numeric values
    df = df[pd.to_numeric(df['ret'], errors='coerce').notnull()]
    df['ret'] = df['ret'].astype(float)
    # Just in case convert stock ids into text format
    df['permno'] = df['permno'].astype(str)
    # convert into monthly periods 
    df['date'] = df['date'].dt.to_period('M')
    # To calculate betas we need to merge the stocks dataset with mkt returns
    df = pd.merge(df, market_ptf, on='date', how='left')
    
    
    # define function to estimate rolling 5 year(60 month) 
    def roll_corr(x):
        return pd.DataFrame(x['ret'].rolling(60).corr(x[market_ptf.columns[1]]))
    
    
    #same for rolling std, but with 1 year horizon
    def roll_var(x):
        return pd.DataFrame(x['ret'].rolling(12).std())
    
    
    #correlation return factor
    df['corr_est'] = df.groupby('permno')[['ret', market_ptf.columns[1]]].apply(roll_corr)
    df['corr_est'] = df.groupby('permno')['corr_est'].shift(1)
    #vol return factor
    df['id_var_est'] = df.groupby('permno')[['ret',  market_ptf.columns[1]]].apply(roll_var)
    df['id_var_est'] = df.groupby('permno')[['id_var_est']].shift(1)
    
    
    #drop all the rows where in any column there is a NAN value
    df = df.dropna(how='any')
    # Estimation betas according to Frazzini and Pedersen
    df['beta_est'] = df['corr_est']*df['id_var_est'].div(df['std_est'])
    #Shrink the betas to make them less noisy 
    df['beta_est'] = 0.6*df['beta_est'] + 0.4
    # assign stocks into portfolios based on the their beta_est quantile
    df['q'] = df.groupby('date')['beta_est'].apply(lambda x: pd.cut(x, bins=2, right=False, labels=range(1,3), duplicates = 'drop'))
    
    
    # check the average mean excess return and average estimated beta and compare it with the table 3
    print(df.groupby('q')[['beta_est', 'ret']].mean())
    alpha = df.groupby(['date', 'q'])[['ret',  market_ptf.columns[1]]].mean().reset_index()
    alpha.columns = ['date', 'q', 'ret',  market_ptf.columns[1]]
    
    
    #Create an empty dataframe to store estimated alphas and betas
    par = pd.DataFrame()
    for i in range(1,3):
        # pick the portfolio
        alpha0 = alpha[alpha['q']==i]
        alpha0=alpha0.dropna()
        x = alpha0[market_ptf.columns[1]].copy()
        x=x.dropna()
        # add a constant to the model, default ols goes without it
        x = sm.add_constant(x)
        #estimate the model
        results = sm.OLS(alpha0['ret'], x).fit(cov_type='HC1')
        #print the table with estimates
        print(results.summary())
        # get alphas and betas
        par0 = results.params
        # give a name to the row of parameters, we need to know for which portfolio we got the estimates
        par0.name= 'port_{}'.format(i)
        # join it with the dataframe of params
        par = par.append(par0)
    par.columns = ['alpha', 'beta_realised']
    return par

f2, axes2 = plt.subplots(figsize=(10,3.75))
Mkt_RF=Final_FF5['Mkt-RF']
SMB=Final_FF5['SMB']
HML=Final_FF5['HML']
CMA=Final_FF5["CMA"]
RMW=Final_FF5['RMW']
list_factors=[Mkt_RF,SMB,HML,CMA,RMW,Final_Momemtum.squeeze()]
alphas=[]
t_stats_alphas=[]

for factor in list_factors:
    port_param= choose_factor_LMH(factor)
    alphas.append(sum(port_param["alpha"]))
    
    #t stat alpha
    mean = np.mean(port_param["alpha"])
    std_error = np.std(port_param["alpha"]) / np.sqrt(len(port_param["alpha"]))
    t_stats_alphas.append(abs(mean)/std_error)
    
    axes2.plot(port_param['beta_realised'], port_param['alpha'],label=factor.name,\
               linestyle='dashed',marker='o',markersize=12, linewidth=2)
    axes2.set_xlabel("post formation beta")
    axes2.set_ylabel("alpha")
    axes2.legend(loc='upper left')
    axes2.set_title('Alphas of LMH sorted portfolios')
plt.show()


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
factor_names = [get_df_name(list_factors[0]),get_df_name(list_factors[1]),\
                get_df_name(list_factors[2]),get_df_name(list_factors[3]),\
                    get_df_name(list_factors[4]),get_df_name(list_factors[5])]
ax.bar(factor_names, alphas)
ax.set_xticks(factor_names, labels=['Mkt_RF', 'SMB', 'HML', 'CMA', 'RMW','MOM'])
ax.set_title('LMH portfolios alphas')
plt.show()

#t-stat alphas plot
fig_t_stat = plt.figure()
ax_t_stat = fig_t_stat.add_axes([0,0,1,1])
factor_names = [get_df_port_name(list_factors[0]),get_df_port_name(list_factors[1]),\
                get_df_port_name(list_factors[2]),get_df_port_name(list_factors[3]),\
                    get_df_port_name(list_factors[4]),get_df_port_name(list_factors[5])]
ax_t_stat.bar(factor_names, t_stats_alphas)
ax_t_stat.set_xticks(factor_names, labels=['Mkt_RF', 'SMB', 'HML', 'CMA', 'RMW','MOM'])
ax_t_stat.set_title('LMH portfolios alphas t-stat')
plt.show()


"""
Part X : Univariate Factors with 8 portfolios merged
"""


df_port=pd.read_pickle("C:/Users/ludov/Documents/Dauphine/M2/S2/outils quant/projet/ret_tot_bet.pkl")
df_port = df_port.reset_index(level=0)
df_port['date'] = pd.to_datetime(df_port['date'], format='%Y%m%d')
df_port['date'] = df_port['date'].dt.to_period('M')

def choose_factor_multivariate(factor,df_port):

    
    market_ptf=factor.to_frame()      
    market_ptf=market_ptf.loc[market_ptf.index>'2005-12-30']
    market_ptf = market_ptf.reset_index(level=0)
    market_ptf.insert(0,'date',pd.to_datetime(market_ptf['Date'], format='%m/%d/%Y'))
    market_ptf=market_ptf.drop(columns=['Date'])
    # convert into monthly periods 
    market_ptf['date']=market_ptf['date'].dt.to_period('M')
    #calculate rolling standard deviation
    # shift below is used to use PREVIOUS 12 month
    market_ptf['std_est'] = market_ptf.iloc[:,1].rolling(12).std().shift(1)
    
    df_port = pd.merge(df_port, market_ptf, on='date', how='left')
    
    
    
    #correlation return factor
    df_port['corr_est'] = pd.DataFrame(df_port['ret'].rolling(60).corr(df_port[market_ptf.columns[1]]))
    df_port['corr_est'] = df_port['corr_est'].shift(1)
    #vol return factor
    df_port['id_var_est'] = pd.DataFrame(df_port['ret'].rolling(12).std())
    df_port['id_var_est'] = df_port[['id_var_est']].shift(1)
    
    
    #drop all the rows where in any column there is a NAN value
    df_port = df_port.dropna(how='any')
    # Estimation betas according to Frazzini and Pedersen
    df_port['beta_est'] = df_port['corr_est']*df_port['id_var_est'].div(df_port['std_est'])
    #Shrink the betas to make them less noisy 
    df_port['beta_est'] = 0.6*df_port['beta_est'] + 0.4
    # assign stocks into portfolios based on the their beta_est quantile
    df_port['q'] = df_port.groupby('date')['beta_est'].apply(lambda x: pd.cut(x, bins=2, right=False, labels=range(1,3), duplicates = 'drop'))
    
    # for each month we rank betas and calculate the weights like in eq(16) in the paper
    df_port['rank'] = df_port.groupby('date')['beta_est'].rank()
    # z_bar
    df_port['rank_avg'] = df_port.groupby('date')['rank'].transform('mean')
    #abs(z-z_bar)
    df_port['weight'] = abs(df_port['rank']-df_port['rank_avg'])
    
    # calculate constant k
    df_port['k'] = 2/(df_port.groupby('date')['weight'].transform('sum')).copy()
    print(df_port)
    # calculate final weights
    df_port['weight'] = df_port['weight']*df_port['k']
    
        # calculating beta_H and beta_L
    df_port['dot'] = (df_port['beta_est']*df_port['weight'])
    # calculating r_H and r_L
    df_port['ret'] = (df_port['ret']*df_port['weight'])
    

    print(df_port.groupby('q')[['dot', 'ret']].mean())
    alpha = df_port.groupby(['date', 'q'])[['ret', market_ptf.columns[1]]].mean().reset_index()
    alpha.columns = ['date', 'q', 'ret',  market_ptf.columns[1]]
    
    
    #Create an empty dataframe to store estimated alphas and betas
    par = pd.DataFrame()
    for i in range(1,3):
        # pick the portfolio
        alpha0 = alpha[alpha['q']==i]
        alpha0=alpha0.dropna()
        x = alpha0[market_ptf.columns[1]].copy()
        x=x.dropna()
        # add a constant to the model, default ols goes without it
        x = sm.add_constant(x)
        #estimate the model
        results = sm.OLS(alpha0['ret'], x).fit(cov_type='HC1')
        #print the table with estimates
        print(results.summary())
        # get alphas and betas
        par0 = results.params
        # give a name to the row of parameters, we need to know for which portfolio we got the estimates
        par0.name= 'port_{}'.format(i)
        # join it with the dataframe of params
        par = par.append(par0)
    par.columns = ['alpha', 'beta_realised']
    return par
    
    
f2, axes2 = plt.subplots(figsize=(10,3.75))
Mkt_RF=Final_FF5['Mkt-RF']
SMB=Final_FF5['SMB']
HML=Final_FF5['HML']
CMA=Final_FF5["CMA"]
RMW=Final_FF5['RMW']
list_factors=[Mkt_RF,SMB,HML,CMA,RMW,Final_Momemtum.squeeze()]
alphas=[]
t_stats_alphas=[]

for factor in list_factors:
    port_param= choose_factor_multivariate(factor,df_port)
    
    #alpha calculation
    alphas.append(sum(port_param["alpha"]))
    
    
    
    #t stat alpha
    mean = np.mean(port_param["alpha"])
    std_error = np.std(port_param["alpha"]) / np.sqrt(len(port_param["alpha"]))
    t_stats_alphas.append(abs(mean)/std_error)
    
    
    #alpha vs beta plot
    axes2.plot(port_param['beta_realised'], port_param['alpha'],label=factor.name,\
                linestyle='dashed',marker='o',markersize=12, linewidth=2)
    axes2.set_xlabel("post formation beta")
    axes2.set_ylabel("alpha")
    axes2.legend(loc='upper left')
    axes2.set_title('Alphas of beta sorted portfolios')
plt.show()

#alphas plot
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
factor_names = [get_df_port_name(list_factors[0]),get_df_port_name(list_factors[1]),\
                get_df_port_name(list_factors[2]),get_df_port_name(list_factors[3]),\
                    get_df_port_name(list_factors[4]),get_df_port_name(list_factors[5])]
ax.bar(factor_names, alphas)
ax.set_xticks(factor_names, labels=['Mkt_RF', 'SMB', 'HML', 'CMA', 'RMW','MOM'])
ax.set_title('Beta sorted portfolios alphas')
plt.show()

#t-stat alphas plot
fig_t_stat = plt.figure()
ax_t_stat = fig_t_stat.add_axes([0,0,1,1])
factor_names = [get_df_port_name(list_factors[0]),get_df_port_name(list_factors[1]),\
                get_df_port_name(list_factors[2]),get_df_port_name(list_factors[3]),\
                    get_df_port_name(list_factors[4]),get_df_port_name(list_factors[5])]
ax_t_stat.bar(factor_names, t_stats_alphas)
ax_t_stat.set_xticks(factor_names, labels=['Mkt_RF', 'SMB', 'HML', 'CMA', 'RMW','MOM'])
ax_t_stat.set_title('Beta sorted portfolios alphas t-stat')
plt.show()



"""
Part XI : Multivariate Factors
"""

def get_factor_model(factor_model:list):
    alphas=[] 
    t_stats_alphas=[]
    for factor in factor_model:
        port_param= choose_factor_multivariate(factor,df_port)
        #alpha calculation
        alphas.append(sum(port_param["alpha"]))
        
        
        
        #t stat alpha
        mean = np.mean(port_param["alpha"])
        std_error = np.std(port_param["alpha"]) / np.sqrt(len(port_param["alpha"]))
        t_stats_alphas.append(abs(mean)/std_error)
        
        
    alpha_total=sum(alphas)
    t_stat_total=sum(t_stats_alphas)
    return alpha_total, t_stat_total


#list of factor models
FF3=[Final_FF3['Mkt-RF'],Final_FF3['SMB'],Final_FF3['HML']]
FF3_MOM=[Final_FF3['Mkt-RF'],Final_FF3['SMB'],Final_FF3['HML'],Final_Momemtum.squeeze()]
FF5=[Final_FF5['Mkt-RF'],Final_FF5['SMB'],Final_FF5['HML'],Final_FF5["CMA"],Final_FF5['RMW']]
FF5_MOM=[Final_FF5['Mkt-RF'],Final_FF5['SMB'],Final_FF5['HML'],Final_FF5["CMA"],Final_FF5['RMW'],Final_Momemtum.squeeze()]
factor_models=[FF3,FF3_MOM,FF5,FF5_MOM]

#loop on factor models to create alphas comparative graph
f3 = plt.figure()
axes3=f3.add_axes([0,0,1,1])
for factor_model in factor_models:
    model_alpha=get_factor_model(factor_model)[0]
    axes3.bar(get_df_name(factor_model),model_alpha)
    axes3.set_title('Multi Factor Beta sorted portfolios alphas')
axes3.set_xticks(['FF3', 'FF3_MOM', 'FF5', 'FF5_MOM'],labels=['FF3', 'FF3_MOM', 'FF5', 'FF5_MOM'])
plt.show()


#loop on factor models to create t-stats alphas comparative graph
f4 = plt.figure()
axes4=f4.add_axes([0,0,1,1])
for factor_model in factor_models:
    model_alpha=get_factor_model(factor_model)[1]
    axes4.bar(get_df_name(factor_model), model_alpha)
    axes4.set_title('Multi Factor Beta sorted portfolios alphas t-stat')
axes3.set_xticks(['FF3', 'FF3_MOM', 'FF5', 'FF5_MOM'],labels=['FF3', 'FF3_MOM', 'FF5', 'FF5_MOM'])
plt.show()




