"""
Author: Ludovic de Villelongue
"""

# packages import
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime
import yfinance as yf


# portfolio allocation class creation
class Portfolio_Allocation:

    def __init__(self, data: pd.DataFrame = None):
        # define the return and log_return
        #self.data = data.ffill().astype(float).dropna()
        #self.log_ret = np.log(self.data / self.data.shift(1)).dropna()
        self.data=data.ffill().astype(float).dropna()
        self.log_ret=(self.data).dropna()

    def get_cov(self,window: int = None):
        cov = self.log_ret.rolling(window).cov()
        cov=cov.reset_index(level=[0]).set_index("date")
        return cov


    def get_mean_return(self):
        mean_return = self.log_ret.mean()
        return mean_return

    def init_weights(self):
        # create equally weighted basket
        nb_assets = data.shape[1]
        equal_weights = np.array([1 / nb_assets] * nb_assets)
        return equal_weights

    def get_vol(self, window: int = None):
        # compute rolling volatility for all stocks
        vols = np.sqrt(12 / window * np.square(self.log_ret).rolling(window - 1).sum())
        return vols

    def portfolio_annualised_performance(self, weights: pd.DataFrame = None, mean_return: pd.DataFrame = None, cov:pd.DataFrame = None):
        returns = np.dot(mean_return, weights)
        std = np.sqrt(float(np.dot(weights, np.dot(cov, weights.T))))

        return returns, std

    def calc_diversification_ratio(self, weights: pd.DataFrame = None,mean_return: pd.DataFrame = None, cov: pd.DataFrame = None,
                                   vol: pd.DataFrame = None):
        # average weighted vol
        w_vol = np.dot(vol, weights)
        # portfolio vol
        port_vol = self.portfolio_annualised_performance(weights,mean_return,cov)[1]
        # maximize w_vol/port_vol is the same as minimizing (-(w_vol/port_vol))
        div_ratio = (-(w_vol / port_vol))
        # negative diversification ratio
        return div_ratio

    def neg_sharpe_ratio(self, weights: pd.DataFrame = None, mean_return: pd.DataFrame = None,
                         cov: pd.DataFrame = None, risk_free_rate: float = None):
        p_ret, p_var = self.portfolio_annualised_performance(weights, mean_return, cov)
        return -(p_ret - risk_free_rate) / p_var

    def portfolio_volatility(self, weights: pd.DataFrame = None, mean_return: pd.DataFrame = None,
                             cov: pd.DataFrame = None):
        return self.portfolio_annualised_performance(weights,mean_return,cov)[1]

    def port_alloc(self, alloc_type: str = None, window=4, risk_free_rate=0.00, long_only=False):
        """
        long_only=False -> short selling is authorized, weights can be negative
        long_only=True -> weights can't be negative
        window -> size of rolling window
        """
        # creation of weights_fin dataframe to add weights at each date
        weights_fin = pd.DataFrame(data=[], index=self.data.index, columns=self.data.columns)
        # insert equal weights in the dataframe weights_fin
        weights_fin.iloc[window, :] = self.init_weights()
        count = window
        # loop through date
        for date in self.data.index[window:-1, ]:
            count += 1
            # volatility for all stocks at each date
            vols = self.get_vol(window).dropna()
            vol = vols[vols.index == date]
            # covariance matrix
            covs = self.get_cov(window)
            cov=covs[covs.index==date]
            # mean returns
            mean_return = self.get_mean_return()
            # scipy constraints
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            # add long only constraint, which states that each weight should be superior or equal to zero
            if long_only:  # add long only constraint
                cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, {'type': 'ineq', 'fun': lambda w: w}]
            # scipy optimization to find max diversified portfolio
            if alloc_type == "Max_Div":
                res = minimize(self.calc_diversification_ratio, weights_fin[weights_fin.index == date],
                               args=(mean_return, cov, vol), method='SLSQP', constraints=cons)
            # scipy optimization to find max-sharpe portfolio
            elif alloc_type == "Max_Sharpe":
                res = minimize(self.neg_sharpe_ratio, weights_fin[weights_fin.index == date],
                               args=(mean_return, cov, risk_free_rate), method='SLSQP', constraints=cons)
            # scipy optimization to find min-vol portfolio
            elif alloc_type == "Min_Vol":
                res = minimize(self.portfolio_volatility, weights_fin.loc[weights_fin.index == date],
                               args=(mean_return, cov), method='SLSQP', constraints=cons)
            # put result from scipy inside dataframe
            weights_fin.iloc[count] = res.x
        # clean missing values
        weights_fin = weights_fin.dropna()

        return weights_fin

    def port_value(self,alloc_type: str = None, freq: str = None, weights_fin:pd.DataFrame=None):

        # monthly rebalancing
        if freq == "Monthly":
            weights_fin = weights_fin.resample('M').last()
        # daily rebalancing (by default)
        else:
            weights_fin = weights_fin


        # portfolio weights evolution plot
        f2, axes2 = plt.subplots(figsize=(10, 3.75))
        for column in weights_fin:
            axes2.plot(weights_fin.index, weights_fin[column], label=column)
            axes2.legend(loc='upper left')
            axes2.set_title(f"{alloc_type} Weight By Asset")
        plt.show()

        # portfolio value according to weights and crypto prices
        df_port_value = (self.data * weights_fin).combine_first(weights_fin).dropna()
        df_port_value["Tot_Port"] = [np.sum(df_port_value.iloc[i]) for i in range(0, len(df_port_value.index))]

        # portfolio total value plot
        f3, axes3 = plt.subplots(figsize=(10, 3.75))
        axes3.plot(df_port_value.index, df_port_value["Tot_Port"])
        axes3.set_title(f"{alloc_type} Portfolio Value Through Time")
        plt.show()
        return df_port_value


if __name__ == '__main__':
    # get data from portfolios

    data = pd.read_pickle("C:/Users/ludov/Documents/Dauphine/M2/S2/outils quant/projet/fullret.pkl")

    # obtain portfolio according to allocation type and frequency
    MDP = Portfolio_Allocation(data)
    weights_max_div = MDP.port_alloc("Max_Div")
    weights_min_vol = MDP.port_alloc("Min_Vol")
    weights_max_sharpe = MDP.port_alloc("Max_Sharpe")
    port_value_max_div=MDP.port_value("Max_Div","Other",weights_max_div)
    port_value_min_vol=MDP.port_value("Min_Vol","Other",weights_min_vol)
    port_value_max_sharpe=MDP.port_value("Max_Sharpe","Other",weights_max_sharpe)


