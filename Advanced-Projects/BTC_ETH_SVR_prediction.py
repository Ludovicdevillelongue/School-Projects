# -*- coding: utf-8 -*-
"""
@authors: Ludovic de Villelongue, Amine Mounazil, Emmanuel Zheng
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class WeightId:
    """
    This is a primary key for Weight at a given date
    -------
    Attributes
        - id (str)
        - product_code (List(str))
        - underlying_code (str)
        - dt (datetime a timestamp)

    """
    def __init__(self,
                 product_code: List[str] = None,
                 underlying_tickers: str = None,
                 dt: datetime = None
                 ):
        self.id = id
        self.product_code = product_code
        self.underlying_tickers = underlying_tickers
        self.dt = dt

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False

    def __hash__(self):
        return hash((self.product_code, self.dt, self.underlying_code))


class Weight:
    """
    This is a Weight at a given date
        Attributes
        -------
        - WeightId (str)
        - value (float)

    """
    def __init__(self,
                 id: WeightId,
                 value=float
                 ):
        self.id = id
        self.value = value

    def __repr__(self):
        return self.__dict__

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.id == other.id if isinstance(other, self.__class__) else False

    def __hash__(self):
        return hash(self.id)


class Params:
    """
    This contains the strategy parameters
    -------
         Attributes
         -------
        - start_ts (datetime)
        - end_ts (datetime)
        - underlying_ticker (str)
        - strategy_name (str)
        - weights (List[Weight])
    """
    def __init__(self,
                 start_ts: datetime,
                 end_ts: datetime,
                 underlying_tickers: List[str],
                 strategy_name: str,
                 weights: List[Weight]
                 ):
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.underlying_tickers = underlying_tickers
        self.strategy_name = strategy_name
        self.weights = weights

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return repr(self)


class Provider(Enum):
    """
    Class for list of data providers
    """
    YAHOO = "YAHOO"

class Data:
    def __repr__(self):
        kwargs = [f"{key}={value!r}" for key, value in self.__dict__.items() if key[0] != "_" or key[:2] != "__"]
        return "{}({})".format(type(self).__name__, "".join(kwargs))

    """
    - Reject keys with single underscore or double when input the variable length arguments de longueur containing 
    the keys
    - Attribute __name__ return the original name attributated to the class
    join allow the conversion of the dictionary of kwargs in string
    """


class QuoteId(Data):
    """
    This is a primary key for Quote object at a given date
    """
    def __init__(self,
                 product_code: List[str] = None,
                 dt: datetime = None,
                 provider: Provider = None
                 ):
        if isinstance(provider, Provider):
            self.provider = provider
        else:
            raise TypeError(f"self.provider must be an instance of Provider")
        self.product_code = product_code
        self.dt = dt

    """
     Parameters
     -------
        - product_code (List[str])
        - dt (datetime)
        - provider (str)
    """

    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ if isinstance(other, self.__class__) else False

    def __hash__(self):
        return hash((self.product_code, self.dt, self.provider))


class Quote(Data):
    """
    This object is a Quote and contains the id, open, high, low, close,adjclose and volume
    """
    def __init__(self,
                 id: QuoteId,
                 open: float = None,
                 high: float = None,
                 low: float = None,
                 close: float = None,
                 adjclose: float = None,
                 volume: int = None
                 ):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume
        self.id = id

    def __eq__(self, other):
        return self.id == other.id if isinstance(other, Quote) else False

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def from_json(cls, **kwargs):
        """
        This method will return a list of Quote for the period from a Json
        """
        product_code = params.underlying_tickers
        open = kwargs.get("Open")
        high = kwargs.get("High")
        low = kwargs.get("Low")
        close = kwargs.get("Close")
        adjclose = kwargs.get("Adj Close")
        volume = kwargs.get("Volume")
        dt = pd.to_datetime(end) - pd.to_datetime(begin)
        id = QuoteId(product_code, dt, Provider.YAHOO)
        return cls(open=open, high=high, low=low, close=close, adjclose=adjclose, volume=volume, id=id)

    @staticmethod
    def get_data(json_format: List[dict]):
        return list(map(lambda obj: Quote.from_json(**obj), json_format))
    """
    Parameters
    ----------
        json_format : List[dict]

    Returns
    -------
        result : list
    """

class QuoteView:
    """
    This object QuoteView will allow us to view the quotes
    """
    def __init__(self,
                 product_code: List[str] = None,
                 dt: datetime = None,
                 provider: Provider = None,
                 open: float = None,
                 high: float = None,
                 low: float = None,
                 close: float = None,
                 adjclose: float = None,
                 volume: int = None
                 ):
        self.product_code = product_code
        self.dt = dt
        self.provider = provider
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume

    """
    Parameters
    ----------
        product_code: List[str]
        dt: datetime
        provider: Provider from class Provider
        open: float
        high: float
        low: float
        close: float
        adjclose: float
        volume: int
    """

    @classmethod
    def from_quote(cls, quote: Quote):
        return cls(product_code=quote.id.product_code,
                   dt=quote.id.dt,
                   provider=quote.id.provider,
                   open=quote.open,
                   high=quote.high,
                   low=quote.low,
                   close=quote.close,
                   adjclose=quote.adjclose,
                   volume=quote.volume)

    """
    This method will give us a list of Quote from Quote
    """

class Factory:
    """
    This Object will give us the list of quotes from the Object QuoteView
    """
    @staticmethod
    def to_quote_view(quotes: list([Quote])):
        return list(map(lambda quote: QuoteView.from_quote(quote), quotes))


class DataJson:
    def __init__(self, df_select):
        self.df_select = df_select
    """
    Parameters
    ----------
        df_select : a dataframe

    Returns
    -------
        list of Quote
    """
    @staticmethod
    def convert_data(df_select):
        df_T = df_select.T
        dict_opt = df_T.groupby(level=0).apply(lambda df: df.xs(df.name).to_dict(orient="index")).to_dict()
        json_format = [value for value in dict_opt.values()]
        quote_list = Quote.get_data(json_format)
        quote_1D = Factory.to_quote_view(quote_list)
        return quote_1D


class Indicators:
    """
    Class Indicators to compute the indicators:
    Functions that we call in TA.compute_signals(), they have as
    an input a pandas dataframes and as an output a pandas dataframe
    that we feed to our model
    """
    @staticmethod
    def compute_sma(df, period=10, target="SMA"):
        """
            Function to compute the Simple Moving Average (SMA).
            In SMA, each value in the time period carries equal weight.
            They do not predict price direction, but can be used to identify the direction of the trend
            or define potential support and resistance levels.
            SMA = (P1 + P2 + ... + Pn) / K
            where K = n and Pn is the most recent price

            Parameters
            -------
                - df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
                - close : String indicating the column name from which the SMA needs to be computed from
                - target : String indicates the column name to which the computed data needs to be stored
                - period : Integer indicates the period of computation in terms of number of candles

            Returns
            -------
                df : Pandas DataFrame with new column added with name 'target'
            """

        num_prices = len(df)
        if num_prices < period:
            # show error message
            raise SystemExit('Error: The length of the set should be greater than the chosen period.')
        df[target] = df.rolling(window=period).mean()
        df[target].fillna(0, inplace=True)
        return df[target]

    @staticmethod
    def compute_wma(df, period=10, target="WMA"):
        """
        Function to compute the Weighted Moving Average (WMA). WMA is a type of moving average that assigns a
        higher weighting to recent price data.
        WMA = (P1 + 2 P2 + 3 P3 + ... + n Pn) / K
        where K = (1+2+...+n) = n(n+1)/2 and Pn is the most recent price after the
        1st WMA we can use another formula
        WMAn = WMAn-1 + w.(Pn - SMA(prices, n-1))
        where w = 2 / (n + 1)

        Parameters
        -------
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            close : String indicating the column name from which the WMA needs to be computed from
            target : String indicates the column name to which the computed data needs to be stored
            period : Integer indicates the period of computation in terms of number of candles

        Returns
        -------
            df : Pandas DataFrame with new column added with name 'target'
        """

        num_prices = len(df)
        if num_prices < period:
            # show error message
            raise SystemExit('Error: The length of the set should be greater than the chosen period.')

        weights = np.arange(1, period + 1)
        df[target] = df["Close"].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
        df[target] = df[target].dropna()
        return df[target]

    @staticmethod
    def compute_rsi(df, period=10, target="RSI"):
        """
        Function to compute the Relative Strength Index (RSI). The Compares the magnitude of
        recent gains and losses over a specified time period to measure speed and change
        of price movements of a security. It is primarily used to attempt to identify overbought or
        oversold conditions in the trading of an asset.

        Parameters
        -------
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            close : String indicating the column name from which the RSI needs to be computed from (Default Close)
            target : String indicates the column name to which the computed data needs to be stored
            period : Integer indicates the period of computation in terms of number of candles

        Returns
        -------
            df : Pandas DataFrame with new columns added for Relative Strength Index (RSI_$period)
        """

        num_prices = len(df)
        if num_prices < period:
            # show error message
            raise SystemExit('Error: The length of the set should be greater than the chosen period.')

        df_rsi = df["Close"].diff()
        up, down = df_rsi.copy(), df_rsi.copy()

        up[up < 0] = 0
        down[down > 0] = 0

        rUp = up.ewm(com=period - 1, adjust=False).mean()
        rDown = down.ewm(com=period - 1, adjust=False).mean().abs()

        df[target] = 100 - 100 / (1 + rUp / rDown)
        df[target].fillna(0, inplace=True)

        return df[target]

    @staticmethod
    def compute_ado(df, period=10, target="ADO"):
        """
       Function to compute the Accumulation/Distribution Oscillator. The ADO is a cumulative indicator that uses
       volume and price to assess whether a stock is being accumulated or distributed. The A/D measure seeks to
       identify divergences between the stock price and the volume flow. This provides insight into how strong a
       trend is. If the price is rising but the indicator is falling, then it suggests that buying or accumulation
       volume may not be enough to support the price rise and a price decline could be forthcoming.

        Parameters
        -------
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            high : String indicating the column name from which the ADO needs to be computed from
            low : String indicating the column name from which the ADO needs to be computed from
            close : String indicating the column name from which the ADO needs to be computed from
            volume : String indicating the column name from which the ADO needs to be computed from
            target : String indicates the column name to which the computed data needs to be stored
            period : Integer indicates the period of computation in terms of number of candles

        Returns
        -------
            df : Pandas DataFrame with new columns added for Relative Strength Index (ADO)

        """
        num_prices = len(df["Close"])
        if num_prices < period:
            # show error message
            raise SystemExit('Error: The length of the set should be greater than the chosen period.')

        ac = []
        df_ado = df
        for index, row in df_ado.iterrows():
            if row["High"] == row["Low"]:
                ac.append(0)
            else:
                ac.append(
                    ((row["Close"] - row["Low"]) - (row["High"] - row["Close"])) / (row["High"] - row["Low"]) * row[
                        "Volume"])
            df_ado = pd.Series(ac)
        df[target] = df_ado.ewm(ignore_na=False, min_periods=0, com=period, adjust=True).mean().values
        return df[target]

    @staticmethod
    def compute_atr(df, period=10, target="ATR"):
        """
        Function to compute the Average True Range (ATR). The true range indicator is taken as the greatest
        of the following: current high less the current low; the absolute value of the current high less the
        previous close; and the absolute value of the current low less the previous close. The ATR is then a
        moving average, generally using 14 days, of the true ranges. The ATR provides an indication of the
        degree of price volatility. Strong moves, in either direction, are often accompanied by large ranges,
        or large True Ranges.

        Parameter
        -------
            df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
            close : String indicating the column name from which the ATR needs to be computed from
            high : String indicating the column name from which the ATR needs to be computed from
            low : String indicating the column name from which the ATR needs to be computed from
            target : String indicates the column name to which the computed data needs to be stored
            period : Integer indicates the period of computation in terms of number of candles

        Returns
        -------
            df : Pandas DataFrame with new columns added for the Average True Range (ATR)
        """

        num_prices = len(df["Close"])
        if num_prices < period:
            # show error message
            raise SystemExit('Error: The length of the set should be greater than the chosen period.')

        TRL = [0]
        for element in range(len(df.index) - 1):
            TR = max(df["High"][element + 1], df["Close"][element]) - min(df["Low"][element + 1], df["Close"][element])
            TRL.append(TR)

        TR_s = pd.Series(TRL)
        df[target] = pd.Series.ewm(TR_s, span=period, min_periods=period).mean().values
        return df[target]


class TA(Data):
    """
    TA is a class that instantiates the indicators, reads the data from quote and then passes it
    to methods created in the class Indicators for computation of the indicators
    """
    def __init__(self,
                 id: QuoteId,
                 open: float = None,
                 high: float = None,
                 low: float = None,
                 close: float = None,
                 adjclose: float = None,
                 volume: int = None,
                 sma: float = None,
                 wma: float = None,
                 rsi: float = None,
                 ado: float = None,
                 atr: float = None):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.adjclose = adjclose
        self.volume = volume
        self.id = id
        self.sma = sma
        self.wma = wma
        self.rsi = rsi
        self.ado = ado
        self.atr = atr

    def __eq__(self, other):
        return self.id == other.id if isinstance(other, Quote) else False

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def from_json(cls, **kwargs):
        product_code = params.underlying_tickers
        open = kwargs.get("Open")
        high = kwargs.get("High")
        low = kwargs.get("Low")
        close = kwargs.get("Close")
        adjclose = kwargs.get("Adj Close")
        volume = kwargs.get("Volume")
        dt = pd.to_datetime(end) - pd.to_datetime(begin)
        id = QuoteId(product_code, dt, Provider.YAHOO)
        return cls(open=open, high=high, low=low, close=close, adjclose=adjclose, volume=volume, id=id)

    @staticmethod
    def get_data(json_format: List[dict]):
        """
        Parameters
        ----------
            json_format : List[dict]

        Returns
        -------
            result : a list of object quote
        """
        return list(map(lambda obj: Quote.from_json(**obj), json_format))

    @staticmethod
    def compute_indicators(dict_close: dict,
                           dict_high: dict,
                           dict_low: dict,
                           dict_volume: dict):
        """
        Parameters
        ----------
            dict_close: dict,
            dict_high: dict,
            dict_low: dict,
            dict_volume: dict

        Returns
        -------
            result : dataframe df of all indicators : sma, wma, rsi, atr, ado, df
        """
        df_close = pd.DataFrame.from_dict(dict_close, orient="index", columns=["Close"])
        df_high = pd.DataFrame.from_dict(dict_high, orient="index", columns=["High"])
        df_low = pd.DataFrame.from_dict(dict_low, orient="index", columns=["Low"])
        df_volume = pd.DataFrame.from_dict(dict_volume, orient="index", columns=["Volume"])
        df_ado = pd.concat([df_close, df_low, df_high, df_volume], axis=1)
        df_atr = pd.concat([df_close, df_low, df_high, ], axis=1)

        sma = Indicators.compute_sma(df_close)
        wma = Indicators.compute_wma(df_close)
        rsi = Indicators.compute_rsi(df_close)
        atr = Indicators.compute_atr(df_atr)
        ado = Indicators.compute_ado(df_ado)
        df = pd.concat([sma, wma, rsi, ado, atr], axis=1) #Concat of all indicators df
        return df


class Model:

    def data_preprocess(dict_close: dict, df: pd.DataFrame):
        """
        Parameters
        ----------
            dict_close: dict,
            df: pd.DataFrame,

        Returns
        -------
            result : dict_pred : dict of predicted prices
        """

        #Import the Quote close
        prices = pd.DataFrame.from_dict(dict_close, orient="index", columns=["Close"]).iloc[9:, :]
        indicators = df

        #Concat Quote close and indicators
        dataset = pd.concat([prices, indicators], axis=1)

        #Define X
        X = dataset.iloc[:,1:6]
        X = np.asarray(X)

        #Define Y
        y = dataset.iloc[:,0:1]
        y = np.asarray(y)

        #Use sklearn train_test_split package to easily split the data into training sets and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        """Standardization scales each input variable separately by subtracting the mean (called centering) and 
        dividing by the standard deviation to shift the distribution to have a mean of zero and a 
        standard deviation of one.
        """
        # Standard scaler sklearn package
        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)
        X_test = standard_scaler.transform(X_test)

        # SVR sklearn package with Radial basis function kernel
        regressor = SVR(kernel='rbf', C=1000, epsilon=0.001)
        regressor.fit(X_train, y_train)

        # output y_pred and put it in a dataframe
        y_pred = regressor.predict(X_test).reshape((-1,1))
        df_predicted = pd.DataFrame({'Predicted Values' : y_pred.reshape(-1)})
        split = int(len(dataset) * 0.8)

        df_indexed=df_predicted.set_index(dataset.iloc[split:,:].index)
        dict_pred=pd.DataFrame.to_dict(df_indexed, orient='index')
        return dict_pred


class SignalGenerator:
    """

    Parameters
    ----------
        dict_pred: dict,
        dict_actual: pd.dict,
    """
    def __init__(self,
                 dict_pred: dict,
                 dict_actual: dict
                 ):
                self.dict_pred = dict_pred
                self.dict_actual = dict_actual

    @staticmethod
    def comparison(dict_actual: dict, dict_pred: dict):
        """
        This method compares our model predicted output (close quote) to the actual and decide to buy or sell
        Parameters
        ----------
            dict_pred: dict,
            dict_actual: pd.dict,

        Returns
        -------
            result : a dataframe trading_signals
        """
        # Initialize the comparison
        df_actual = pd.DataFrame.from_dict(dict_actual, orient="index")
        df_actual_ta=df_actual.iloc[9:,:]
        split = int(len(df_actual_ta) * 0.8)
        df_actual_fit=df_actual_ta.iloc[split:,:]
        df_pred = pd.DataFrame.from_dict(dict_pred, orient="index")

        # Initialize the `signals` DataFrame with the `signal` column
        signals = pd.DataFrame(index=df_pred.index)
        signals['signal'] = 0.0

        # Create short simple moving average over the short window
        signals['actual_price'] = df_actual_fit

        # Create long simple moving average over the long window
        signals['predicted_price'] = df_pred

        # Create signals
        signals['signal'] = np.where(signals['actual_price'] < signals['predicted_price'], 1.0, 0.0)

        # Generate trading orders
        trading_signals = signals['signal']
        return trading_signals


class Backtest(object):
    """
    Simple vectorized backtester. Works with pandas objects.

    """

    def __init__(self, prices: dict, trading_signal: Series, initialcash : float = 1000):
        """
        Parameters
        -----------
        prices :  instrument price
        trading_signal : capital to invest (long+,short-) or number of shares
        initialcash : float = 1000 starting cash

        """

        # first thing to do is to clean up the signal, removing nans and duplicate entries or exits
        self.trades = trading_signal.diff()

        # now create internal data structure
        split = int((len(prices) -9)* 0.8)
        self.prices = pd.DataFrame.from_dict(prices, orient="index").iloc[split+9:,:]
        self.data = pd.DataFrame(index=self.prices.index, columns=['prices', 'shares', 'value', 'cash', 'pnl'])
        self.data['prices'] = self.prices

        self.data['shares'] = self.trades.fillna(0)
        self.data['value'] = self.data['shares'] * self.data['prices'].fillna(0)

        delta = self.data['shares'].diff() # shares bought sold

        self.data['cash'] = (-delta * self.data['prices']).fillna(0).cumsum() + initialcash
        self.data['pnl'] = (self.data['cash'] + self.data['value'] - initialcash).fillna(0)

    @property
    def sharpe(self):
        """ return annualized sharpe ratio of the pnl """
        pnl = (self.data['pnl'].diff()).shift(-1)[self.data['shares'] != 0]  # use only days with position.
        return np.sqrt(250) * pnl.mean() / pnl.std()  # need the diff here as sharpe works on daily returns.

    @property
    def pnl(self):
        """ easy access to pnl data column """
        return self.data['pnl']

    @property
    def plottrades(self):
        """
        visualise trades on the price chart
        long entry : green triangle up
        short entry : red triangle down
        exit : black circle
        """
        legend = ['prices']

        p = self.data['prices']
        p.plot(style='x-')

        # --- plot trades
        # colored line for long positions
        idx = (self.data['shares'] > 0) | ((self.data['shares'] > 0).shift(1))
        if idx.any():
            p[idx].plot(style='go')
            legend.append('long')

        # colored line for short positions
        idx = (self.data['shares'] < 0) | ((self.data['shares'] < 0).shift(1))
        if idx.any():
            p[idx].plot(style='ro')
            legend.append('short')

        plt.xlim([p.index[0], p.index[-1]])  # show full axis

        plt.legend(legend, loc='best')
        plt.title('trades')
        return plt.show()


if __name__ == '__main__':

    #Parameters
    end = datetime.now(timezone.utc) - timedelta(days=0)
    begin = end - timedelta(days=1500)
    tickers = ["BTC-USD","ETH-USD"]
    tickers.sort()
    params = Params(begin, end, tickers, "SVR", [Weight(WeightId())])
    df_api = yf.download(params.underlying_tickers, params.start_ts, params.end_ts, group_by="ticker", interval="1d")
    List_Quote = DataJson.convert_data(df_api)

    # Loop for more list of more than 1 ticker
    for i in range(len(List_Quote)):
        #TA indicators not significant for the first 10 values
        df_indicators = TA.compute_indicators(List_Quote[i].close,
                                              List_Quote[i].open,
                                              List_Quote[i].high,
                                              List_Quote[i].volume).fillna(0).iloc[9:, :]

        model_input = Model.data_preprocess(List_Quote[i].close, df_indicators)
        signal = SignalGenerator.comparison(List_Quote[i].close, model_input)
        backtester = Backtest(List_Quote[i].close, signal)
        print(backtester.data, backtester.plottrades)


