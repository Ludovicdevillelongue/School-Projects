from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple


class MissingValueMessari(Exception):
    def __init__(self, *args, **kwargs):
        pass


class NestedObject(metaclass=ABCMeta):

    @abstractmethod
    def to_view(self):
        """
        this property must be implemented in subclasses
        Returns
        -------
            object
        """
        raise NotImplementedError


class Embedded(type):
    def __new__(mcs, *args):
        return super().__new__(mcs, *args)


@dataclass
class MessariNews(NestedObject):
    id: str=None,
    title: str=None,
    content: str=None,
    references: List[dict]=None,
    reference_title: str=None,
    published_at: datetime=None,
    author: dict=None,
    tags: List[str]=None,
    url: str=None

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariMarkets(NestedObject):
    id: str=None,
    exchange_id: str=None,
    base_asset_id: str=None,
    quote_asset_id: str=None,
    trade_start: datetime=None,
    trade_end: datetime=None,
    version: int=None,
    excluded_from_price: bool=None,
    exchange_name: str=None,
    exchange_slug: str=None,
    base_asset_symbol: str=None,
    quote_asset_symbol: str=None,
    pair: str=None,
    price_usd: float=None,
    vwap_weight: int=None,
    volume_last_24_hours: float=None
    has_real_volume: bool=None,
    deviation_from_vwap_percent: float=None,
    last_trade_at: datetime=None

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAssetMetrics(NestedObject):
    id: str=None,
    serial_id : int=None,
    symbol: str=None,
    name: str=None,
    slug: str=None,
    contract_addresses: Tuple[None] = None,
    _internal_temp_agora_id: str=None,
    market_data: dict=None,
    ohlcv_last_24_hour:dict=None,
    marketcap : dict=None,
    supply: dict=None,
    blockchain_stats_24_hours: dict=None,
    market_data_liquidity: dict=None,
    all_time_high: dict=None,
    cycle_low: dict=None,
    token_sale_stats: dict=None,
    mining_stats: dict=None
    developer_activity: dict=None,
    roi_data: dict=None,
    roi_by_year: dict=None,
    risk_metrics: dict=None,
    misc_data: dict=None
    reddit: dict=None,
    on_chain_data: dict=None,
    exchange_flows : dict=None,
    miner_flows: dict=None,
    supply_activity: dict=None,
    supply_distribution: dict=None,
    alert_messages : Tuple[None]= None

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAllUsdPrices(NestedObject):
    slug: str = None,

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAllAssets(NestedObject):
    id: str = None,
    serial_id: int = None,
    symbol: str = None,
    name: str = None,
    slug: str = None,
    contract_addresses: Tuple[None] = None,
    _internal_temp_agora_id: str = None,
    metrics: dict = None,
    profile: dict = None

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAssetMarketData(NestedObject):
    @dataclass
    class _MarketData(metaclass=Embedded):
        price_usd: float = None,
        price_btc: float = None,
        price_eth: float = None,
        volume_last_24_hours: float = None,
        real_volume_last_24_hours: float = None,
        volume_last_24_hours_overstatement_multiple: float = None,
        percent_change_usd_last_1_hour: float = None,
        percent_change_btc_last_1_hour: float = None,
        percent_change_eth_last_1_hour: float = None,
        percent_change_usd_last_24_hours: float = None,
        percent_change_btc_last_24_hours: float = None,
        percent_change_eth_last_24_hours: float = None,
        ohlcv_last_1_hour: dict = None,
        ohlcv_last_24_hour: dict = None,
        last_trade_at: datetime = None

        @classmethod
        def from_json(cls, **kwargs):
            return cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})

    id: str = None,
    serial_id: int=None,
    symbol: str = None,
    name: str=None,
    slug: str = None,
    contract_addresses: Tuple[None] = None,
    _internal_temp_agora_id: str=None,
    market_data: _MarketData = None,

    @classmethod
    def from_json(cls, **kwargs):
        print(kwargs)
        market_data = kwargs.pop("market_data")
        if market_data is None:
            raise MissingValueMessari(f"Missing market_data for {kwargs.get('symbol')}")
        obj = cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})
        obj.market_data = cls._MarketData.from_json(**market_data)
        print(obj)
        return obj

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            if isinstance(type(value), Embedded):
                _view.update(**value.__dict__)
            else:
                _view.update({name: value})
        return _view


@dataclass
class MessariAssetBlockchainStats(NestedObject):
    @dataclass
    class _BlockchainStats(metaclass=Embedded):
        count_of_active_addresses: int = None,
        transaction_volume: float = None,
        adjusted_transaction_volume: float = None,
        adjusted_nvt: float = None,
        median_tx_value: float = None,
        median_tx_fee: float = None,
        count_of_tx: int = None,
        count_of_payments: int = None,
        new_issuance: float = None,
        average_difficulty: float = None,
        kilobytes_added: float = None,
        count_of_blocks_added: int = None

        @classmethod
        def from_json(cls, **kwargs):
            return cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})

    id: str = None,
    slug: str=None,
    symbol: str = None,
    blockchain_stats_24h: _BlockchainStats = None,

    @classmethod
    def from_json(cls, **kwargs):
        print(kwargs)
        blockchain_stats_24h = kwargs.pop("blockchain_stats_24_hours")
        if blockchain_stats_24h is None:
            raise MissingValueMessari(f"Missing blockchain_stats_24h for {kwargs.get('symbol')}")
        obj = cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})
        obj.blockchain_stats_24h = cls._BlockchainStats.from_json(**blockchain_stats_24h)
        return obj

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            if isinstance(type(value), Embedded):
                _view.update(**value.__dict__)
            else:
                _view.update({name: value})
        return _view


@dataclass
class MessariAssetAllTimeHigh(NestedObject):
    @dataclass
    class _AllTimeHigh(metaclass=Embedded):
        price: float = None,
        at: datetime = None
        days_since: int = None,
        percent_down: float = None,
        breakeven_multiple: float = None

        @classmethod
        def from_json(cls, **kwargs):
            return cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})

    id: str = None,
    slug: str = None,
    symbol: str = None,
    all_time_high: _AllTimeHigh = None

    @classmethod
    def from_json(cls, **kwargs):
        all_time_high = kwargs.pop("all_time_high")
        if all_time_high is None:
            raise MissingValueMessari(f"Missing all_time_high for {kwargs.get('symbol')}")
        obj = cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})
        obj.all_time_high = cls._AllTimeHigh.from_json(**all_time_high)
        return obj

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            if isinstance(type(value), Embedded):
                _view.update(**value.__dict__)
            else:
                _view.update({name: value})
        return _view



@dataclass
class MessariAssetDevAct(NestedObject):
    @dataclass
    class _DevAct(metaclass=Embedded):
        stars:int=None
        watchers:int=None
        @classmethod
        def from_json(cls, **kwargs):
            return cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})

    id: str = None,
    slug: str = None,
    symbol: str = None,
    developer_activity: _DevAct = None

    @classmethod
    def from_json(cls, **kwargs):
        developer_activity= kwargs.pop("developer_activity")
        if developer_activity is None:
            raise MissingValueMessari(f"Missing developer_activity for {kwargs.get('symbol')}")
        obj = cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})
        obj.developer_activity = cls._DevAct.from_json(**developer_activity)
        return obj

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            if isinstance(type(value), Embedded):
                _view.update(**value.__dict__)
            else:
                _view.update({name: value})
        return _view


@dataclass
class MessariAssetRoiData(NestedObject):
    @dataclass
    class _RoiData(metaclass=Embedded):
        percent_change_last_1_week: float=None,
        percent_change_last_1_month: float=None,
        percent_change_last_3_months: float=None,
        percent_change_last_1_year: float=None,
        percent_change_btc_last_1_week: float=None,
        percent_change_btc_last_1_month: float=None,
        percent_change_btc_last_3_months: float=None,
        percent_change_btc_last_1_year: float=None,
        percent_change_eth_last_1_week: float=None,
        percent_change_eth_last_1_month: float=None,
        percent_change_eth_last_3_months: float=None,
        percent_change_eth_last_1_year: float=None,
        percent_change_month_to_date: float=None,
        percent_change_quarter_to_date: float=None,
        percent_change_year_to_date: float=None

        @classmethod
        def from_json(cls, **kwargs):
            return cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})

    id: str = None,
    slug: str = None,
    symbol: str = None,
    roi_data: _RoiData = None

    @classmethod
    def from_json(cls, **kwargs):
        roi_data = kwargs.pop("roi_data")
        if roi_data is None:
            raise MissingValueMessari(f"Missing roi_data for {kwargs.get('symbol')}")
        obj = cls(**{key: kwargs.get(key) for key in cls.__dict__.keys() & kwargs.keys()})
        obj.roi_data = cls._RoiData.from_json(**roi_data)
        return obj

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            if isinstance(type(value), Embedded):
                _view.update(**value.__dict__)
            else:
                _view.update({name: value})
        return _view


@dataclass
class MessariAssetRoiByYear(NestedObject):
    id: str = None,
    slug: str = None,
    symbol: str = None,
    roi_by_year: dict = None  # keys are 2011_usd_percent, 2012_usd_percent etc. They begin by a digit so we can't
    # really define attributes and so an embedded class

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAssetGeneralOverviewData(NestedObject):
    profile: str = None,


    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAssetIssuingOrganizations(NestedObject):
    slug: str = None,
    name: str = None,
    logo: str = None,
    description: str = None

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAssetIndivContributors(NestedObject):
    slug: str = None,
    first_name: str = None,
    last_name: str = None,
    title: str = None,
    description: str = None
    avatar_url: str = None

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view


@dataclass
class MessariAssetKnownVulnerabilities(NestedObject):
    title: str = None,
    date: datetime = None,
    type: str = None,
    details: str = None

    @classmethod
    def from_json(cls, **kwargs):
        return cls(**kwargs)

    def to_view(self):
        _view = {}
        for name, value in self.__dict__.items():
            _view.update({name: value})
        return _view
