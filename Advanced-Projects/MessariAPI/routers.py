from fastapi import APIRouter
from messari_client import MessariClient


'''
----------------------------------------------------------------------------------------------------------
Classe permettant d'ajouter des get queries sur le router correspondant aux requêtes effectuées sur l'API. 
Les données sont sorties selon un format prédéfini.
----------------------------------------------------------------------------------------------------------
'''

class _QuoteRouter(APIRouter):

    def __init__(self, *args, **kwargs):
        super(_QuoteRouter, self).__init__(*args, **kwargs)
        self._prepare()

    def _prepare(self):
        @self.get("/news/{key}", tags=self.tags)
        def get_all_news(key: str):
            client = MessariClient(key)
            res = client.get_news()
            return res

        @self.get("/news/{symbol}/{key}", tags=self.tags)
        def get_news_per_asset(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_news_asset(symbol)
            return res

        @self.get("/markets/{key}", tags=self.tags)
        def get_all_markets(key: str):
            client = MessariClient(key)
            res = client.get_all_markets()
            return res

        @self.get("/assets/{key}/{limit}", tags=self.tags)
        def get_all_assets(key: str, limit: int):
            client = MessariClient(key)
            res = client.get_all_assets(limit)
            return res

        @self.get("/metrics/{symbol}/{key}", tags=self.tags)
        def get_asset_metrics(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_metrics(symbol)[0].to_view()
            return res

        @self.get("/metrics/{symbol}/market_data/{key}", tags=self.tags)
        def get_asset_market_data(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_market_data(symbol)[0].to_view()
            return res

        @self.get("/metrics/{symbol}/blockchain stats/{key}", tags=self.tags)
        def get_asset_blockchain_stats_24h(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_blockchain_stats_24h(symbol)[0].to_view()
            return res

        @self.get("/metrics/{symbol}/all time high/{key}", tags=self.tags)
        def get_asset_all_time_high(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_all_time_high(symbol)[0].to_view()
            return res

        @self.get("/metrics/{symbol}/dev act/{key}", tags=self.tags)
        def get_asset_dev_act(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_dev_act(symbol)[0].to_view()
            return res

        @self.get("/metrics/{symbol}/roi data/{key}", tags=self.tags)
        def get_asset_roi_data(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_roi_data(symbol)[0].to_view()
            return res

        @self.get("/metrics/{symbol}/roi by year/{key}", tags=self.tags)
        def get_asset_roi_by_year(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_roi_by_year(symbol)[0].to_view()
            return res

        @self.get("/metrics/market data/assets usd prices/{key}", tags=self.tags)
        def get_all_price_usd(key: str):
            client = MessariClient(key)
            res = client.get_all_price_usd()
            return res

        @self.get("/profile/{symbol}/overview/{key}", tags=self.tags)
        def get_asset_profile_general_overview(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_profile_general_overview(symbol)[0].to_view()
            return res

        @self.get("/profile/{symbol}/issuing organizations/{key}", tags=self.tags)
        def get_asset_issuing_organizations(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_issuing_organizations(symbol)[0].to_view()
            return res

        @self.get("/profile/{symbol}/individual contributors/{key}", tags=self.tags)
        def get_asset_individual_contributors(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_individual_contributors(symbol)[0].to_view()
            return res

        @self.get("/profile/{symbol}/known vulnerabilities/{key}", tags=self.tags)
        def get_asset_known_vulnerabilities(key: str, symbol: str):
            client = MessariClient(key)
            res = client.get_asset_known_vulnerabilities(symbol)
            if not res:
                return {'title': None, 'date': None, 'type': None, 'details': None}
            res = res[0].to_view()
            return res

# formattage des get
router = _QuoteRouter(prefix="/quotes", tags=["Messari-Queries"])