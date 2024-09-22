from base import BaseClient
from requests import Session
from models.messari import *

"""
    By Stefania Kukovski & Ludovic de Villelongue
"""

'''
-----------------------------------------------------------------------
Classe permettant d'appeler l'API pour récupérer les données souhaitées 
-----------------------------------------------------------------------
'''


class MessariClient(BaseClient):
    base_url = "https://data.messari.io/api"

    def __init__(self, credentials: str):
        self._credentials = credentials

    @property
    def credentials(self):
        raise self._credentials

    @property
    def session(self):
        return Session()

    def _get_url(self, route):
        return self.base_url + route

    @property
    def headers(self):
        return {"x-messari-api-key": self._credentials, "Content-Type": "application/json"}

    # requête des dernières actualités sur l'ensemble des cryptos
    def get_news(self) -> dict:
        response = self._get(f"/v1/news")
        data = response.get("data")
        new_dict = [{item['title']: item} for item in data]
        return new_dict

    # requête des dernières actualités d'une crypto
    def get_news_asset(self, asset_key: str) -> dict:
        response = self._get(f"/v1/news/{asset_key}")
        data = response.get("data")
        new_dict = [{asset_key + " News id " + item['id'] : item} for item in data]
        return new_dict

    # requête des caractéristiques de marché sur l'ensemble des cryptos
    def get_all_markets(self) -> dict:
        response = self._get(f"/v1/markets")
        data = response.get("data")
        new_dict = [{"Markets id " + item['id']: item} for item in data]
        return new_dict

    # requête de l'ensemble des caractéristiques d'un certain nombre de crypto
    def get_all_assets(self, limit) -> dict:
        if 20 <= limit <= 500:
            response = self._get(f"/v2/assets?limit="+str(limit))
            data = response.get("data")
            new_dict = [{item['slug']: item} for item in data]  # permits to print for each asset their information
            return new_dict
        else:
            raise ValueError(f"limit must between 20 and 500")


    # requête de l'ensemble des caractéristiques d'une crypto
    def get_asset_metrics(self, symbol) -> List[MessariAssetMetrics]:
        response = self._get(f"/v1/assets/{symbol}/metrics")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetMetrics.from_json(**x), data))

    # requête des prix et caractéristiques de prix d'une crypto
    def get_asset_market_data(self, symbol) -> List[MessariAssetMarketData]:
        response = self._get(f"/v1/assets/{symbol}/metrics/market-data")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetMarketData.from_json(**x), data))

    # requête des données relatives à la blockchain pour chaque crypto
    def get_asset_blockchain_stats_24h(self, symbol) -> List[MessariAssetBlockchainStats]:
        response = self._get(f"/v1/assets/{symbol}/metrics?fields=id,slug,symbol,blockchain_stats_24_hours")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetBlockchainStats.from_json(**x), data))

    # requête du prix au plus haut d'une crypto
    def get_asset_all_time_high(self, symbol) -> List[MessariAssetAllTimeHigh]:
        response = self._get(f"/v1/assets/{symbol}/metrics?fields=id,slug,symbol,all_time_high")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetAllTimeHigh.from_json(**x), data))

    # requête du prix au plus haut d'une crypto
    def get_asset_dev_act(self, symbol) -> List[MessariAssetDevAct]:
        response = self._get(f"/v1/assets/{symbol}/metrics?fields=id,slug,symbol,developer_activity")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetDevAct.from_json(**x), data))

    # requête des caractéristiques du ROI d'une crypto
    def get_asset_roi_data(self, symbol) -> List[MessariAssetRoiData]:
        response = self._get(f"/v1/assets/{symbol}/metrics?fields=id,slug,symbol,roi_data")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetRoiData.from_json(**x), data))

    # requête de l'historique du ROI par année d'une crypto
    def get_asset_roi_by_year(self, symbol) -> List[MessariAssetRoiByYear]:
        response = self._get(f"/v1/assets/{symbol}/metrics?fields=id,slug,symbol,roi_by_year")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetRoiByYear.from_json(**x), data))

    # requête du prix actuel de l'ensemble des cryptos
    def get_all_price_usd(self) -> dict:
        response = self._get(f"/v2/assets?fields=id,slug,symbol,metrics/market_data/price_usd")
        data = response.get("data")
        new_dict = [{item['slug']:item} for item in data]
        return new_dict

    # requête des informations générales et sites web relatifs à une cypto
    def get_asset_profile_general_overview(self, symbol) -> List[MessariAssetGeneralOverviewData]:
        response = self._get(f"/v2/assets/{symbol}/profile?fields=id,name,profil")
        data = [response.get("data")]
        return list(map(lambda x: MessariAssetGeneralOverviewData.from_json(**x), data))

    # requête du descriptif de l'entreprise à l'origine d'une crypto
    def get_asset_issuing_organizations(self, symbol) -> List[MessariAssetIssuingOrganizations]:
        response = self._get(f"/v2/assets/{symbol}/profile?fields=id,name,profile/general/background/"
                             f"issuing_organizations")
        data = response.get("data")['profile']['general']['background']['issuing_organizations']
        return list(map(lambda x: MessariAssetIssuingOrganizations.from_json(**x), data))

    # requête d'informations sur le créateur ou principal contributeur d'une crypto
    def get_asset_individual_contributors(self, symbol) -> List[MessariAssetIndivContributors]:
        response = self._get(f"/v2/assets/{symbol}/profile?fields=id,name,profile/contributors/individuals")
        data = response.get("data")['profile']['contributors']['individuals']
        return list(map(lambda x: MessariAssetIndivContributors.from_json(**x), data))

    # requête des vulnérabilités découvertes d'une crypto
    def get_asset_known_vulnerabilities(self, symbol) -> List[MessariAssetKnownVulnerabilities]:
        response = self._get(f"/v2/assets/{symbol}/profile?fields=id,name,profile/technology/security/"
                             f"known_exploits_and_vulnerabilities")
        data = response.get("data")['profile']['technology']['security']['known_exploits_and_vulnerabilities']
        return list(map(lambda x: MessariAssetKnownVulnerabilities.from_json(**x), data))


# utilisation de la clé et choix du ticker pour les fonctions relatives à une crypto
if __name__ == '__main__':
    client = MessariClient("0e22d837-7111-4e66-a5c0-096bdab0806e")
    asset_symbol = "btc"

    """
    -----------------------------
    Choix des fonctions à appeler
    -----------------------------
    """
    # client.get_news()
    # client.get_news_asset(asset_symbol)
    # client.get_all_markets()
    # client.get_all_assets(21)
    # client.get_asset_metrics(asset_symbol)
    # client.get_asset_market_data(asset_symbol)
    # client.get_asset_blockchain_stats_24h(asset_symbol)
    # client.get_asset_all_time_high(asset_symbol)
    # client.get_asset_roi_data(asset_symbol)
    # client.get_asset_roi_by_year(asset_symbol)
    # client.get_all_price_usd()
    # client.get_asset_profile_general_overview(asset_symbol)
    # client.get_asset_issuing_organizations(asset_symbol)
    # client.get_asset_individual_contributors(asset_symbol)
    # client.get_asset_known_vulnerabilities(asset_symbol)
    client.get_asset_dev_act(asset_symbol)

