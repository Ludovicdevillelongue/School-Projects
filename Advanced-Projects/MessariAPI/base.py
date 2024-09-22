from abc import ABCMeta, abstractmethod
from typing import Union

from authlib.integrations.requests_client import OAuth2Session
from requests import Response
from models.security import *


class BaseClient(metaclass=ABCMeta):
    base_url = None

    @property
    def credentials(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def session(self):
        raise NotImplementedError

    @abstractmethod
    def _get_url(self, route):
        raise NotImplementedError

    @property
    @abstractmethod
    def headers(self):
        raise NotImplementedError

    @staticmethod
    def _handle_response(response: Response):
        if response.status_code == 200 or response.status_code == 201:
            print(response.json())
            return response.json()
        else:
            print(response.json())

    def _get(self, route, **kwargs):
        url = self._get_url(route)
        print(f'request : {url}')
        response = self.session.get(url, headers=self.headers, **kwargs)
        return self._handle_response(response)


class Client(BaseClient):

    def __init__(self):
        self._credentials = None
        pass

    @property
    def credentials(self) -> Union[ClientCredentials, PasswordCredentials]:
        return self._credentials

    @credentials.setter
    def credentials(self, credentials):
        if isinstance(credentials, ClientCredentials) or isinstance(credentials, PasswordCredentials):
            self._credentials = credentials

    @property
    def headers(self):
        raise NotImplementedError

    @property
    def session(self):
        raise NotImplementedError

    def _get_url(self, url):
        return self.base_url + url


class OAuth2Client(Client):
    __token_uri = None

    def __init__(self, credentials: Union[ClientCredentials, PasswordCredentials]):
        super().__init__()
        self.credentials = credentials

    @property
    def headers(self):
        """
        Must be implement in subclasses to feed request headers
        Returns
        -------

        """
        raise NotImplementedError

    @property
    def token_uri(self):
        """

        Returns
        -------
            str
                the url for token request
        """
        return self.base_url + self.__token_uri

    @property
    def session(self) -> OAuth2Session:
        """

        Returns
        -------
            OAuth2Session
                the session with the fetch token to process http requests
        """
        _session = None

        if not issubclass(type(self.credentials), Credentials):
            raise TypeError(f'credentials should be an instance of subclasses from type Credentials')

        _session = OAuth2Session(
            self.credentials.client_id,
            self.credentials.client_secret,
            scope=[scope.value for scope in self.credentials.scope]

        )

        if isinstance(self.credentials, ClientCredentials):
            _session.fetch_token(url=self.token_uri)

        if isinstance(self.credentials, PasswordCredentials):
            _session.fetch_token(
                url=self.token_uri,
                username=self.credentials.password,
                password=self.credentials.password
            )
        return _session
