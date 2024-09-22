from dataclasses import dataclass
from enum import Enum
from typing import List


class GrantType(Enum):
    ClientCredentials = 'client_credentials'
    PasswordCredentials = 'password'


class Scope(Enum):
    READ = "read"
    WRITE = "write"


@dataclass
class Credentials:
    grant_type: GrantType = None
    scope: List[Scope] = None
    pass


@dataclass
class ClientCredentials(Credentials):
    client_id: str = None
    client_secret: str = None

    def __post_init__(self):
        self.grant_type = GrantType.ClientCredentials


@dataclass
class PasswordCredentials(ClientCredentials):
    user: str = None
    password: str = None

    def __post_init__(self):
        self.grant_type = GrantType.PasswordCredentials


