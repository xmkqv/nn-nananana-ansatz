global print
from rich import print as _print
from rich.pretty import pprint

def print(*args, **kwargs):
    """print with rich"""
    for arg in args:
        if isinstance(arg, dict | list):
            pprint(arg)
        else:
            _print(arg, **kwargs)
    return args

import os
from pathlib import Path
from pydantic import BaseModel, Field, BaseSettings, SecretStr
from rich import print
from rich.pretty import pprint
from functools import partial
from typing import Optional, Callable

env_file = (Path(__file__).parent.parent / '.env').expanduser()
print(f'env_file: {str(env_file)}')

class Env(BaseSettings):
    """ env = Env(); env holds variables from .env """
    test: str = Field('xyz', env= 'test') # example

    # DOCKER_API_KEY: SecretStr
    # DOCKER_USER: SecretStr
    # GITLAB_TOKEN: SecretStr

    WANDB_USER: Optional[str]
    WANDB_PROJECT: Optional[str]
    WANDB_ENTITY: Optional[str]
    GITHUB_USER: Optional[str]
    GITHUB_REPO: Optional[str]
    GITHUB_TOKEN: Optional[SecretStr]
    DOCKER_API_KEY: Optional[SecretStr]
    DOCKER_USER: Optional[str]
    GITLAB_TOKEN: Optional[SecretStr]

    class Config:
        env_file = env_file,
        env_file_encoding = 'utf-8'
env = Env()
pprint(env)

from dotenv import load_dotenv
_ = load_dotenv(dotenv_path= env_file) # envs go to os.environ
pprint('environ test: ' + str('GITHUB_TOKEN' in os.environ))

from .pyfig.log import create_logger
level = os.environ.get('log_level', 'INFO')
handler = create_logger(level= level, remove= True)
print(f'logger handler: {str(handler)}')
print(f'logger level: {str(level)}')
print('change level: os.environ["LOG: or "log"] = "LEVEL" or "level"')
print('levels: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL')

from . import utils