global print
from rich import print as _print

def print(*args, **kwargs):
	"""print with rich"""
	for arg in args:
		if isinstance(arg, dict | list):
			print(arg)
		else:
			_print(arg, **kwargs)
	return args

import os
from pathlib import Path
from pydantic import Field, BaseSettings, SecretStr
from rich import print
from typing import Optional

env_file = (Path(__file__).parent / '.env').expanduser()
print(f'env_file: {str(env_file)}')

class Env(BaseSettings):
	""" env = Env(); env holds variables from .env """
	test: str = Field('xyz', env= 'test') # example

	WANDB_USER: Optional[str]
	WANDB_PROJECT: Optional[str]
	WANDB_ENTITY: Optional[str]
	GITHUB_USER: Optional[str]
	GITHUB_REPO: Optional[str]
	GITHUB_TOKEN: Optional[SecretStr]

	class Config:
		env_file = env_file,
		env_file_encoding = 'utf-8'

global env
env = Env()
print(env)

from dotenv import load_dotenv
_ = load_dotenv(dotenv_path= env_file) # envs go to os.environ
print('environ test: ' + str('GITHUB_TOKEN' in os.environ))

from nn_nananana_ansatz.pyfig.log import create_logger
level = os.environ.get('log_level', 'INFO')
handler = create_logger(level= level, remove= True)
print(f'logger handler: {str(handler)}')
print(f'logger level: {str(level)}')
print('change level: os.environ["LOG: or "log"] = "LEVEL" or "level"')
print('levels: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL')