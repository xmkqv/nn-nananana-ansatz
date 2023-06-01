from pathlib import Path
from pydantic import BaseModel, Field
from rich import print
from uuid import uuid4
import os
from .utils import gen_rnd

class Paths(BaseModel):

	project: str
	exp_name: str
	run_name: Path				= Path('./run.py')

	lo_ve_path: 		str 	= '' # for LOad & saVE -> lo_ve
	exp_id: 			str 	= Field(default_factory= lambda: gen_rnd()) # gen random id for wandb tag !info:default_factory

	home: Path 					= Path()
	exp_dir: Path 				= Path('exp')
	
	@property
	def project_dir(self) -> Path: 
		return self.home / 'projects' / self.project
	@property
	def dump_dir(self) -> Path:
		return Path('dump')
	@property
	def tmp_dir(self) -> Path:
		return Path(self.dump_dir,'tmp')
	@property
	def dump_exp_dir(self) -> Path:
		return Path(self.dump_dir, 'exp')
	@property
	def cluster_dir(self) -> Path:
		return Path(self.exp_dir, 'cluster')
	@property
	def exchange_dir(self) -> Path:
		return Path(self.exp_dir, 'exchange')
	@property
	def state_dir(self) -> Path:
		return Path(self.exp_dir, 'state')
	@property
	def code_dir(self) -> Path:
		return Path(self.exp_dir, 'code')
	@property
	def exp_data_dir(self) -> Path:
		return Path(self.exp_dir, 'exp_data')
	@property
	def run_path(self) -> Path:
		return Path(self.project_dir, self.run_name)
	
	class Config:
		arbitrary_types_allowed = True