from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Callable
import subprocess

from uuid import UUID
from time import time
from typing import Callable
from functools import partial

from pathlib import Path
from rich import inspect, print

from loguru import logger
from shlex import split as split_cmd

def sub_run(cmd: str | list[str], cwd: Path | str = Path()) -> str: # subprocess.CompletedProcess
	""" run and print simple """
	cmd = cmd if isinstance(cmd, list) else split_cmd(cmd, posix= True)

	cwd = Path(cwd)
	cwd = cwd.expanduser() # makes path absolute

	out = subprocess.run(' '.join(cmd), text= True, shell= True, capture_output= True, cwd= cwd)

	if out.returncode != 0:
		logger.info(f'cmd: {cmd}')
		logger.info(f'cwd: {cwd}')
		logger.info(f'stdout: {out.stdout}')
		logger.info(f'stdout: {out.stderr}')

	return out.stdout

def run_cli(cmd: str | list[str], cwd: str = '.'):
	if isinstance(cmd, str):
		cmd = cmd.split(' ')
	print('Running: ', cmd, sep='\n')
	out = subprocess.run(cmd, cwd= cwd, text= True, capture_output= True)
	print(out.stdout)
	return out

def filter_none(data: BaseModel):
	""" helper function to remove None values from a pydantic model and make dict """
	return dict(filter(lambda x: x[1] is not None, data.dict().items()))

def load_file(path: Path, encoding='utf-8') -> str:
	with open(path, encoding=encoding) as f:
		text = f.read()
	return text

def run_cmds(cmd: str|list, cwd:str | Path='.', silent=True, _res=None):
	if _res is None:
		_res = []
	for cmd_1 in (cmd if isinstance(cmd, list) else [cmd,]): 
		try:
			cmd_1 = [c.strip() for c in cmd_1.split(' ')]
			_res = subprocess.run(cmd_1, cwd=str(cwd), capture_output=True, text=True)
			if not silent:
				print(f'Run: {cmd_1} at {cwd}')
				print('stdout:', _res.stdout, 'stderr:', _res.stderr, sep='\n')
		except Exception as e:
			if not silent:
				print(':run_cmds:', cmd_1, e)
			return ('Fail', '')
	return _res.stdout.rstrip('\n')

def mkdir(path: Path) -> Path:
	path = Path(path)
	if path.suffix != '':
		path = path.parent
	try:
		if not path.exists() or not path.is_dir():
			path.mkdir(parents=True)
	except Exception as e:
		print(':mkdir:', e)
	return path

def inspect_if(obj: Any, condition: bool, methods= True, **kwargs):
	if condition:
		inspect(obj, methods= methods, **kwargs)

import os
from uuid import uuid4

def gen_rnd(seed: int = None, n_char: int = 10) -> str:
	""" generate random string of length """
	seed = seed or os.environ.get('seed', 0)
	return uuid4().hex[:n_char]

def gen_time_id(n= 7):
	return str(round(time() * 1000))[-n:]

def print_maybe_inspect(
		delay_fn: Callable, 
		title: str|None= None, 
		show: bool= True,
		show_methods: bool= True,
		break_on_error: bool= False
	):
	print(title or '')
	try:
		obj = delay_fn() # delay the call to catch errors better by lambdafying ie lambda: fn()
	except Exception as e:
		print('print_maybe_inspect:error: ', e)
		obj = None
		if break_on_error:
			raise e
	
	if obj is not None:
		if show:
			inspect(obj, methods= show_methods)
		print(obj)
	
	return obj

def mega_inspect_pydanmod(
		pydanmod: BaseModel,
		title: str= None, 
		show: bool= True,
		show_methods: bool= True,
		break_on_error: bool= False,
		mega: bool= True,
	):
	print(title or '')

	print('CONFIG:')
	for k,v in pydanmod.Config.__dict__.items():
		print(f'{k}: {v}')

	_print_maybe_inspect: Callable = partial(
		print_maybe_inspect, 
		show= show, 
		break_on_error= break_on_error,
		show_methods= show_methods,
	)
	if mega:
		for _ in range(2): # loop to check the parsed json
			_print_maybe_inspect(lambda: pydanmod.dict(), 'PYDANMOD_DICT')
			js = _print_maybe_inspect(lambda: pydanmod.json(), 'PYDANMOD_JSON')
			_print_maybe_inspect(lambda: pydanmod.schema_json(by_alias= False), 'PYDANMOD_SCHEMA_JSON') # alias r dum
			_print_maybe_inspect(lambda: pydanmod.schema(), 'PYDANMOD_SCHEMA')
			print(js)
			pydanmod = _print_maybe_inspect(lambda: pydanmod.parse_raw(js, encoding= 'utf8'), 'PYDANMOD_PARSE_RAW')
	return pydanmod

def this_is_noop(step, n_step, n= None, every= None):
	"""
	convert n to every
	every    | n 

	n negative log every 1
	n 0 log every 0
	n positive log every n_step//n

	every negative log last step
	every 0 log every 0
	every positive log every
	"""

	if n is not None:
		if n == 0:
			every = 0
		elif n < 0:
			every = 1
		else:
			every = n_step//n

	if step == 1:
		print(f'every: {every}' )

	return (every >= 0 or n_step != step) and (every == 0 or step % every != 0)
	
import torch
import numpy as np

def d_flatten_r(
	d:dict, 
	name= '',
	sep_head= '/',
	sep= '.', 
	items: list= None
) -> dict:
	""" recursively flatten a dict """
	
	items = items or {}
	
	if not isinstance(d, dict):
		items[name] = d
		return items
	
	for k, v in d.items():
		k = (name + sep_head + k) if name else k
		if isinstance(v, dict):
			items |= d_flatten_r(v, name= k, sep_head= sep_head, sep= sep, items= items) 
		else:
			items[k] = v
	
	return items

# from typing import TypeVar
# func = TypeVar['lambda', Callable]
class AvailableMetrics():
	# mean: func 		= def mean(v): return np.mean(v),
	# std: Callable 	= lambda v: np.std(v),
	# median: Callable= lambda v: np.median(v),
	# min: Callable 	= lambda v: np.min(v),
	# max: Callable 	= lambda v: np.max(v),
	# sum: Callable 	= lambda v: np.sum(v),
	# noop: Callable 	= lambda v: v

	def mean(v): return np.mean(v)
	def std(v): return np.std(v)
	def median(v): return np.median(v)
	def min(v): return np.min(v)
	def max(v): return np.max(v)
	def sum(v): return np.sum(v)
	def noop(v): return v

from typing import Generator

class VarMetric(BaseModel):
	""" a variable, a metric to compute on it, and the method """
	v: list | np.ndarray | float | int | dict

	group: bool = Field(True, description= 'group with other VarMetrics, or not (new panel)')

	metrics: list[Callable] = Field(
		default= [AvailableMetrics.mean, AvailableMetrics.std], # 
		description= 'list of metrics to compute on v, empty for nada'
	)

	@property
	def sep(self) -> str:
		return '.' if self.group else '/'

	def __iter__(self) -> Generator[Any, None, None]:  # maybe this should be a generator
		for item in self.v:
			yield item

	class Config:
		arbitrary_types_allowed = True

from copy import deepcopy

def compute_metrix(
	v_cpu_d: dict | VarMetric | list[VarMetric],
	name= '',
	sep_metric = '.' # separate plot by name
):

	d = d_flatten_r(v_cpu_d, name= name, sep_head= '/', sep= '.') # not the same sep
	
	metrix = dict()
	for key, value in d.items():

		if np.isscalar(value):
			value = VarMetric(v= value, metrics= [AvailableMetrics.noop])

		if isinstance(value, list): # to np array, to VarMetric
			value = np.array(value)

		if isinstance(value, (np.ndarray, np.generic)): # to VarMetric
			value = np.squeeze(value) # maybe redundant
			value = VarMetric(v= value)
		
		if isinstance(value, VarMetric): # compute metrics
			for metric in value.metrics:
				metric_name = key + sep_metric + metric.__name__
				metrix[metric_name] = metric(value.v)
	return metrix


# def compute_metrix(
# 	v_cpu_d: dict | VarMetric | list[VarMetric], 
# 	name= '', 
# 	sep_metric = '.' # separate plot by name
# ):

# 	""" metrix = dict("upper.lower/varsep/ifdict" = value.mean, ...= value.std, ...)
# 	- sep idetifies unique values to be tracked / computed
# 	- does not accept torch tensors

# 	# wandb naming convention
# 	## panels
# 	keys/names/sepfwdslash/panel0/var0
# 	keys/names/sepfwdslash/panel1/var0
# 	keys/names/sepfwdslash/key/panel2/var0

# 	## dots are ignored
# 	keys/names/sepdot/panel0.bias/var0 # panel0.bias = one panel
# 	keys/names/sepdot/panel0.weights/var0 # panel0.weights = second panel

# 	## panels then split by var name
# 	keys/names/sepdot/panel0.bias/var0 # panel0.bias = one panel, one plot
# 	keys/names/sepdot/panel0.bias/var1 # panel0.bias = same panel, different plot

# 	## goal: 
# 	if group, sep becomes dot
# 	if not group, sep becomes fwdslash
# 	if its a VarMetric, execute
# 	"""

# 	# sep_head = sep_head or sep
# 	# v_cpu_d_flat: dict = d_flatten_r(v_cpu_d, name= name, sep_head= sep_head, sep= sep) # not the same sep
	
	
# 	metrix = dict()

# 	if isinstance(v_cpu_d, list):
# 		for i, v in enumerate(v_cpu_d):
# 			if not isinstance(v, VarMetric):
# 				v = VarMetric(v= v, group= True)
# 			name_= name + v.group + str(i)
# 			metrix[name_] = compute_metrix(v)
# 		return metrix

# 	if isinstance(v_cpu_d, VarMetric): # compute metrics
# 		for metric in v_cpu_d.metrics:
# 			metric_name = name + sep_metric + metric.__name__
# 			metrix[metric_name] = metric(v_cpu_d.v)
# 		return metrix
	
# 	for key, value in v_cpu_d.items():

# 		if np.isscalar(value):
# 			value = VarMetric(v= value, metrics= [AvailableMetrics.noop])

# 		if isinstance(value, list): # to np array, to VarMetric
# 			value = np.array(value)

# 		if isinstance(value, (np.ndarray, np.generic)): # to VarMetric
# 			value = np.squeeze(value) # maybe redundant
# 			value = VarMetric(v= value)
# 			metrix |= compute_metrix(value, name= name + '/' + key)
# 			continue

# 		if isinstance(value, VarMetric): # compute metrics
# 			metrix |= compute_metrix(value, name= name + '/' + key)
# 			continue
		
# 		if isinstance(value, dict):
# 			for k, v in value.items():
# 				v = VarMetric(v= v, name= k)
# 				name_ = name + '/' + key + v.sep + k
# 			metrix |= compute_metrix(v, name= name_)

# 	return metrix

def get_attrs(pydanmod: BaseModel):
    def _get_attrs():
        everything = [(k, getattr(pydanmod, k)) for k in dir(pydanmod) if not k.startswith('_')]
        for k, v in everything:
            if not callable(v) and not isinstance(v, BaseModel):
                yield k, v
    return dict(_get_attrs())



# def compute_metrix(v: dict, name= '', sep='/'):
# 	""" metrix = dict("upper.lower/varsep/ifdict" = value.mean, ...= value.std, ...)
# 	- metrix sep not the same as d_flatten_r sep 
# 	"""
# 	v = d_flatten_r(v, name= name) # not the same sep
	
# 	items = {}

# 	if isinstance(v, list):
# 		all_scalar = all([np.isscalar(_vi) for _vi in v])
# 		if all_scalar:
# 			v = np.array(v)
# 		else:
# 			v = {str(i): v_item for i, v_item in enumerate(v)}
	
# 	if isinstance(v, dict):
# 		for k_item, v_item in v.items():
# 			k = ((parent + sep) if parent else '') + k_item
# 			items |= compute_metrix(v_item, parent= k)

# 	elif isinstance(v, torch.Tensor):
# 		v = v.detach().cpu().numpy()

# 	if isinstance(v, (np.ndarray, np.generic)):
# 		v = np.squeeze(v)

# 	if np.isscalar(v):
# 		items[parent] = v

# 	elif isinstance(v, (np.ndarray, np.generic)):
# 		items[parent + r'_\mu$'] = v.mean()
# 		if v.std() and debug:
# 			items['std'+sep+parent + r'_\sigma$'] = v.std()

# 	return items