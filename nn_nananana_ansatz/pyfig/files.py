from pathlib import Path
from .pretty import print_table
import pickle as pk
import yaml
import json
import numpy as np
from functools import partial

def read(path: Path) -> str:
	with open(path, encoding='utf-8') as f:
		text = f.read()
	return text

def write_file(path: str | Path, content: str):
    with open(path, 'w') as f:
        f.write(content)

def get_files(file_dir: Path, filetype: str= '.pdf') -> list[Path]:
	""" get all files in a directory """
	return list(file_dir.glob(f'*.{filetype}'))

def lo_ve(path: Path= None, data: dict= None) -> dict:
	""" loads anything you want (add other interfaces as needed) 

	1- str -> pathlib.Path
	2- get suffix (filetype)
	3- from filetype, decide encoding eg 'w' or 'wb'
	4- if data is present, decide to save
	5- from filetype, encoding, save or load, get or dump the data 
	"""

	file_interface_all = dict(
		pk = dict(
			rb = pk.load,
			wb = partial(pk.dump, protocol=pk.HIGHEST_PROTOCOL),
		),
		yaml = dict(
			r = partial(yaml.load, Loader=yaml.FullLoader),
			w = yaml.dump,
		),
		json = dict(
			r = json.load,
			w = json.dump,
		),
		cmd = dict(
			r = lambda f: ' '.join(f.readlines()),
			w = lambda x, f: f.writelines(x),
		),
		npy = dict(
			r = np.load,
			w = np.save,
		),
		npz = dict(
			r = np.load,
			w = np.savez_compressed,
		)
	)

	if isinstance(data, dict) and not data:
		print('lo_ve: data is empty dict- not dumping anything. Setting None and trying to load path.')
		data = None

	if data is None:
		print(f'lo_ve: loading {path}')
	else:
		print(f'lo_ve: dumping {data.keys()} \n to {path}')

	path = Path(path)
	file_type = path.suffix[1:]
	if file_type not in file_interface_all:
		print(f'lo_ve: file_type is {file_type} not supported. Returning empty dict')
		return {}

	mode = ('r' if data is None else 'w') + ('b' if file_type in ['pk',] else '')
	interface = file_interface_all[file_type][mode]

	if 'r' in mode and not path.exists():
		return print('path: ', path, 'n\'existe pas- returning None')

	if 'state' in file_type:
		data = interface(data, path) if data is not None else interface(path)
	elif 'npz' in file_type:
		data = interface(path, **data) if data is not None else interface(path)
	else:
		with open(path, mode) as f:
			data = interface(data, f) if data is not None else interface(f)
	return data