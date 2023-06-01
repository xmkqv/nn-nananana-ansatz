
import sys
import traceback
import os
from typing import Callable
import print
import inspect
import numpy as np
from copy import deepcopy
import pickle as pk
import numpy as np
from pathlib import Path

from .core import run_cmds, iterate_n_dir, gen_time_id, flat_any
from .core import mkdir, dict_to_cmd, cmd_to_dict, ins_to_dict, walk_ins_tree

this_file_path = Path(__file__) 

from .dist_repo import DistBase, Naive, HFAccelerate, SingleProcess
from .logger_repo import LoggerBase, Wandb
from .resource_repo import ResourceBase, Niflheim
from .sweep_repo import SweepBase, Optuna
from .other_repo import OptBase, DataBase, SchedulerBase, PathsBase, ModelBase

class PyfigBase:

	env: 				str 	= ''
	user: 				str 	= None
 
	project:            str     = ''
	run_name:       	Path	= ''
	exp_name:       	str		= '' # default is demo
	exp_id: 			str		= ''
	run_id:		        str		= ''
	group_exp: 			bool	= False

	multimode: 			str		= '' # 'max_mem:profile:opt_hypam:train:eval'
	mode: 				str		= ''
	debug: 				bool    = False
	submit: 			bool 	= False
	
	seed:           	int   	= 0
	dtype:          	str   	= ''
	device: 		int|str 	= ''


	n_log_metric:		int  	= 100
	n_log_state:		int  	= 4
	is_logging_process: bool    = False

	lo_ve_path:			str 	= ''

	_group_i: int = 0
	@property
	def group_i(ii):
		return ii._group_i

	zweep: str = ''

	n_default_step: 	int 	= 10
	n_train_step:   	int   	= 0
	n_pre_step:    		int   	= 0
	n_eval_step:        int   	= 0
	n_opt_hypam_step:   int   	= 0
	n_max_mem_step:     int   	= 0

	@property
	def n_step(ii):
		n_step = dict(
			train		= ii.n_train_step, 
			pre			= ii.n_pre_step, 
			eval		= ii.n_eval_step, 
			opt_hypam	= ii.n_opt_hypam_step, 
			max_mem		= ii.n_max_mem_step
		).get(ii.mode)
		if not n_step: 
			n_step = ii.n_default_step
		return n_step
  
	class model(ModelBase):
		pass

	class sweep(SweepBase):
		pass

	class logger(LoggerBase):
		pass

	class dist(DistBase):
		pass

	class resource(ResourceBase):
		pass

	class data(DataBase):
		pass

	class opt(OptBase):
		pass

	class scheduler(SchedulerBase):
		pass

	data_tag: str		= 'data'
	
	max_mem_alloc_tag: str = 'max_mem_alloc'
	opt_obj_all_tag: str = 'opt_obj_all'
	opt_obj_tag: str = 'opt_obj'
		
	pre_tag: str = 'pre'
	train_tag: str = 'train'
	eval_tag: str = 'eval'
	opt_hypam_tag: str = 'opt_hypam'
		
	v_cpu_d_tag: str = 'v_cpu_d'
	c_update_tag: str = 'c_update'
		
	lo_ve_path_tag: str = 'lo_ve_path'
	gather_tag: str = 'gather'
	mean_tag: str = 'mean'
	
	class paths(PathsBase):
		pass

	ignore_f = ['commit', 'pull', 'backward']
	ignore_p = ['parameters', 'scf', 'tag', 'mode_c']
	ignore: list = ['ignore', 'ignore_f', 'ignore_c'] + ignore_f + ignore_p
	ignore += ['d', 'cmd', 'sub_ins', 'd_flat', 'repo', 'name', 'base_d', 'c_init', 'p']
	base_d: dict = None
	c_init: dict = None
	run_debug_c: bool = False
	run_sweep: bool = False

	repo = dict(
		dist = dict(
			Naive = Naive,
			HFAccelerate = HFAccelerate,
			SingleProcess = SingleProcess,
		),
		logger = dict(Wandb = Wandb),
		resource = dict(Niflheim = Niflheim),	
		sweep = dict(Optuna = Optuna),
		opt = dict(OptBase = OptBase),
		data = dict(DataBase = DataBase),
		scheduler = dict(SchedulerBase = SchedulerBase),
		paths = dict(PathBase = PathsBase),
		model = dict(ModelBase = ModelBase),
	) 

	def __init__(ii, 
		notebook:bool=False,  		# removes sys_arg for notebooks
		sweep: dict={},				# special properties for config update so is separated
		c_init: dict|str|Path={},  	# args specific
		post_init_arg: dict={},
		**other_arg
	):
		# base_c (hardcoded in base Pyfig) <  mode_c (in app.mode_c)  < debug_c (in app.debug_c) <  c_init (cmd line + hard coded inputs)

		"""
		todo 
		# now
		

		# soon
		- mode + multimode rehaul (property interplay idea)
		- optimal threading setup 
		
		# later
		
		# """


		ii.init_sub_cls(sweep= sweep)

		if not notebook:
			ref_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore)
			sys_arg = cmd_to_dict(sys.argv[1:], ref_d)

		print('\npyfig:post_init')
		ii.c_init = deepcopy((c_init or {}) | (other_arg or {}) | (sys_arg or {}))
		ii.update(ii.c_init)
		
		# because you suck
		app_post_init = ii.app.post_init_update()
		ii.update(app_post_init)
		ii.setup_exp_dir(group_exp= False, force_new_id= False)
		ii.c_init |= app_post_init | dict(exp_name= ii.exp_name, exp_id= ii.exp_id)

		if ii.debug:
			os.environ['debug'] = 'True'
			print('\n\npyfig:post_init:post_init_arg')
			print.print(ii.d, sort_dicts=True)
			print('\n\npyfig:post_init:c_init')
			print.print(ii.c_init, sort_dicts=True)

		err_path = Path(ii.paths.exp_dir, str(ii.dist.rank) + '.pyferr')
		os.environ['PYFERR_PATH'] = str(err_path)

		import atexit
		from time import sleep 
		
		def exit_handler():
			""" function to catch the traceback at exit """	
			job_id = os.environ.get('SLURM_JOB_ID', False)
			print('exit: job_id', job_id)
			exc_type, exc_value, exc_traceback = sys.exc_info()
			traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
			print(exc_type, exc_value, exc_traceback)
			print(traceback_string)
			err_path.write_text(str(exc_value))
			err_path.write_text(str(traceback_string))
			if job_id:
				sleep(100)
				run_cmds(f'scancel {job_id}')
		atexit.register(exit_handler)

		ii.if_debug_log([sys_arg, dict(os.environ.items()), ii.d], 
		[f'log_sys_arg_{ii.dist.pid}.log', f'log_env_run_{ii.dist.pid}.log', f'log_d_{ii.dist.pid}.log'])

	def deepcopy(ii):
		d = ii.super()(c_init=deepcopy(ii.d))
		return d

	def start(ii):

		if ii.is_logging_process:
			print('pyfig:start: logging process creating logger')
			assert ii.submit == False

			ii._group_i += 1
			ii.run_id = ii.exp_id + '.' + ii.mode + '.' + str(ii._group_i) + '.' + str(ii.dist.rank)
		
			print('pyfig:start: exp_dir = \n ***', ii.paths.exp_dir, '***')
			print('pyfig:start: run_id = \n ***', ii.run_id, '***')
			tags = [str(s) for s in [*ii.mode.split('-'), ii.exp_id, ii._group_i]]
			tags = [s for s in tags if s]  # wb crashes if tags is empty
			print(f'pyfig:logger: tags- {tags}')

			ii.logger.start(ii.d, tags= tags, run_id= ii.run_id)

	def end(ii, plugin: str= None):
		import torch
		import gc
		gc.collect()

		with torch.no_grad(): # https://stackoverflow.com/questions/57858433/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel
			torch.cuda.empty_cache()

		if plugin is None:
			ii.dist.wait_for_everyone()
			ii.logger.end()
			ii.dist.end()
		else:
			plugin = getattr(ii, plugin).end()


	def run_submit(ii):
		try: 
			print('submitting to cluster')

			run_or_sweep_d = ii.get_run_or_sweep_d()

			ii.submit = False # docs:submit
			ii.run_sweep = False # docs:submit
			ii.zweep = False
			
			for i, run_d in enumerate(run_or_sweep_d):
				
				exp_name, exp_id = ii.setup_exp_dir(group_exp= False, force_new_id= True)

				ii.save_code_state()

				compulsory = dict(exp_id= exp_id, exp_name= exp_name, submit= False, run_sweep= False, zweep= '')

				ii.resource.cluster_submit(ii.c_init | run_d | compulsory)

				ii.if_debug_log([dict(os.environ.items()), run_d], ['log-submit_env.log', 'log-submit_d.log'])

				print('log_group_url: \t\t\t',  ii.logger.log_run_url)
				print('exp_dir: \t\t\t', ii.paths.exp_dir)
				print('exp_log: \t\t\t', ii.resource.device_log_path(0))

			sys.exit('Success, exiting from submit.')
		
		except Exception as e:
			sys.exit(str(e))


	def save_code_state(ii, exts = ['.py', '.ipynb', '.md']):
		import shutil
		shutil.copytree('things', ii.paths.code_dir/'things')
		[shutil.copyfile(p, ii.paths.code_dir/p.name) for p in ii.paths.project_dir.iterdir() if p.suffix in exts]
		[shutil.copyfile(p, ii.paths.code_dir/p.name) for p in ii.paths.dump_dir.iterdir() if p.suffix in exts]

	@property
	def _paths(ii):
		path_filter = lambda item: any([p in item[0] for p in ['path', 'dir']])
		paths = dict(filter(path_filter, ii.d.items()))
		ii.if_debug_print_d(paths, '\npaths')
		return paths

	@property
	def cmd(ii):
		return dict_to_cmd(ii.d_flat)

	@property
	def d(ii):
		return ins_to_dict(ii, sub_ins=True, prop=True, attr=True, ignore=ii.ignore)

	@property
	def d_flat(ii):
		return flat_any(ii.d)

	def init_sub_cls(ii, sub_cls: type=None, name: str=None, sweep: dict= None) -> dict:

		if name and not (sub_cls is None):
			print('adding plugin: ', name, sub_cls)
			sub_ins = sub_cls(parent=ii)
			setattr(ii, name, sub_ins)
		else:
			sub_cls = ins_to_dict(ii, sub_cls=True)
			for sub_k, sub_cls_i in sub_cls.items():
				print('init plugin: ', sub_k, sub_cls_i)
				sub_ins = sub_cls_i(parent=ii)
				setattr(ii, sub_k, sub_ins)

		for k,v in (sweep or {}).items():
			setattr(ii.sweep, k, v)

	def setup_exp_dir(ii, group_exp= False, force_new_id= False):

		if (not ii.exp_id) or force_new_id:
			ii.exp_id = gen_time_id(7)
			exp_name = ii.exp_name or 'junk'
			group_exp = group_exp or ('~' in exp_name)
			exp_group_dir = Path(ii.paths.dump_exp_dir, exp_name)
			exp_group_dir = iterate_n_dir(exp_group_dir, group_exp= group_exp) # pyfig:setup_exp_dir does not append -{i} if group allowed
			ii.exp_name = exp_group_dir.name

		print('pyfig:setup_exp_dir: ', ii.paths.exp_dir)
		print('\nexp_dir:mkdir ', ii.paths.exp_dir) 
		[mkdir(p) for _, p in ii.paths.d.items()]
		return ii.exp_name, ii.exp_id
	
	def get_run_or_sweep_d(ii,):
		if ii.zweep: 
			zweep = ii.zweep.split('-')
			zweep_v = zweep[0]
			t = zweep[-1]
			from things.core import cmd_to_dict
			sweep_over = []
			for v in zweep[1:-1]:
				cmd = f' --{zweep_v} {v}'
				d = cmd_to_dict(cmd, ref= ii.d_flat)
				sweep_over.append(d)
			print('pyfig:zweep: ', zweep_v, t, [d.values() for d in sweep_over])  # !!! include type me 
			return sweep_over
		
		elif ii.run_sweep:
			return ii.sweep.get_sweep()

		else:
			""" single run takes c from base in submit loop """
			c = [dict(),] 
			return c

	def if_debug_log(ii, d_all:list, name_all: list):
		try:
			if ii.debug:
				for d, name in zip(d_all, name_all):
					if Path(ii.paths.exp_dir).is_dir():
						ii.log(d, path=ii.paths.cluster_dir/name, group_exp=False)
		except Exception as e:
			print('pyfig:if_debug_log: ', e)
			
	def partial(ii, f:Callable, args=None, **kw):
		d: dict = flat_any(args if args else {}) | ii.d_flat | (kw or {})
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:v for k,v in d.items() if k in d_k} 
		ii.if_debug_print_d(d, f'pyfig:partial setting for {f.__name__}')
		return f(**d)

	def update(ii, _arg: dict=None, c_update: dict=None, silent: bool= False, **kw):
		print('\npyfig:update')
		silent = silent or ii.debug

		c_update = (_arg or {}) | (c_update or {}) | (kw or {})
		c_update = flat_any(c_update)
		c_keys = list(ii.d_flat.keys())
		arg = dict(filter(lambda kv: kv[0] in c_keys, c_update.items()))

		for k_update, v_update in deepcopy(arg).items():

			if k_update=='dtype': # !!! pyfig:fix: Special case, generalify
				ii.dtype = v_update
				if not silent:
					print('dtype: ---> \t\t\t\t\t', v_update)
			elif k_update in ii.repo.keys():
				plugin = ii.repo[k_update][v_update]
				plugin_name = k_update.split('_')[0]
				if not silent:
					print('adding plugin:', plugin_name, plugin, '...')
				ii.init_sub_cls(sub_cls=plugin, name=plugin_name)
			else:
				is_updated = walk_ins_tree(ii, k_update, v_update)
				if not is_updated:
					print(f'not updated: k={k_update} v={v_update} type={type(v_update)}')

		not_arg = dict(filter(lambda kv: kv[0] not in c_keys, arg.items()))

		ii.if_debug_print_d(not_arg, msg='\npyfig:update: arg not in pyfig')	

	def if_debug_print_d(ii, d: dict, msg: str=''):
		
		from .typfig import AnyTensor
		from .utils import print_tensor
		from torch import Tensor

		if ii.debug and d:
			if isinstance(d, dict):
				print('-'*50, msg, sep='\n')
				for k,v in d.items():
					if isinstance(v, Tensor | np.ndarray):
						print_tensor(v, k= k)
					else:
						print(k, v, sep='\t'*3)
		
	@staticmethod
	def log(info: dict|str, path: Path, group_exp: bool=True):
		mkdir(path)
		info = print.pformat(info)
		path = iterate_n_dir(path, group_exp=group_exp)
		with open(path, 'w') as f:
			f.writelines(info)

	def to(ii, framework='torch', device= None, dtype= None):
		import torch

		base_d = ins_to_dict(ii, attr=True, sub_ins=True, flat=True, ignore=ii.ignore+['parameters'])
		ii.if_debug_print_d(base_d, msg='\npyfig:to: base_d')

		# write a function to filter lists out of a dictionary
		def get_numerical_arrays():
			from torch import Tensor

			d = {}
			for k,v in base_d.items():
				if isinstance(v, (np.ndarray, np.generic, Tensor, list)):
					if isinstance(v, (np.ndarray, np.generic, Tensor)):
						v = v.tolist()
					if len(v) == 0:
						continue
					else:
						v_flat0 = flat_any(v)[0]
						is_str = isinstance(v_flat0, str)
						if not is_str:
							d[k] = np.array(v).astype(np.float64)
			return d

		d = get_numerical_arrays()
		ii.if_debug_print_d(d, msg='pyfig:to: d')

		if 'torch' in framework.lower():
			d = {k:torch.tensor(v, requires_grad=False).to(device= ii.device, dtype= ii.dtype) for k,v in d.items()}

		if 'numpy' in framework.lower():
			d = {k:v.detach().cpu().numpy() for k,v in d.items() if isinstance(v, torch.Tensor)}

		ii.update(d)

	def _memory(ii, sub_ins=True, attr=True, _prop=False, _ignore=[]):
		return deepcopy(ins_to_dict(ii, attr=attr, sub_ins=sub_ins, prop=_prop, ignore=ii.ignore+_ignore+['logger']))


# class Repo(PlugIn):
# 	class dist(PlugIn):
# 		Naive = Naive
# 		HFAccelerate = HFAccelerate
# 		SingleProcess = SingleProcess
	
# 	class logger(PlugIn):
# 		Logger = Wandb

# 	class resource(PlugIn):
# 		Niflheim = Niflheim
	
# 	class sweep(PlugIn):
# 		Optuna = Optuna

# 	class opt(PlugIn):
# 		Opt = OptBase

# 	class data(PlugIn):
# 		DataBase = DataBase

# 	class scheduler(PlugIn):
# 		SchedulerBase = SchedulerBase

# 	class paths(PlugIn):
# 		PathBase = PathsBase

# 	class model(PlugIn):
# 		ModelBase = ModelBase





