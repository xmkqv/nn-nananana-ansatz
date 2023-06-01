import inspect
from typing import Callable, Union
import wandb
from pathlib import Path
import sys
import print
from copy import copy
import numpy as np
import os
import re
from time import sleep, time
import optree
from copy import deepcopy
from pyfig_util import cluster_options, PyfigBase

from things import PlugIn from things import get_cartesian_product
from things import run_cmds, run_cmds_server, count_gpu, gen_alphanum
from things import mkdir, cmd_to_dict, dict_to_wandb, iterate_n_dir
from things import type_me
from things import add_to_Path, flat_any
from things import load, dump, cls_to_dict, dict_to_cmd

import dump.user_secret as user_secret

import gc


""" 
System
n_e = \sum_i charge_nuclei_i - charge = n_e
spin = n_u - n_d
n_e = n_u + n_d
n_u = 1/2 ( spin + n_e ) = 1/2 ( spin +  \sum_i charge_nuclei_i - charge)


"""

""" pyfig
wandb
wandb sync wandb/dryrun-folder-name    # to sync data stored offline


"""



"""
### Alert
- push to lumi
- read gpu intro
- email rene about running now 
- run on lumi
- demo sweep niflheim
- demo sweep niflheim (offline) and push
- demo nodes > 1 niflheim
- profiler working

### Medium
- user file


### Stretch


### Vanity


"""

""" Pyfig Docs
### NEED BEFORE
- wandb api key 
- change secrets file

### Usage 10/1/23
- python run.py

Flags:
- --submit (submit to Niflheim, UPDATE SO USER INDEPENDENT IN USER.PY)
- --n_gpu x (sets for x gpus)
- --debug (creates log files in dump/tmp)
- --run_sweep (runs sweep from variables in sweep subclass parameters dict)

Examples:
- python run.py --submit --run_sweep --debug --n_gpu 8
- python run.py --submit --run_sweep

### What can you do 

### Issues 
- PlugIn classes can NOT call each other
- properties can NOT recursively call each other

### TODO
- type _p as Pyfig

"""

class Pyfig(PyfigBase):
 
	run_name:       Path        = 'run.py'
	exp_dir:        Path	    = ''
	exp_name:       str     	= ''
	exp_id: 		str		 	= ''
 
	seed:           int         = 808017424 # grr
	dtype:          str         = 'float32'
	n_step:         int         = 10000
	log_metric_step:int         = 10
	log_state_step: int         = 10          
	
	class data(PlugIn):
		charge:     int         = 0
		spin:       int         = 0
		a:          np.ndarray  = np.array([[0.0, 0.0, 0.0],])
		a_z:        np.ndarray  = np.array([4.,])

		n_e:        int         = property(lambda _: int(sum(_.a_z)))
		n_u:        int         = property(lambda _: (_.spin + _.n_e)//2)
		n_d:        int         = property(lambda _: _.n_e - _.n_u)
		
		n_b:        int         = 256
		n_corr:     int         = 20
		n_equil:    int         = 10000
		acc_target: int         = 0.5

	class model(PlugIn):
		with_sign:      bool    = False
		n_sv:           int     = 32
		n_pv:           int     = 16
		n_fbv:          int     = property(lambda _: _.n_sv*3+_.n_pv*2)
		n_fb:           int     = 2
		n_det:          int     = 1
		terms_s_emb:    list    = ['ra', 'ra_len']
		terms_p_emb:    list    = ['rr', 'rr_len']
		ke_method:      str     = 'vjp'

	class sweep(PlugIn):
		method          = 'grid'
		parameters = dict(
			n_b  = {'values' : [16, 32, 64]},
		)
  
	class wandb_c(PlugIn):
		job_type        	= 'debug'
		entity          	= property(lambda _: _._p.project)
		program         	= property(lambda _: Path(_._p.run_dir,_._p.run_name))

	class dist(PlugIn):
		_dist_id 			= ''
		_sync 				= ['grads',]
		accumulate_step		= 5
		head		        = True
		gpu_id		        = property(lambda _: \
			''.join(run_cmds('nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader')).split('.')[0]
		)

	class cluster(PlugIn):
		pass

	project:            str     = property(lambda _: 'hwat')
	project_dir:        Path    = property(lambda _: Path().home() / 'projects' / _.project)
	dump:               str     = property(lambda _: Path('dump'))
	dump_exp_dir: 		Path 	= property(lambda _: _.dump/'exp')
	TMP:                Path    = property(lambda _: _.dump/'tmp')
	
	cluster_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'cluster'))
	exchange_dir: 		Path    = property(lambda _: Path(_.exp_dir, 'exchange'))
	
	run_dir:            Path    = property(lambda _: _.project_dir)
	# run_dir:            Path    = property(lambda _: Path(__file__).parent.relative_to(Path().home()))

	sweep_path_id:      str     = property(lambda _: (f'{_.wandb_c.entity}/{_.project}/{_.exp_name}')*bool(_.exp_name))

	hostname: 			str      = property(lambda _: _._static.setdefault('hostname', run_cmds('hostname')))
	n_device:           int      = property(lambda _: count_gpu())

	run_sweep:          bool    = False	
	wandb_sweep: 		bool	= False
	group_exp: 			bool	= False
	
	wb_mode:            str      = 'disabled'
	debug:              bool     = False
	submit:             bool     = False
	cap:                int      = 40

	commit_id           = property(lambda _: run_cmds('git log --pretty=format:%h -n 1', cwd=_.project_dir))
	_git_commit_cmd:    list     = 'git commit -a -m "run"' # !NB no spaces in msg 
	_git_pull_cmd:      list     = ['git fetch --all', 'git reset --hard origin/main']

	### User/Env Deets
	user:               str     = user_secret.user
	git_remote:         str     = user_secret.git_remote
	git_branch:         str     = user_secret.git_branch
	env:                str     = user_secret.env
	cluster_name: 		str		= user_secret.cluster_name
	backend:	 		str 	= user_secret.backend
	n_gpu:              int     = 1  					# submission devices

	# backend_cmd: type = backend_cmd_all[backend]
	# pci_id: str = property(lambda _: _.backend_cmd['pci_id'])
 
	pci_id = property(lambda _: ''.join(run_cmds()))
	### things that should be put somewhere better
	# slurm
	_n_job_running: int = \
		property(lambda _: len(run_cmds('squeue -u amawi -t pending,running -h -r', cwd='.').split('\n')))
	# wandb
	_wandb_run_url = property(lambda _: 
			f'https://wandb.ai/{_.wandb_c.entity}/{_.project}/' \
   				+(_.wandb_sweep*('sweeps/'+_.exp_name) 
		 		or _.run_sweep*f'groups/{_.exp_name}' 
		   		or f'runs/{_.exp_id}'))
	_wandb_ignore:      list     = ['sbatch', 'sweep']
	# commands
	_useful = 'ssh amawi@svol.fysik.dtu.dk "killall -9 -u amawi"'
	_end_this_job:	str = property(lambda _: ''.join(run_cmds(f'scancel {_.cluster._job_id}')))

	### these are left alone by pyfig ops
	_ignore:            list     = ['d', 'cmd', 'partial', 'lo_ve', 'log', 'merge', 'accumulate', '_static']
	_static: 			dict     = dict() # allows single execution properties

	def __init__(ii, arg={}, wb_mode='online', submit=False, run_sweep=False, notebook=False, **kw):
		super().__init__()
  
		init_arg = dict(run_sweep=run_sweep, submit=submit, wb_mode=wb_mode) | kw

		### unstable ###
		slurm = cluster_options[ii.cluster_name]
		print(slurm)
		print('init PlugIn classes')
		for k, v in Pyfig.__dict__.items():
			if isinstance(v, type):
				if 'cluster' in k:
					for k_attr,v_attr in slurm.__dict__.items():
						setattr(v, k_attr, v_attr)
				v = v(parent=ii)
				setattr(ii, k, v)
		### unstable ###

		print('### updating configuration ###')
		sys_arg = cmd_to_dict(sys.argv[1:], flat_any(ii.d)) if not notebook else {}
		ii.merge(arg | init_arg | sys_arg)

		ii.dist._dist_id = ii.dist.gpu_id + '-' + ii.hostname.split('.')[0]
		print(f'### Hardware IDs {ii.dist._dist_id} ###')

		if not ii.submit:
			print('### running script ###')
			if ii.dist.head:
				ii.setup_exp(group_exp=ii.group_exp, force_new=False)
				if ii.wandb_sweep:
					wandb.agent(sweep_id=ii.sweep_path_id)
					ii._run = wandb.init(project=ii.project, entity=ii.wandb_c.entity)
				else:
					ii._run = wandb.init(
						entity      = ii.wandb_c.entity,  # team name is hwat
						project     = ii.project,         # PlugIn project in team
						dir         = ii.exp_dir,
						config      = dict_to_wandb(ii.d, ignore=ii._wandb_ignore),
						mode        = wb_mode,
						group		= ii.exp_name,
						id          = ii.exp_id,
						# settings    = wandb.Settings(start_method='fork'), # idk y this is issue, don't change
					)
				print(ii._run.get_url())
			ii._debug_log([dict(os.environ.items()), ii.d,], ['env.log', 'd.log',])
		
		if ii.submit:
			print(f'### running on cluster ###')
			ii.submit = False
			if ii.run_sweep:
				if ii.wandb_sweep:
					sweep_d = ii.get_wandb_sweep()
				else:
					sweep_d = ii.get_pyfig_sweep()
				for i, d in enumerate(sweep_d):
					ii.setup_exp(group_exp=bool(i), force_new=True)
					base_c = cls_to_dict(ii, sub_cls=True, flat=True, ignore=ii._ignore+['sweep', 'sbatch', *d.keys()])
					job = base_c | d
					ii.cluster.submit(job)

			elif ii.wandb_sweep:
				
				raise NotImplementedError

			else:
				ii.setup_exp(group_exp=(ii.group_exp or ii.debug), force_new=True) # ii.group_exp
				job = cls_to_dict(ii, sub_cls=True, flat=True, ignore=ii._ignore+['sweep', 'sbatch',])
				ii.cluster.submit(job)

			ii._debug_log([dict(os.environ.items()), ii.d], ['dump/tmp/env.log', 'dump/tmp/d.log',])
			sys.exit(ii._wandb_run_url)

	@property
	def cmd(ii):
		d = cls_to_dict(ii, sub_cls=True, flat=True, ignore=ii._ignore + ['sweep', 'head',])
		return dict_to_cmd(d)

	@property
	def d(ii):
		return cls_to_dict(ii, sub_cls=True, prop=True, ignore=ii._ignore)

	@property
	def sub_cls(ii) -> dict:
		return {k:v for k,v in ii.__dict__.items() if isinstance(v, PlugIn)}

	def setup_exp(ii, group_exp=False, force_new=False):
		if ii.exp_dir and not force_new:
			return None
		exp_name = ii.exp_name or 'junk'
		exp_group_dir = Path(ii.dump_exp_dir, 'sweep'*ii.run_sweep, exp_name)
		exp_group_dir = iterate_n_dir(exp_group_dir, group_exp=group_exp)
		ii.exp_name = exp_group_dir.name
		ii.exp_id = (~force_new)*ii.exp_id or ii._time_id(7)
		ii.exp_dir = exp_group_dir/ii.exp_id
		[mkdir(ii.exp_dir/_dir) for _dir in ['cluster', 'exchange', 'wandb']]
  
	def get_pyfig_sweep(ii):
		d = deepcopy(ii.sweep.d)
		sweep_keys = list(d['parameters'].keys())
		sweep_vals = [v['values'] for v in d['parameters'].values()]
		sweep_vals = get_cartesian_product(*sweep_vals)
		print(f'### sweep over {sweep_keys} ({len(sweep_vals)} total) ###')
		return [{k:v for k,v in zip(sweep_keys, v_set)} for v_set in sweep_vals]


	def _set_debug_mode(ii):
		ii.group_exp = ii.debug # force single exps into single folder for neatness

	def _debug_log(ii, d_all:list, p_all: list):
		if ii.debug:
			d_all = d_all if isinstance(d_all, list) else [d_all]
			p_all = p_all if isinstance(p_all, list) else [p_all]
			for d, p in zip(d_all, p_all):
				ii.log(d, create=True, path=p)
			
	def partial(ii, f:Callable, args=None, **kw):
		d = flat_any(args if args else ii.d)
		d_k = inspect.signature(f.__init__).parameters.keys()
		d = {k:copy(v) for k,v in d.items() if k in d_k} | kw
		return f(**d)

	def merge(ii, merge: dict):
		for k,v in merge.items():
			assigned = False
			for cls in [ii,] + list(ii.sub_cls.values()):
				ref = cls_to_dict(cls,)
				if k in ref:
					v_ref = ref[k]
					v = type_me(v, v_ref)
					try:
						setattr(cls, k, copy(v))
						assigned = True
						print(f'merge {k}: {v_ref} --> {v}')
					except Exception:
						print(f'Unmerged {k} at setattr')
			if not assigned:
				print(k, v, 'not assigned')

	def accumulate(ii, step: int, v_tr: dict, sync=None):
		"""
		potential issues:
			- loading / unloading too fast / slow? Crashes occasionally.
		"""
		try:
			gc.disable()
			v_path = (ii.exchange_dir / f'{step}_{ii.dist._dist_id}').with_suffix('.pk')
			v, treespec = optree.tree_flatten(deepcopy(v_tr))
			dump(v_path, v)
		except Exception as e:
			print(e)
		finally:
			gc.enable()
		
		if ii.dist.head:
			### 1 wait for workers to dump ###
			n_ready = 0
			while n_ready < ii.n_gpu:
				k_path_all = list(ii.exchange_dir.glob(f'{step}_*')) 
				n_ready = len(k_path_all)
				sleep(0.02)

			### 2 collect arrays ###
			try:
				gc.disable()
				leaves = []
				for p in k_path_all:
					v_dist_i = load(p)
					leaves += [v_dist_i]
			except Exception as e:
				print(e)
			finally:
				gc.enable()

			### 3 mean arrays ###
			v_mean = [np.stack(leaves).mean(axis=0) for leaves in zip(*leaves)]

			try:
				gc.disable()
				for p in k_path_all:
					dump(add_to_Path(p, '-mean'), v_mean)
			except Exception as e:
				print(e)
			finally:
				for p in k_path_all:
					p.unlink()
				gc.enable()

		v_sync_path = add_to_Path(v_path, '-mean')
		while v_path.exists():
			while not v_sync_path.exists():
				sleep(0.02)
			sleep(0.02)
		try:
			gc.disable()
			v_sync = load(v_sync_path)  # Speed: Only load sync vars
			v_sync = optree.tree_unflatten(treespec=treespec, leaves=v_sync)
		except Exception as e:
			print(e)
			gc.enable()
			v_sync = v_tr
		finally: # ALWAYS EXECUTED
			v_sync_path.unlink()
			gc.enable()
			return v_sync

	def convert(ii, ):
		ii.merge(d)
  
		import torch
		device, dtype
		d = cls_to_dict(ii, sub_cls=True, flat=True)
		d = {k:v for k,v in d.items() if isinstance(v, (np.ndarray, np.generic, list))}
		d = {k:torch.tensor(v, dtype=dtype, device=device, requires_grad=False) for k,v in d.items() if not isinstance(v[0], str)}
  
		

	def _setup_wandb_sweep(ii):
		d = deepcopy(ii.sweep.d)
		sweep_keys = list(d['parameters'].keys())
		n_sweep = [len(v['values']) for k,v in ii.sweep.parameters.items() if 'values' in v] 
		print(f'### sweep over {sweep_keys} ({n_sweep} total) ###')
  
		base_c = cls_to_dict(ii, sub_cls=True, flat=True, ignore=ii._ignore + ['sweep', 'head',])
		base_c = dict(parameters=dict((k, dict(value=v)) for k,v in base_c.items()))
		d['parameters'] |= base_c['parameters']

		ii.exp_name = wandb.sweep(
			sweep   = d,
			entity  = ii.wandb_c.entity,
			program = ii.run_name,
			project = ii.project,
			name	= ii.exp_name,
		)
		# command = ['$\{env\}', 'python -u', '$\{program\}', '$\{args\}', f'--sweep_id_pseudo={ii.exp_id}']
		return [dict() for i in range(n_sweep)]
		
	def log(ii, info: Union[dict,str], create=False, path='dump/tmp/log.tmp'):
		mkdir(path)
		mode = 'w' if create else 'a'
		info = print.pformat(info)
		with open(path, mode) as f:
			f.writelines(info)

""" 
System
n_e = \sum_i charge_nuclei_i - charge = n_e
spin = n_u - n_d
n_e = n_u + n_d
n_u = 1/2 ( spin + n_e ) = 1/2 ( spin +  \sum_i charge_nuclei_i - charge)


"""

""" pyfig
wandb
wandb sync wandb/dryrun-folder-name    # to sync data stored offline


"""



""" Bone Zone

		# if ii.dist:
			# https://groups.google.com/g/cluster-users/c/VpdG0IFZ4n4
		# else:
		# 	s += [submit_cmd + ' --head True']


	#SBATCH --cpus-per-task       4
	#SBATCH --mem-per-cpu         1024
	#SBATCH --error               dump/exp/demo-12/Wvyonoa/slurm/e-%j.err
	#SBATCH --gres                gpu:RTX3090:2
	#SBATCH --job-name            demo
	#SBATCH --mail-type           FAIL
	#SBATCH --nodes               1-1
	#SBATCH --ntasks              2
	#SBATCH --output              dump/exp/demo-12/Wvyonoa/slurm/o-%j.out
	#SBATCH --partition           sm3090
	#SBATCH --time                0-01:00:00


# 
# srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label hostname &
# srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=1024 --ntasks=1 --exclusive --label hostname &
# wait
#  set | grep SLURM | while read line; do echo "# $line"; done
		# https://uwaterloo.ca/math-faculty-computing-facility/services/service-catalogue-teaching-linux/job-submit-commands-examples
		

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# module purge
# source ~/.bashrc
# module load foss
# module load CUDA/11.7.0
# # export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# conda activate lumi
# # 192GB 

# # srun  python run.py --run_name run.py --seed 808017424 --dtype float32 --n_step 2000 --log_metric_step 10 --log_state_step 10 --charge 0 --spin 0 --a [[0.0,0.0,0.0]] --a_z [4.0] --n_b 256 --n_corr 50 --n_equil 10000 --acc_target 0.5 --with_sign False --n_sv 32 --n_pv 16 --n_fb 2 --n_det 1 --terms_s_emb ['ra','ra_len'] --terms_p_emb ['rr','rr_len'] --ke_method vjp --job_type training --mail_type FAIL --partition sm3090 --nodes 1-1 --cpus_per_task 8 --time 0-01:00:00 --TMP dump/tmp --exp_id Wvyonoa --run_sweep False --user amawi --server svol.fysik.dtu.dk --git_remote origin --git_branch main --env lumi --debug False --wb_mode online --submit False --cap 3 --exp_path dump/exp/demo-12/Wvyonoa --exp_name demo --n_gpu 2 --device_type cuda --head True & 
# # srun --gres=gpu:RTX3090:1 --ntasks=1 --label --exact python run.py --run_name run.py --seed 808017424 --dtype float32 --n_step 2000 --log_metric_step 10 --log_state_step 10 --charge 0 --spin 0 --a [[0.0,0.0,0.0]] --a_z [4.0] --n_b 256 --n_corr 50 --n_equil 10000 --acc_target 0.5 --with_sign False --n_sv 32 --n_pv 16 --n_fb 2 --n_det 1 --terms_s_emb ['ra','ra_len'] --terms_p_emb ['rr','rr_len'] --ke_method vjp --job_type training --mail_type FAIL --partition sm3090 --nodes 1-1 --cpus_per_task 8 --time 0-01:00:00 --TMP dump/tmp --exp_id Wvyonoa --run_sweep False --user amawi --server svol.fysik.dtu.dk --git_remote origin --git_branch main --env lumi --debug False --wb_mode online --submit False --cap 3 --exp_path dump/exp/demo-12/Wvyonoa --exp_name demo --n_gpu 2 --device_type cuda --head False & 
# # wait 
		
#         module load foss
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

# submit_cmd = dict(
# 	sweep=f'wandb agent {ii.sweep_path_id}',
# 	python=f'python {ii.run_name} {ii.cmd}'
# )[ii.exe_mode]

export MKL_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export OMP_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1
nvidia-smi`

| tee $out_dir/py.out date "+%B %V %T.%3N


env     = f'conda activate {ii.env};',
# pyfig
def _debug_print(ii, on=False, cls=True):
		if on:
			for k in vars(ii).keys():
				if not k.startswith('_'):
					print(k, getattr(ii, k))    
			if cls:
				[print(k,v) for k,v in vars(ii.__class__).items() if not k.startswith('_')]

@property
	def wandb_cmd(ii):
		d = flat_dict(cls_to_dict(ii, sub_cls=True, ignore=['sweep',] + list(ii.sweep.parameters.keys()), add=['exp_path',]))
		d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in d.items()}
		cmd_d = {str(k).replace(" ", ""): str(v).replace(" ", "") for k,v in d.items()}
		cmd = ' '.join([f' --{k}={v}' for k,v in cmd_d.items() if v])
		return cmd

# if not re.match(ii.server, ii.hostname): # if on local, ssh to server and rerun
			#     # sys.exit('submit')
			#     print('Submitting to server \n')
			#     run_cmds([ii._git_commit_cmd, 'git push origin main --force'], cwd=ii.project_dir)
			#     run_cmds_server(ii.server, ii.user, ii._git_pull_cmd, ii.server_project_dir)  
			#     run_cmds_server(ii.server, ii.user, ii._run_single_cmd, ii.run_dir)                
			#     sys.exit(f'Submitted to server \n')
				##############################################################################

			# run_cmds([ii._git_commit_cmd, 'git push origin main'], cwd=ii.project_dir)

def cls_filter(
	cls, k: str, v, 
	ref:list|dict=None,
	is_fn=False, 
	is_sub=False, 
	is_prop=False, 
	is_hidn=False,
	ignore:list=None,
	keep = False,
):  
	
	is_builtin = k.startswith('__')
	should_ignore = k in (ignore if ignore else [])
	not_in_ref = k in (ref if ref else [k])
	
	if not (is_builtin or should_ignore or not_in_ref):
		keep |= is_hidn and k.startswith('_')
		keep |= is_sub and isinstance(v, PlugIn)
		keep |= is_fn and isinstance(v, partial)
		keep |= is_prop and isinstance(cls.__class__.__dict__[k], property)
	return keep
	
	def accumulate(ii, step: int, v_tr:dict, sync=None):
		
		assert all([k in v_tr.keys() for k in ii.dist.sync])

		v_sync = dict()
		for k,v in v_tr.copy().items():
			v_path = (ii.exchange_dir / f'{k}_{step}_{ii.dist._dist_id}').with_suffix('.pk')
			ii.lo_ve(path=v_path, data=v)

			v_path_mean = add_to_Path(v_path, '-mean')

			if ii.dist.head:
				
				n_ready = 0
				while n_ready < ii.n_gpu:
					k_path_all = list(ii.exchange_dir.glob(f'{k}_{step}*')) 
					n_ready = len(k_path_all)
					sleep(0.1)
				
				leaves = []
				for p in k_path_all:
					v_dist_i = ii.lo_ve(path=p)
					l_sub, treespec = optree.tree_flatten(v_dist_i)
					leaves += l_sub
	  
				assert len(leaves) == len(k_path_all) == ii.n_gpu

				leaves_mean = [np.stack(leaves).mean(axis=0) for leaves in zip(*leaves)]
				v_sync[k] = optree.tree_unflatten(treespec=treespec, leaves=leaves_mean)
	
				[ii.lo_ve(path=add_to_Path(p, '-mean').with_suffix('.pk'), data=v_sync[k]) for p in k_path_all]
				[p.unlink() for p in k_path_all]
				
			while not v_path_mean.exists():
				sleep(0.01)
			v_sync[k] = ii.lo_ve(path=v_path_mean)  # Speed: Only load sync vars
			v_path_mean.unlink()

		return v_sync
print('Server -> hostname', ii.server, ii.hostname, 'Place local dreams here, ...')
"""