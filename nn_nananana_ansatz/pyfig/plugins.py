from pydantic import BaseModel, Field
from pathlib import Path

from rich import print
from enum import Enum

import wandb

log = Path('.log')

class Scheduler(BaseModel):
	sch_name: 	str		= 'ExponentialLR'
	sch_max_lr:	float 	= 0.01
	sch_epochs: int 	= 1
	sch_gamma: 	float 	= 0.9999
	sch_verbose: bool 	= False

class Param(BaseModel): 
	values:     list | None     = None
	domain:     tuple | None    = None
	dtype:      type | None     = None
	log:        bool | None     = False
	step_size:  float | int | None = None
	sample:     str | None      = None # docs:Param:sample from ('uniform', )
	condition:  list | None     = None

	def get(self, key, dummy=None):
		return getattr(self, key, dummy)

class Sweep:
	sweep_name: 	str		= 'study'	
	n_trials: 		int		= 20
	parameters: 	dict 	= dict(
		opt_name		=	Param(values=['AdaHessian', 'RAdam'], dtype=str),
		hessian_power	= 	Param(values=[0.5, 0.75, 1.], dtype=float, condition=['AdaHessian',]),
		weight_decay	= 	Param(domain=(0.0001, 1.), dtype=float, condition=['AdaHessian',]),
		lr				=	Param(domain=(0.0001, 1.), log=True, dtype=float),
	)

def d_to_wb(d:dict, parent='', sep='.', items:list=None) -> dict:
	items = items or []
	for k, v in d.items():
		if callable(v):
			continue
		elif isinstance(v, Path):
			parent = f'path{sep}'
		elif isinstance(v, dict):
			items.extend(d_to_wb(v, parent=k, items=items).items())
		name = parent + k
		items.append((name, v))
	return dict(items)

class Data(BaseModel):

	n_b: 			int     = Field(128, description= 'number of walkers / batchsize')
	n_corr: 		int    	= Field(20, description= 'number of walker steps before resampling')
	acc_target: 	float   = Field(0.5, description= 'target acceptance rate')
	init_data_scale:float	= Field(0.1, description= 'scale of the initial random walkers std around the nuclei')

	@property
	def n_equil_step(self):
		return 10000 // self.n_corr

class Model(BaseModel):

	n_l: 			int     = Field(3, description= 'number of fermi block layers')
	n_sv: 			int     = Field(32, description= 'number hidden nodes in the single electron layers')
	n_pv: 			int     = Field(16, description= 'number hidden nodes in the pairwise interaction layers')
	n_fb: 			int     = Field(3, description= 'number of Fermi Blocks')

	n_det:          int     = Field(4, description= 'number of determinants')
	n_final_out:	int     = Field(1, description= 'number of final output nodes')
	
	terms_s_emb:    list    = Field(['ra', 'ra_len'], description= 'terms to include in the single electron embedding')
	terms_p_emb:    list    = Field(['rr', 'rr_len'], description= 'terms to include in the pairwise interaction embedding')
	ke_method:      str     = Field('grad_grad', description= '(torch) method to compute the kinetic energy')
	
	@property
	def n_emb(self): 
		return len(self.terms_s_emb) + len(self.terms_p_emb)

	@property
	def n_fbv(self):
		return self.n_sv*3 + self.n_pv*2



class RUNTYPE(str, Enum):
	groups = 'groups'
	sweeps = 'sweeps'
	runs = 'runs'

class LOGMODE(str, Enum):
	online = 'online'
	disabled = 'disabled'
	offline = 'offline'

class Logger(BaseModel):

	entity: 		str = Field(..., description= 'wandb entity')
	project:		str
	exp_name: 		str
	run_path: 		Path | str
	exp_data_dir: 	Path | str

	log_interval: 	int = 10
	save_interval: 	int = 100
	resume_checkpoint: str = ''

	n_log_metric: 	int = 5
	n_log_state: 	int = 1

	log_mode: 		LOGMODE = LOGMODE.online
	log_model: 		bool = True

	runtype: RUNTYPE 	= RUNTYPE.runs

	job_type: str      	= 'dev' # 'dev', 'train', 'eval', 'test'

	@property
	def log_run_type(self):
		return f'{self.runtype}/{self.exp_name}/workspace'
	
	@property
	def log_run_url(self):
		return f'{self.project}/{self.log_run_type}'
	
	def get_c(self, 
		d:dict, 
		parent = '', 
		sep = '.', 
		items: list= None
	) -> dict:
		return d_to_wb(d, parent, sep, items)

	def start(self, d: dict, run_id: str, tags: list):
		self.run = wandb.init(
			project=self.project,
			group=self.exp_name,
			dir=self.exp_data_dir,
			entity=self.entity,
			mode=self.log_mode,
			config=self.get_c(d, parent='', sep='.'),
			id=run_id,
			tags=tags,
			reinit=self.run is not None,
		)

class OPTS(str, Enum):
	Adahessian  : str = 'AdaHessian'
	RAdam       : str = 'RAdam'
	Apollo      : str = 'Apollo'
	AdaBelief   : str = 'AdaBelief'
	LBFGS       : str = 'LBFGS'
	Adam        : str = 'Adam'
	AdamW       : str = 'AdamW'


class Opt(BaseModel):
	
	name: OPTS = OPTS.AdamW
	
	lr:  			float 			= 1e-3
	init_lr: 		float 			= 1e-3

	betas:			list[float]		= [0.9, 0.999]
	beta: 			float 			= 0.9
	warm_up: 		int 			= 100
	eps: 			float 			= 1e-4
	weight_decay: 	float 			= 0.0
	hessian_power: 	float 			= 1.0

	lr_anneal_steps:int 			= 1000
	ema_rate: 		float 			= "0.9999"







