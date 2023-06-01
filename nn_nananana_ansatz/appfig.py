# Anything that counts as configuration for this project should be in this file.
import os
from pydantic import BaseModel, Field

from .pyfig.pyfig import ModdedModel, DEVICES, DTYPES, RUNMODE
from .pyfig.plugins import Logger, Opt, OPTS, LOGMODE, RUNTYPE
from .pyfig.paths import Paths
from .pyfig.pydanmod import Pydanmod

from .systems import System

system: System = System( # looks good here, also typing issue when nested
	a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
	a_z = [4, 4],
	charge = 0,
	spin = 0,
)

class Walkers(BaseModel):

	n_b: 			int     = Field(128, description= 'number of walkers / batchsize')
	n_corr: 		int    	= Field(20, description= 'number of walker steps before resampling')
	n_equil_step: 	int 	= Field(100, description= 'warmup steps')
	acc_target: 	float   = Field(0.5, description= 'target acceptance rate')
	init_data_scale:float	= Field(0.1, description= 'scale of the initial random walkers std around the nuclei')

class Ansatzcfg(BaseModel):

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

class Pyfig(Pydanmod):
	
	seed: int           	= 42
	debug: bool         	= False

	device: DEVICES     	= DEVICES.cpu
	dtype: DTYPES          	= DTYPES.float32
	
	paths: Paths = Paths(
		project         = 'hwat',
		exp_name        = 'test',
		run_name        = 'run.py',
		exp_data_dir    = 'data',

	)

	mode: RUNMODE       	= RUNMODE.train  # adam, sgd

	n_epoch: int      		= 100
	n_step: int      		= 100

	loss: str        		= 'vmc'  # orb_mse, vmc

	logger: Logger      = Logger(
		exp_name 		= paths.exp_name, 
		entity			= os.environ.get('WANDB_ENTITY', 'WANDB_ENTITY NOT IN ENV'),
		project			= paths.project,
		n_log_metric	= 10,
		n_log_state		= -1,
		run_path		= paths.run_path,
		exp_data_dir	= paths.exp_data_dir,
	)

	opt: Opt         	= Opt(
		name = OPTS.AdamW,
	)

	walkers: Walkers = Walkers(
		n_walkers = 64,
	)

	system: System = system

	ansatzcfg: Ansatzcfg = Ansatzcfg(
		n_l 	= 3,
		ke_method 	= 'grad_grad', # inefficient, can do jit compiled version
	)



