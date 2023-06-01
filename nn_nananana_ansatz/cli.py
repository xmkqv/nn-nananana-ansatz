import typer
from pathlib import Path
import run

app = typer.Typer()

@app.command()
def main(Paths__project: str, Paths__exp_name: str, Paths__exp_id: str, System__a: float, System__a_z: int, Logger__entity: str, Logger__project: str, Logger__exp_name: str, Logger__run_path: Path|str, Logger__exp_data_dir: Path|str, seed: int = 42, device: run.DEVICES = run.DEVICES.cpu, dtype: run.DTYPES = run.DTYPES.float32, Paths__run_name: Path = 'run.py', Paths__lo_ve_path: str = '', Paths__home: Path = '.', Paths__exp_dir: Path = 'exp', mode: run.RUNMODE = run.RUNMODE.train, n_step: int = 100, loss: str = 'vmc', compute_energy: bool = False, dashboard: bool = False, System__charge: int = 0, System__spin: int = 0, Ansatz__n_l: int = 3, Ansatz__n_sv: int = 32, Ansatz__n_pv: int = 16, Ansatz__n_fb: int = 3, Ansatz__n_det: int = 4, Ansatz__n_final_out: int = 1, Ansatz__terms_s_emb: list = None, Ansatz__terms_p_emb: list = None, Ansatz__ke_method: str = 'grad_grad', Walkers__n_b: int = 128, Walkers__n_corr: int = 20, Walkers__acc_target: float = 0.5, Walkers__init_data_scale: float = 0.1, Logger__n_log_metric: int = 5, Logger__n_log_state: int = 1, Logger__log_mode: run.LOGMODE = run.LOGMODE.online, Logger__run_mode: run.RUNTYPE = run.RUNTYPE.runs, Logger__job_type: str = '', Opt__name: str = 'AdamW', Opt__lr: float = 0.001, Opt__init_lr: float = 0.001, Opt__betas: float = None, Opt__beta: float = 0.9, Opt__warm_up: int = 100, Opt__eps: float = 0.0001, Opt__weight_decay: float = 0.0, Opt__hessian_power: float = 1.0):

	if Ansatz__terms_s_emb is None:
		Ansatz__terms_s_emb = ['ra', 'ra_len']
	if Ansatz__terms_p_emb is None:
		Ansatz__terms_p_emb = ['rr', 'rr_len']
	if Opt__betas is None:
		Opt__betas = [0.9, 0.999]
		paths = run.Paths(
			project = Paths__project,
			exp_name = Paths__exp_name,
			exp_id = Paths__exp_id,
			run_name = Paths__run_name,
			lo_ve_path = Paths__lo_ve_path,
			home = Paths__home,
			exp_dir = Paths__exp_dir
	)
		system = run.System(
			a = System__a,
			a_z = System__a_z,
			charge = System__charge,
			spin = System__spin
	)


		logger = run.Logger(
			entity = Logger__entity,
			project = Logger__project,
			exp_name = Logger__exp_name,
			run_path = Logger__run_path,
			exp_data_dir = Logger__exp_data_dir,
			n_log_metric = Logger__n_log_metric,
			n_log_state = Logger__n_log_state,
			log_mode = Logger__log_mode,
			run_mode = Logger__run_mode,
			job_type = Logger__job_type
	)


		pyfig = run.Pyfig(
			seed = seed,
			device = device,
			dtype = dtype,
			mode = mode,
			n_step = n_step,
			loss = loss,
			compute_energy = compute_energy,
			dashboard = dashboard,
			paths = paths,
			system = system,
			logger = logger,
			ansatz = ansatz,
			walkers = walkers,
			opt = opt
	)


		ansatz = run.Ansatz(
			n_l = Ansatz__n_l,
			n_sv = Ansatz__n_sv,
			n_pv = Ansatz__n_pv,
			n_fb = Ansatz__n_fb,
			n_det = Ansatz__n_det,
			n_final_out = Ansatz__n_final_out,
			terms_s_emb = Ansatz__terms_s_emb,
			terms_p_emb = Ansatz__terms_p_emb,
			ke_method = Ansatz__ke_method
	)


		walkers = run.Walkers(
			n_b = Walkers__n_b,
			n_corr = Walkers__n_corr,
			acc_target = Walkers__acc_target,
			init_data_scale = Walkers__init_data_scale
	)


		opt = run.Opt(
			name = Opt__name,
			lr = Opt__lr,
			init_lr = Opt__init_lr,
			betas = Opt__betas,
			beta = Opt__beta,
			warm_up = Opt__warm_up,
			eps = Opt__eps,
			weight_decay = Opt__weight_decay,
			hessian_power = Opt__hessian_power
	)


	run.run(pyfig)
