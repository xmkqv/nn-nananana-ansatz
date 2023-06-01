from uuid import UUID
from time import time

def gen_rnd(seed: int = None, n_char: int = 10) -> str:
    """ generate random string of length """
    if seed is None:
        seed = int(time())
    return UUID(int= seed).hex[:n_char]


import torch
import torch_optimizer
from functools import partial


def get_opt(
	*,
	name: str = None,
	lr: float = None,
	betas: tuple[float] = None,
	beta: float = None,
	eps: float = None,
	weight_decay: float = None,
	hessian_power: float = None,
	default: str = 'RAdam',
	**kw, 
) -> torch.optim.Optimizer:

    if name.lower() == 'RAdam'.lower():
        return partial(torch.optim.RAdam, lr=lr)

    elif name.lower() == 'Adahessian'.lower():
        return partial(
            torch_optimizer.Adahessian,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
    elif name.lower() == 'AdaBelief'.lower():
        return partial(
            torch_optimizer.AdaBelief,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            weight_decouple=False,
            fixed_decay=False,
            rectify=False,
        )
    elif name.lower() == 'LBFGS'.lower():
        return partial(
            torch.optim.LBFGS,
            lr=lr,
        )

    elif name.lower() == 'AdamW'.lower():
        return partial(
            torch.optim.AdamW,
            lr=lr,
        )

    elif name.lower() == 'Apollo'.lower():
        return partial(
            torch_optimizer.Apollo,
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            warmup=50,
            init_lr=1e-6,
        )
    else:
        print(f'!!! opt {name} not available, returning {default}')
        return get_opt(name=default, lr=0.001)



def get_scheduler(
	sch_name: str = None,
	sch_max_lr: float = None,
	sch_gamma: float = None,
	sch_epochs: int = None, 
	sch_verbose: bool = None,
	n_scheduler_step: int = None,
	**kw,
) -> torch.optim.lr_scheduler._LRScheduler:

    if sch_name.lower() == 'OneCycleLR'.lower():
        return partial(
            torch.optim.lr_scheduler.OneCycleLR,
            max_lr=sch_max_lr,
            steps_per_epoch=n_scheduler_step,
            epochs=sch_epochs,
        )
    elif sch_name.lower() == 'ExponentialLR'.lower():
        return partial(
            torch.optim.lr_scheduler.ExponentialLR,
            gamma=sch_gamma,
            last_epoch=-1,
            verbose=sch_verbose,
        )
    else:
        print(f'!!! Scheduler {sch_name} not available, returning DummyScheduler ')
        class DummySchedule(torch.optim.lr_scheduler._LRScheduler):
        	def step(self, epoch= None):
        		pass
        return DummySchedule()

