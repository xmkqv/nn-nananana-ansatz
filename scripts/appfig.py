# Anything that counts as configuration for this project should be in this file.

from pydantic import BaseModel, Field

from nn_nananana_ansatz.appfig import Walkers, Ansatzcfg, Pyfig
from nn_nananana_ansatz.systems import System

class Walkers(Walkers):
	...

class Ansatzcfg(Ansatzcfg):
	...

class Pyfig(Pyfig):
	...


from loguru import logger
from typing import Callable
import torch

from nn_nananana_ansatz.hwat import update_grads, update_params

from torch import nn

def compute_grads(
	c: Pyfig, 
	loss_fn: Callable,
	data: torch.Tensor,
	model: nn.Module,
	step: int,
):
	loss, v_d = loss_fn(model, data, step)

	if loss: # if loss is not None or 0.0
			
		logger.trace(loss) # SHOULD BE NON-ZERO!

		create_graph = c.opt.name.lower() == 'AdaHessian'.lower()
		loss.backward(create_graph= create_graph)
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0) # torch stuff: clip gradients
		
		grads = {k: p.grad.detach() for k,p in model.named_parameters()}
		v_d |= dict(loss= loss.item(), grads= grads)

	return v_d




