
import functools
import torch as th

class ModelCfg(BaseModel):

	clip:           float   = Field(1.0, description= 'gradient clipping')
	n_l: 			int     = Field(3, description= 'number of fermi block layers')
	

class Model(nn.Module):

	c: ModelCfg

	def __init__(self, c: ModelCfg):
		super().__init__()
		self.c = c

	def run_step(self, batch, cond):
		self.forward_backward(batch, cond)
		if self.clip:
			th.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.clip)
		if self.use_fp16:
			self.optimize_fp16()
		else:
			self.optimize_normal()
		self.log_step()

	def forward_backward(self, batch, cond):
		zero_grad(self.model_params)
		for i in range(0, batch.shape[0], self.microbatch):
			micro = batch[i : i + self.microbatch].to(dist_util.dev())
			micro_cond = {
				k: v[i : i + self.microbatch].to(dist_util.dev())
				for k, v in cond.items()
			}
			last_batch = (i + self.microbatch) >= batch.shape[0]
			t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
			compute_losses = functools.partial(
				self.diffusion.training_losses,
				self.ddp_model,
				micro,
				t,
				model_kwargs=micro_cond,
				max_num_mask_frames=self.max_num_mask_frames,
				mask_range=self.mask_range,
				uncondition_rate=self.uncondition_rate,
				exclude_conditional=self.exclude_conditional,
			)
			if last_batch or not self.use_ddp:
				losses = compute_losses()
			else:
				with self.ddp_model.no_sync():
					losses = compute_losses()
			if isinstance(self.schedule_sampler, LossAwareSampler):
				self.schedule_sampler.update_with_local_losses(
					t, losses["loss"].detach()
				)
			loss = (losses["loss"] * weights).mean()
			log_loss_dict(
				self.diffusion, t, {k: v * weights for k, v in losses.items()}
			)
			loss = loss / self.accumulation_steps
			if self.use_fp16:
				loss_scale = 2 ** self.lg_loss_scale
				(loss * loss_scale).backward()
			else:
				loss.backward()

	def optimize_fp16(self):
		if any(not th.isfinite(p.grad).all() for p in self.model_params):
			self.lg_loss_scale -= 1
			logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
			return

		model_grads_to_master_grads(self.model_params, self.master_params)
		self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
		self._log_grad_norm()
		self._anneal_lr()
		self.opt.step()
		for rate, params in zip(self.ema_rate, self.ema_params):
			update_ema(params, self.master_params, rate=rate)
		master_params_to_model_params(self.model_params, self.master_params)
		self.lg_loss_scale += self.fp16_scale_growth

	def optimize_normal(self):
		self._log_grad_norm()
		self._anneal_lr()
		self.opt.step()
		for rate, params in zip(self.ema_rate, self.ema_params):
			update_ema(params, self.master_params, rate=rate)



# 


import torch as th
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from accelerate import Accelerator

import wandb

from nn_nananana_ansatz.hwat import Ansatz as Model, Scf

pl.seed_everything(c.seed)

accelerator = Accelerator(
    device_placement = True,  # Automatically places tensors on the proper device
    fp16 = True,  # Enables automatic mixed precision training (AMP)
    cpu=True,  # Forces the use of CPU even when GPUs are available
    split_batches=True,  # Splits the batches on the CPU before sending them to the device
    num_processes=1,  # Number of processes to use for distributed training (1 means no distributed training)
    local_rank=0,  # Local rank of the process (for distributed training)
)


model = Model(c.modelcfg)

wandb_logger = WandbLogger(
    project		= c.paths.project,
    log_model	= c.logger.log_model
)

wandb_logger.watch(model)

checkpoint_callback = ModelCheckpoint(
    dirpath				= c.model_save_path,
    filename			= "model-{epoch:02d}-{val_loss:.2f}",
    save_top_k			= 1,
    monitor				= "val_loss",
    mode				= "min",
    save_weights_only	= True,
    save_last			= True,
    period				= c.checkpoint_frequency,
    verbose				= True,
)

trainer = pl.Trainer(
    max_epochs	= c.n_epoch,
    accelerator	= accelerator,
    logger		= wandb_logger,
    callbacks	= [checkpoint_callback],
    gpus		= th.cuda.device_count(),
)

trainer.fit(model, train_loader, val_loader)

wandb.finish()

