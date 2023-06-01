from typing import Callable

import torch 
from torch.jit import Final
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset

import functorch
import numpy as np
import pandas as pd
from functools import partial

from .appfig import Pyfig, Walkers, Ansatzcfg


def fb_block(s_v: Tensor, p_v: Tensor, n_u: int, n_d: int):
	n_e = n_u + n_d

	fb_f = [s_v,]
	for s_x in torch.split(s_v, (n_u, n_d), dim=1):
		mean_s_x = s_x.mean(dim=1, keepdim=True).tile((1, n_e, 1))
		fb_f += [mean_s_x] # two element list of (n_b, n_e, n_sv) tensor
	
	for p_x in torch.split(p_v, (n_u, n_d), dim=1):
		mean_p_x = p_x.mean(dim=1)
		fb_f += [mean_p_x] # two element list of (n_b, n_e, n_sv) tensor

	return torch.cat(fb_f, dim=-1) # (n_b, n_e, 3n_sv+2n_pv)

class Ansatz(nn.Module):

	n_d: Final[int]                 	# number of down electrons
	n_det: Final[int]               	# number of determinants
	n_fb: Final[int]                	# number of feedforward blocks
	n_pv: Final[int]                	# latent dimension for 2-electron
	n_sv: Final[int]                	# latent dimension for 1-electron
	n_a: Final[int]                 	# nuclei positions
	with_sign: Final[bool]          	# return sign of wavefunction
	debug: Final[bool]          		# return sign of wavefunction
	a: Tensor      

	def __init__(
		ii, 
		*, 
		n_e, 
		n_u, 
		n_d, 
		n_det, 
		n_fb, 
		n_pv, 
		n_sv, 
		n_final_out,
		a: Tensor, 
		mol,
		mo_coef: Tensor,
		with_sign=False, **kwargs):


		super().__init__()
  
		ii.with_sign = with_sign      # return sign of wavefunction

		ii.n_e: Final[int] = n_e      					# number of electrons
		ii.n_u: Final[int] = n_u                  # number of up electrons
		ii.n_d: Final[int] = n_d                  # number of down electrons
		ii.n_det = n_det              # number of determinants
		ii.n_fb = n_fb                # number of feedforward blocks
		ii.n_pv = n_pv                # latent dimension for 2-electron
		ii.n_sv = n_sv                # latent dimension for 1-electron
		ii.n_sv_out = 3*n_sv+2*n_pv
		ii.n_final_out = n_final_out  # number of output channels

		ii.n_a = len(a)

		n_p_in = 4
		n_s_in = 4*ii.n_a
	
		s_size = [(n_s_in*3 + n_p_in*2, n_sv),] + [(ii.n_sv_out, n_sv),]*n_fb
		p_size = [(n_p_in, n_pv),] + [(n_pv, n_pv),]*n_fb
		print('model fb layers: \n', s_size, p_size)

		ii.s_lay = nn.ModuleList([nn.Linear(*dim) for dim in s_size])
		ii.p_lay = nn.ModuleList([nn.Linear(*dim) for dim in p_size])

		ii.v_after = nn.Linear(ii.n_sv_out, ii.n_sv)

		ii.w_spin_u = nn.Linear(ii.n_sv, ii.n_u * ii.n_det)
		ii.w_spin_d = nn.Linear(ii.n_sv, ii.n_d * ii.n_det)
		ii.w_spin = [ii.w_spin_u, ii.w_spin_d]

		ii.w_final = nn.Linear(ii.n_det, ii.n_final_out, bias=False)
		ii.w_final.weight.data.fill_(1.0 / ii.n_det)

		device, dtype = a.device, a.dtype
		a = a.detach()
		eye = torch.eye(ii.n_e)[None, :, :, None].requires_grad_(False).to(device, dtype)
		ones = torch.ones((1, ii.n_det)).requires_grad_(False).to(device, dtype)
		zeros = torch.zeros((1, ii.n_det)).requires_grad_(False).to(device, dtype)
		mo_coef = mo_coef.detach().to(device, dtype)
		
		ii.register_buffer('a', a, persistent=False)    # tensor
		ii.register_buffer('eye', eye, persistent=False) # ! persistent moves with model to device etc
		ii.register_buffer('ones', ones, persistent=False)
		ii.register_buffer('zeros', zeros, persistent=False)
		ii.register_buffer('mo_coef', mo_coef, persistent=False) # (n_b, n_e, n_mo)

		ii.mol = mol

	def forward(self, r: Tensor) -> Tensor:
		
		if r.ndim in (1, 2):
			r = r.reshape(-1, self.n_e, 3)

		n_b = r.shape[0]

		ra = r[:, :, None, :] - self.a[None, None, :, :]
		ra_u, ra_d = torch.split(ra, [self.n_u, self.n_d], dim=1)

		s_v, p_v = self.compute_embedding(r, ra)
		s_v = self.compute_stream(s_v, p_v)

		s_u, s_d = torch.split(s_v, [self.n_u, self.n_d], dim=1)

		orb_u = self.compute_orb_from_stream(s_u, ra_u, spin=0)
		orb_d = self.compute_orb_from_stream(s_d, ra_d, spin=1)

		sign_u, logdet_u = torch.linalg.slogdet(orb_u)
		sign_d, logdet_d = torch.linalg.slogdet(orb_d)

		sign = sign_u * sign_d
		logdet = logdet_u + logdet_d

		maxlogdet, _ = torch.max(logdet, dim=-1, keepdim=True) # (n_b, 1), (n_b, 1)

		sub_det = sign * torch.exp(logdet - maxlogdet) 

		psi_ish: Tensor = self.w_final(sub_det)

		sign_psi = torch.sign(psi_ish)  

		log_psi = torch.log(torch.absolute(psi_ish.squeeze())) + maxlogdet.squeeze()

		return (log_psi, sign_psi) if self.with_sign else log_psi

	def compute_orb(self, r: Tensor) -> tuple[Tensor, Tensor]:

		ra = r[:, :, None, :] - self.a[None, None, :, :]
		ra_u, ra_d = torch.split(ra, [self.n_u, self.n_d], dim=1)
		s_v, p_v = self.compute_embedding(r, ra)
		s_v = self.compute_stream(s_v, p_v)
		s_u, s_d = torch.split(s_v, [self.n_u, self.n_d], dim=1)

		orb_u = self.compute_orb_from_stream(s_u, ra_u, spin= 0)
		orb_d = self.compute_orb_from_stream(s_d, ra_d, spin= 1)

		### under construction ###
		# makes sure the final weight is in the preing loop for distribution
		zero = ((orb_u*0.0).sum(dim=(-1,-2)) + (orb_d*0.0).sum(dim=(-1,-2)))
		zero = self.w_final(zero)[..., None, None]
		### under construction ###

		return orb_u + zero, orb_d + zero

	def compute_embedding(self, r: Tensor, ra: Tensor) -> tuple[Tensor, Tensor]:
		n_b, n_e, _ = r.shape

		ra_len = torch.linalg.vector_norm(ra, dim=-1, keepdim=True) # (n_b, n_e, n_a, 1)

		rr = r[:, None, :, :] - r[:, :, None, :] # (n_b, n_e, n_e, 1)
		rr_len = torch.linalg.vector_norm(rr + self.eye, dim=-1, keepdim=True) 

		s_v = torch.cat([ra, ra_len], dim=-1).reshape(n_b, self.n_e, -1)
		p_v = torch.cat([rr, rr_len], dim=-1) # (n_b, n_e, n_e, 4)

		return s_v, p_v

	def compute_stream(self, s_v: Tensor, p_v: Tensor):
			
		s_v_block = fb_block(s_v, p_v, self.n_u, self.n_d)

		for l_i, (s_lay, p_lay) in enumerate(zip(self.s_lay, self.p_lay)):

			s_v_tmp = torch.tanh(s_lay(s_v_block))
			s_v = s_v_tmp + s_v if l_i else s_v_tmp

			p_v_tmp = torch.tanh(p_lay(p_v))
			p_v = p_v_tmp + p_v if l_i else p_v_tmp

			s_v_block = fb_block(s_v, p_v, self.n_u, self.n_d)

		s_v: Tensor = torch.tanh(self.v_after(s_v_block))
		return s_v

	def compute_orb_from_stream(self, s_v: Tensor, ra_v: Tensor, spin: int) -> Tensor:
		n_b, n_spin, _ = s_v.shape

		s_w = self.w_spin[spin](s_v).reshape(n_b, n_spin, n_spin, self.n_det)

		exp = torch.exp(-torch.linalg.vector_norm(ra_v, dim=-1)) # (n_b, n_spin(r), n_a)
		sum_a_exp = exp.sum(dim= -1, keepdim= True)[..., None] # n_b, n_u(r)

		orb = s_w * sum_a_exp # (n_b, n_u(r), n_u(orb), n_det) 
		return orb.transpose(-1, 1) # (n_b, n_det, n_u(r), n_u(orb))

	def compute_hf_orb(self, r: Tensor) -> np.ndarray:
		"""
		r: 		(n_b, n_e, 3)
		r_hf: 	(n_b*n_e, 3)
		mo_coef: (2, n_mo, n_ao)
		ao: 	(n_b*n_e, n_ao)
		mo: 	(n_b, n_e, n_mo)
		"""
		n_b, n_e, _ = r.shape

		# https://github.com/deepmind/ferminet/blob/7d4ef0ad52f84792c2a5e0c64a9cd15e282724aa/ferminet/pretrain.py
		# - orbitals are the orbitals 

		r_hf = r.detach().cpu().numpy().reshape(-1, 3)

		ao = self.mol.eval_gto('GTOval_cart', r_hf)

		mo_coef = self.mo_coef.detach().cpu().numpy().astype(np.float64)

		mo = ao[None] @ mo_coef # (2, n_e*n_b, n_mo)
		mo = mo.reshape(mo.shape[0], n_b, n_e, mo.shape[-1]) # (2, n_b, n_e, n_mo)
		mo = mo[:, :, None, :, :].repeat(self.n_det, axis= 2)

		return mo[0][..., :self.n_u, :self.n_u], mo[1][..., self.n_u:, :self.n_d]
	

compute_vv = lambda v_i, v_j: torch.unsqueeze(v_i, dim=-2)-torch.unsqueeze(v_j, dim=-3)

def compute_emb(r: Tensor, terms: list, a=None):  
	dtype, device = r.dtype, r.device
	n_e, _ = r.shape
	eye = torch.eye(n_e)[..., None]

	z = []  
	if 'r' in terms:  
		z += [r]  
	if 'r_len' in terms:  
		z += [torch.linalg.norm(r, dim=-1, keepdims=True)]  
	if 'ra' in terms:  
		z += [(r[:, None, :] - a[None, ...]).reshape(n_e, -1)]  
	if 'ra_len' in terms:  
		z += [torch.linalg.norm(r[:, None, :] - a[None, ...], dim=-1)]
	if 'rr' in terms:
		z += [compute_vv(r, r)]
	if 'rr_len' in terms:  # 2nd order derivative of norm is undefined, so use eye
		z += [torch.linalg.norm(compute_vv(r, r)+eye, dim=-1, keepdims=True)]
	return torch.concatenate(z, dim=-1)

### energy ###

def compute_pe_b(r, a= None, a_z= None):

	rr = torch.unsqueeze(r, -2) - torch.unsqueeze(r, -3)
	rr_len = torch.linalg.norm(rr, dim= -1)
	pe = torch.tril(1./rr_len, diagonal= -1).sum((-1,-2))

	if a is not None:
		ra = r[:, :, None, :] - a[None, None, :, :]
		ra_len = torch.linalg.norm(ra, dim= -1)
		pe -= (a_z / ra_len).sum((-1,-2))

		if len(a_z) > 1:
			aa = a[:, None, :] - a[None, :, :]
			aa_len = torch.linalg.norm(aa, dim= -1)
			a_z2 = a_z[:, None] * a_z[None, :]
			pe += torch.tril(a_z2 / aa_len, diagonal= -1).sum((-1,-2))

	return pe.squeeze()  

# torch:issue !!! torch.no_grad does not work with torch.autograd.grad 

def compute_ke_b(
	model: nn.Module, 
	model_fn: Callable,
	r: Tensor,
	ke_method='vjp', 
	elements=False
):
	dtype, device = r.dtype, r.device
	n_b, n_e, n_dim = r.shape
	n_jvp = n_e * n_dim

	params = [p.detach() for p in model.parameters()]
	buffers = list(model.buffers())
	model_rv = lambda _r: model_fn(params, buffers, _r).sum()

	ones = torch.ones((n_b,), device=device, dtype=dtype)
	eyes = torch.eye(n_jvp, dtype=dtype, device=device)[None].repeat((n_b, 1, 1))

	r_flat = r.reshape(n_b, n_jvp).detach().contiguous().requires_grad_(True)

	assert r_flat.requires_grad
	for p in params:
		assert not p.requires_grad

	def ke_grad_grad_method(r_flat):

		def grad_fn(_r: Tensor):
			lp: Tensor = model(_r)
			g = torch.autograd.grad(lp.sum(), _r, create_graph= True)[0]
			return g

		def grad_grad_fn(_r: Tensor):
			g = grad_fn(_r)
			ggs = [torch.autograd.grad(g[:, i], _r, grad_outputs=ones, retain_graph=True)[0] for i in range(n_jvp)]
			ggs = torch.stack(ggs, dim=-1)
			return torch.diagonal(ggs, dim1=1, dim2=2)

		g = grad_fn(r_flat)
		gg = grad_grad_fn(r_flat)
		return g, gg

	def ke_vjp_method(r_flat):
		grad_fn = functorch.grad(model_rv)
		g, fn = functorch.vjp(grad_fn, r_flat)
		gg = torch.stack([fn(eyes[..., i])[0][:, i] for i in range(n_jvp)], dim=-1)
		return g, gg

	def ke_jvp_method(r_flat):
		grad_fn = functorch.grad(model_rv)
		jvp_all = [functorch.jvp(grad_fn, (r_flat,), (eyes[:, i],)) for i in range(n_jvp)]  # (grad out, jvp)
		g = torch.stack([x[:, i] for i, (x, _) in enumerate(jvp_all)], dim=-1)
		gg = torch.stack([x[:, i] for i, (_, x) in enumerate(jvp_all)], dim=-1)
		return g, gg

	ke_function = dict(
		grad_grad	= ke_grad_grad_method, 
		vjp			= ke_vjp_method, 
		jvp			= ke_jvp_method
	)[ke_method]

	g, gg = ke_function(r_flat) 

	if elements:
		return g, gg

	e_jvp = gg + g**2
	return -0.5 * e_jvp.sum(-1)


def keep_around_points(r, points, l=1.):
	""" points = center of box each particle kept inside. """
	""" l = length side of box """
	r = r - points[None, ...]
	r = r/l
	r = torch.fmod(r, 1.)
	r = r*l
	r = r + points[None, ...]
	return r


def get_center_points(a: Tensor, a_z: Tensor, charge: int):
	""" from a set of centers, selects in order where each electron will start """
	""" loop concatenate pattern """
	n_atom = len(a)
	
	assert len(a_z) == n_atom, "a_z must be same length as a"
	
	charge = torch.tensor(charge, dtype= a_z.dtype, device= a_z.device)
	sign = torch.sign(charge)
	for i in range(torch.abs(charge.to(torch.int64)).item()):
		idx = i % n_atom
		a_z[idx] -= sign 
	
	r_cen = None
	placed = torch.zeros_like(a_z)
	idx = 0
	while torch.any(placed < a_z):
		if placed[idx] < a_z[idx]:
			center = a[idx][None]
			r_cen = center if r_cen is None else torch.cat([r_cen, center])
			placed[idx] += 1
		idx = (idx+1) % n_atom
	return r_cen

def init_r(center_points: Tensor, std=0.1):
	""" init r on different gpus with different rngs """
	return center_points + torch.randn_like(center_points)*std
	# """ loop concatenate pattern """
	# sub_r = [center_points + torch.randn((n_b,n_e,3), device=device, dtype=dtype)*std for i in range(n_device)]
	# return torch.stack(sub_r, dim=0) if len(sub_r)>1 else sub_r[0][None, ...]


def is_a_larger(a: Tensor, b: Tensor) -> Tensor:
	""" given a condition a > b returns 1 if true and 0 if not """
	a = torch.where(torch.isnan(a), 1., a)
	v = a - b # + ve when a bigger
	a_is_larger = (torch.sign(v) + 1.) / 2. # 1. when a bigger
	b_is_larger = (a_is_larger - 1.) * -1. # 1. when b bigger
	return a_is_larger, b_is_larger

@torch.no_grad()
def sample_b(
	model: nn.Module = None, 
	data: Tensor = None, 
	deltar: Tensor = 0.02, 
	n_corr: int = 10,
	acc_target: float = 0.5,
	**kwargs,
):

	""" metropolis hastings sampling with automated step size adjustment 
	!!upgrade .round() to .floor() for safe masking """
	device, dtype = data.device, data.dtype

	p_0: Tensor = 2. * model(data) # logprob
	p_start = torch.exp(p_0)**2

	deltar = torch.clip(deltar, 0.01, 1.)
	deltar_1 = deltar + 0.01 * torch.randn_like(deltar)
	
	acc = torch.zeros_like(deltar_1)
	acc_all = torch.zeros_like(deltar_1)

	for dr_test in [deltar, deltar_1]:

		acc_test_all = []

		for _ in torch.arange(1, n_corr + 1):
			
			data_1 = data + torch.randn_like(data, device= device, dtype= dtype, layout= torch.strided) * dr_test
			p_1 = 2. * model(data_1)

			alpha = torch.log(torch.rand_like(p_1, device= device, dtype= dtype))
			ratio = p_1 - p_0
			
			a_larger, b_larger = is_a_larger(ratio, alpha)
			
			p_0		= p_1 		* a_larger	 				+ p_0	* b_larger
			data 	= data_1 	* a_larger[:, None, None] 	+ data	* b_larger[:, None, None]

			acc_test_all += [torch.mean(a_larger, dim= 0, keepdim= True)] # docs:torch:knowledge keepdim requires dim=int|tuple[int]

		acc_test = torch.mean(torch.stack(acc_test_all), dim= 0)

		del data_1
		del p_1
		del alpha
		del a_larger
		del b_larger
		del acc_test_all

		a_larger, b_larger = is_a_larger((acc_target - acc).abs(), (acc_target - acc_test).abs())

		acc 	= acc_test	* a_larger + acc	* b_larger
		deltar 	= dr_test	* a_larger + deltar	* b_larger

		acc_all += acc_test
	
	return dict(data= data, acc= acc_all/2., deltar= deltar, p_start= p_start)

def get_starting_deltar(sample_b: Callable, data: Tensor, acc_target: float= 0.5):
	
	deltar_domain = torch.logspace( # from base^start to base^end in delta_steps 
		-3, 0, steps= 10, base= 10.,
		device= data.device, dtype= data.dtype, requires_grad= False,
	)[:, None]
	
	acc_all = torch.zeros_like(deltar_domain)
	diff_acc_all = torch.zeros_like(deltar_domain)
	for i, deltar in enumerate(deltar_domain):
		v_d = dict(data= data, deltar= deltar)
		for n in range(1, 11):
			v_d = sample_b(**v_d)
			acc_all[i] += v_d['acc'].mean()
			diff_acc_all[i] += (acc_target - v_d['acc']).abs().mean()

	print(pd.DataFrame({
		'deltar': deltar_domain.squeeze().detach().cpu().numpy(),
		'acc': acc_all.squeeze().detach().cpu().numpy() / float(n),
		'diff': diff_acc_all.squeeze().detach().cpu().numpy() / float(n),
	}).to_markdown())
	
	return deltar_domain[torch.argmin(diff_acc_all)]



from pyscf import gto, scf
from nn_nananana_ansatz.systems import System

class PyfigDataset(Dataset):

	def __init__(ii, c: Pyfig, system: System, walkers: Walkers, state: dict=None):

		state = state or {}

		a = torch.tensor(system.a, requires_grad= False)
		a_z = torch.tensor(system.a_z, requires_grad= False)
		ii.n_step = c.n_step
		ii.n_corr = walkers.n_corr
		ii.mode = c.mode
		ii.n_b = walkers.n_b

		print('hwat:dataset: init')
		center_points = get_center_points(a, a_z, system.charge).detach()
		device, dtype = center_points.device, center_points.dtype
		print('hwat:dataset:init: center_points')
		print(center_points)

		def init_data(n_b, trailing_shape) -> torch.Tensor:
			shift = torch.randn(size=(n_b, *trailing_shape)).requires_grad_(False).to(device, dtype)
			return center_points + walkers.init_data_scale * shift 

		data = init_data(ii.n_b, center_points.shape)

		data_loaded = state.get('data')
		if data_loaded is not None: 
			print('dataset: loading data from state')
			data = data_loaded
		ii.data = data.requires_grad_(False).to(device, dtype)

		print('dataset:len ', c.n_step)

	def init_dataset(self, c: Pyfig, device= None, dtype= None, model= None, **kw) -> torch.Tensor:

		self.data = self.data.to(device= device, dtype= dtype)
		print(
			'dataset:init_dataset: data ',
			self.data.shape,
			self.data.dtype,
			self.data.device,
		)

		self.sample = partial(sample_b, model= model, n_corr=self.n_corr)

		deltar = get_starting_deltar(self.sample, self.data, acc_target= 0.66)
		self.deltar = deltar.to(device= device, dtype= dtype)
		print('dataset:init_dataset deltar ', self.deltar)

		self.v_d = {'data': self.data, 'deltar': self.deltar}

		for equil_i in range(c.walkers.n_equil_step):
			self.v_d = self.sample(**self.v_d)
			if equil_i % 100 == 0:
				acc = torch.mean(self.v_d['acc'])
				deltar = torch.mean(self.v_d['deltar'])
				print('equil ', equil_i, ' acc ', acc.item(), ' deltar ', deltar.item())

		print('dataset:init_dataset sampler is pretraining ', self.mode == 'pre')
		print('dataset:init_dataset')
		[print(k, v.shape, v.device, v.dtype, v.mean(), v.std()) for k, v in self.v_d.items()]

	def __len__(self):
		return self.n_step

	def __getitem__(self, i):
		# ii.wait() # for dist
		self.v_d = self.sample(**self.v_d)
		return self.v_d
	

from .systems import System
from pyscf import gto, scf
# from pydantic.dataclasses import dataclass


class Scf:
	system: System
	
	def __init__(self, system: System):
		self.system = system
		self.mo_coef = None
		self.mol = None
		
	def init_app(self,):
		print('\npyfig:pyscf: ')

		self.mol: gto.Mole = gto.Mole(
			atom	= self.system.system_id, 
			basis	='sto-3g', 
			charge	= self.system.charge, 
			spin 	= self.system.spin, 
			unit	= 'bohr'
		)
		self.mol.build()
		mean_field_obj = scf.UHF(self.mol)
		mean_field_obj.kernel()
		self._hf = mean_field_obj

		# Molecular orbital (MO) coefficients 
		# matrix where rows are atomic orbitals (AO) and columns are MOs
		self.mo_coef = np.array(mean_field_obj.mo_coeff)
		print('app:init_app: mo_coef shape:', self.mo_coef.shape)
		mean_field_obj.analyze()
		print(mean_field_obj)

	def record_summary(self, summary: dict= None, opt_obj_all: list= None) -> None:
		import wandb
		summary = summary or {}

		atomic_id = "-".join([str(int(float(i))) for i in self.system.a_z.flatten()])
		spin_and_charge = f'{self.system.charge}_{self.system.spin}'
		geometric_hash = f'{self.system.a.mean():.0f}'
		exp_metaid = '_'.join([atomic_id, spin_and_charge, geometric_hash])

		if not opt_obj_all:
			print('no opt_obj_all, setting to 0.0')
			opt_obj_all = np.array([0.0, 0.0])

		columns = ["charge_spin_az0-az1-...pmu", "opt_obj", "Error (+/- std)"]
		data = [exp_metaid, np.array(opt_obj_all).mean(), np.array(opt_obj_all).std()]

		data += list(summary.values())
		columns += list(summary.keys())

		print('pyfig:app:record_summary:Result ')
		for i, j in zip(columns, data):
			print(i, j)
		print(summary)
		# print(pd.DataFrame.from_dict(summary | dict(zip(columns, data))).to_markdown())

		Result = wandb.Table(columns= columns)
		Result.add_data(*data)
		wandb.log(dict(Result= Result) | (summary or {}))

		return True
	
	def post_init_update(self):
		systems = {}
		system = systems.get(self.system.system_name, {})
		if self.system.system_name and not system:
			print('pyfig:app:post_init_update: system not found')
			return system
		
		def ang2bohr(tensor):
			return np.array(tensor) * 1.889725989
		
		unit = system.get('unit', None)
		if unit is None:
			print('pyfig:app:post_init_update: unit not specified, assuming bohr')
			unit = 'bohr'

		if system and 'a' in system and unit.lower() == 'angstrom': 
			system['a'] = ang2bohr(system['a'])
		return system




def loss_fn(
	model:torch.nn.Module,
	data: torch.Tensor,
	model_fn: Callable,
	system: dict,
	ansatzcfg: Ansatzcfg,
	**kw
):
	v_d = {}

	ke = compute_ke_b(model, model_fn, data, ke_method= ansatzcfg.ke_method)
	with torch.no_grad():
		pe = compute_pe_b(data, system['a'], system['a_z'])
		e = pe + ke
		e_mean_dist = torch.mean(torch.absolute(torch.median(e) - e))
		e_clip = torch.clip(e, min= e-5*e_mean_dist, max= e+5*e_mean_dist)
		energy_center = (e_clip - e_clip.mean())

		v_d |= dict(e= e, pe= pe, ke= ke)

	loss: Tensor = ((energy_center / system['a_z'].sum()) * model(data)).mean()

	return loss, dict(loss= loss.item(), **v_d)

def loss_fn_pretrain(
	model: Ansatz,
	data: torch.Tensor,
	c: Pyfig,
	step: int,
	**kw
):

	m_orb_ud = model.compute_hf_orb(data.detach())
	m_orb_ud = [torch.tensor(mo, dtype= data.dtype, device= data.device, requires_grad= False) for mo in m_orb_ud]
	orb_ud = model.compute_orb(data.detach())

	loss = sum(
		(torch.diagonal(o - mo, dim1=-1, dim2=-2) ** 2).mean()
		for o, mo in zip(orb_ud, m_orb_ud)
	)
	loss *= float(step / c.n_step)
	return loss, dict(loss= loss.item())


def kl_div_loss(
	model: Ansatz,
	data: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
	
	p = 2.*model(data)
	samples = (torch.randn_like(p) + 1.).clamp_min(1e-8)
	kl_div_loss = torch.nn.functional.kl_div(input= p, target= samples, reduction= 'batchmean')
	
	loss = kl_div_loss
	
	kl_d = dict(kl_div_loss= kl_div_loss.item(), kl_std= p.std().item(), kl_samples= samples.detach(), kl_p= p.detach())

	return loss, kl_d