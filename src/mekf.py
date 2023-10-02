

import torch
from torch.optim import Optimizer

class NRLS_stable(Optimizer):
	"""Nonlinear Recursive Least Squares optimizer (experimental).
	"""

	def __init__(self, params, dim_out, p0=1e-2, eps=1e-2, lbd=0.98, alpha=None, lbd_decay=False,
	            lbd_max_step=1000000,lookahead_k=1,lookahead_alpha=1.0,momentum=0.0,momentum_p=0.0,
	             lr=1):

		if alpha is None:
			alpha = max(1/lbd-1,0)
		self._check_format(dim_out, p0, eps, lbd, alpha, lbd_decay,lbd_max_step,lookahead_k,lookahead_alpha,momentum)
		defaults = dict(p0=p0, eps=eps, lbd=lbd, alpha=alpha,
		                lbd_decay=lbd_decay,lbd_max_step=lbd_max_step,
		                lookahead_k=lookahead_k,lookahead_alpha=lookahead_alpha,
		                momentum=momentum,momentum_p=momentum_p,lr=lr)
		super(NRLS_stable, self).__init__(params, defaults)

		self.dim_out = dim_out
		with torch.no_grad():
			self._init_iekf_matrix()

	def _check_format(self, dim_out, p0, eps, lbd, alpha, lbd_decay,lbd_max_step,lookahead_k,lookahead_alpha,momentum):
		if not isinstance(dim_out, int) and dim_out>0:
			raise ValueError("Invalid output dimension: {}".format(dim_out))
		if not 0.0 < p0:
			raise ValueError("Invalid initial P value: {}".format(p0))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 < lbd:
			raise ValueError("Invalid lambda parameter {}".format(lbd))
		if not 0.0 <= alpha:
			raise ValueError("Invalid alpha parameter: {}".format(alpha))
		if not isinstance(lbd_decay, int) and not isinstance(lbd_decay, bool):
			raise ValueError("Invalid lambda decay flag: {}".format(lbd_decay))
		if not isinstance(lbd_max_step, int):
			raise ValueError("Invalid max step for lambda decaying: {}".format(lbd_max_step))
		if not isinstance(lookahead_k, int):
			raise ValueError("Invalid  k value for inner lookahead: {}".format(lookahead_k))
		if not 0.0 <= lookahead_alpha<=1.0:
			raise ValueError("Invalid alpha value for inner lookahead: {}".format(lookahead_alpha))
		if not 0.0 <= momentum <= 1.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))

	def _init_iekf_matrix(self):
		self.state['step']=0
		self.state['dim_out'] = self.dim_out
		self.state['iekf_groups']=[]
		for group in self.param_groups:
			iekf_mat=[]
			for p in group['params']:
				matrix = {}
				size = p.size()
				dim_w=1
				for dim in size:
					dim_w*=dim
				device= p.device
				matrix['P'] = group['p0']*torch.eye(dim_w,dtype=torch.float,device=device)
				matrix['EPS'] = group['eps']*torch.eye(dim_w,dtype=torch.float,device=device)
				matrix['R'] = group['lbd']*torch.eye(self.dim_out,dtype=torch.float,device=device)
				matrix['alpha'] =group['alpha']
				matrix['H'] = None
				matrix['dim_w']=dim_w
				matrix['device'] = device
				matrix['hist_P'] = matrix['P']
				matrix['lookahead_k'] = group['lookahead_k']
				matrix['lookahead_alpha'] = group['lookahead_alpha']
				iekf_mat.append(matrix)
			self.state['iekf_groups'].append(iekf_mat)

	def set_iekf_matrix(self,iekf_mats):
		for group_ind in range(len(self.param_groups)):
			group = self.param_groups[group_ind]
			new_iekf_mat = iekf_mats[group_ind]
			for ii in range(len(group['params'])):
				self.state['iekf_groups'][group_ind][ii].update(new_iekf_mat[ii])

	def get_H(self):
		H_groups=[]
		for iekf_mats in self.state['iekf_groups']:
			H_mats=[]
			for iekf_mat in iekf_mats:
				H_mats.append({'H':iekf_mat['H'].detach()})
			H_groups.append(H_mats)
		return H_groups

	def step(self,closure=None,skip_cal_grad=False,err=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		example:
			# y -> observed value, y_hat -> predicted value
			update_t=2
			coordinate_dim=3
			y = y[:update_t].contiguous().view((-1, 1))
			y_hat = y_hat[:update_t].contiguous().view((-1, 1))
			err = (y - y_hat).detach()
			def closure(index=0):
				optimizer.zero_grad()
				retain = index < (update_t * coordinate_dim - 1)
				y_hat[index].backward(retain_graph=retain)
				return err
		"""
		self.state['step'] += 1

		if not skip_cal_grad:
			for y_ind in range(self.state['dim_out']):
				err = closure(y_ind)
				for group_ind in range(len(self.param_groups)):
					group = self.param_groups[group_ind]
					iekf_mat = self.state['iekf_groups'][group_ind]
					for ii, w in enumerate(group['params']):
						if w.grad is None:
							continue
						H_n = iekf_mat[ii]['H']
						grad = w.grad.data.detach()
						if len(w.size())>1:
							grad = grad.transpose(1, 0)
						grad = grad.contiguous().view((1,-1))
						if y_ind ==0:
							H_n=grad
						else:
							H_n = torch.cat([H_n,grad],dim=0)
						self.state['iekf_groups'][group_ind][ii]['H'] = H_n

		err_T = err.transpose(0,1)

		for group_ind in range(len(self.param_groups)):
			group = self.param_groups[group_ind]
			iekf_mat = self.state['iekf_groups'][group_ind]
			momentum = group['momentum']
			momentum_p = group['momentum_p']
			for ii,w in enumerate(group['params']):
				if w.grad is None and not skip_cal_grad:
					continue

				lookahead_k = iekf_mat[ii]['lookahead_k']
				lookahead_alpha = iekf_mat[ii]['lookahead_alpha']
				P_n = iekf_mat[ii]['P']
				EPS = iekf_mat[ii]['EPS']
				H_n = iekf_mat[ii]['H']
				R_n = iekf_mat[ii]['R']
				alpha = iekf_mat[ii]['alpha']
				H_n_T = H_n.transpose(0, 1)
				if group['lbd_decay']:
					miu = 1.0 / min(self.state['step'],group['lbd_max_step'])
					R_n = R_n + miu * (err.mm(err_T) + H_n.mm(P_n).mm(H_n_T) - R_n)
					self.state['iekf_groups'][group_ind][ii]['R']= R_n

				g_n = R_n + H_n.mm(P_n).mm(H_n_T)
				g_n = g_n.inverse()

				K_n = P_n.mm(H_n_T).mm(g_n)
				delta_w = group['lr'] * K_n.mm(err)
				if len(w.size()) > 1:
					delta_w = delta_w.view((w.size(1),w.size(0))).transpose(1,0)
				else:
					delta_w = delta_w.view(w.size())

				new_P = (alpha + 1) * (P_n - K_n.mm(H_n).mm(P_n) + EPS)
				if momentum_p>0:
					new_P = (1-momentum_p)*new_P+momentum_p*P_n
				if lookahead_k>0:
					if self.state['step'] % lookahead_k==0:
						P_t_k = iekf_mat[ii]['hist_P']
						new_P = P_t_k + lookahead_alpha*(new_P-P_t_k)
						self.state['iekf_groups'][group_ind][ii]['hist_P'] = new_P

				if momentum>0:
					param_state = self.state[w]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(delta_w).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(delta_w.mul(1-momentum).detach())
					delta_w=buf

				self.state['iekf_groups'][group_ind][ii]['P'] =new_P
				w.data.add_(delta_w)

		return err

class MEKF(Optimizer):
	"""Nonlinear Recursive Least Squares optimizer (experimental).
	Modified Extended Kalman Filter
	"""

	def __init__(self, params, dim_out, p0=1e-2, eps=1e-2, lbd=0.98, alpha=None, lbd_decay=False,
	            lbd_max_step=1000000,lookahead_k=1,lookahead_alpha=1.0,momentum=0.0,momentum_p=0.0,
	             lr=1):

		if alpha is None:
			alpha = max(1/lbd-1,0)
		self._check_format(dim_out, p0, eps, lbd, alpha, lbd_decay,lbd_max_step,lookahead_k,lookahead_alpha,momentum)
		defaults = dict(p0=p0, eps=eps, lbd=lbd, alpha=alpha,
		                lbd_decay=lbd_decay,lbd_max_step=lbd_max_step,
		                lookahead_k=lookahead_k,lookahead_alpha=lookahead_alpha,
		                momentum=momentum,momentum_p=momentum_p,lr=lr)
		super(MEKF, self).__init__(params, defaults)

		self.dim_out = dim_out
		with torch.no_grad():
			self._init_iekf_matrix()

	def _check_format(self, dim_out, p0, eps, lbd, alpha, lbd_decay,lbd_max_step,lookahead_k,lookahead_alpha,momentum):
		if not isinstance(dim_out, int) and dim_out>0:
			raise ValueError("Invalid output dimension: {}".format(dim_out))
		if not 0.0 < p0:
			raise ValueError("Invalid initial P value: {}".format(p0))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 < lbd:
			raise ValueError("Invalid lambda parameter {}".format(lbd))
		if not 0.0 <= alpha:
			raise ValueError("Invalid alpha parameter: {}".format(alpha))
		if not isinstance(lbd_decay, int) and not isinstance(lbd_decay, bool):
			raise ValueError("Invalid lambda decay flag: {}".format(lbd_decay))
		if not isinstance(lbd_max_step, int):
			raise ValueError("Invalid max step for lambda decaying: {}".format(lbd_max_step))
		if not isinstance(lookahead_k, int):
			raise ValueError("Invalid  k value for inner lookahead: {}".format(lookahead_k))
		if not 0.0 <= lookahead_alpha<=1.0:
			raise ValueError("Invalid alpha value for inner lookahead: {}".format(lookahead_alpha))
		if not 0.0 <= momentum <= 1.0:
			raise ValueError("Invalid momentum value: {}".format(momentum))

	def _init_iekf_matrix(self):
		self.state['step']=0
		self.state['dim_out'] = self.dim_out
		self.state['iekf_groups']=[]
		for group in self.param_groups:
			iekf_mat=[]
			for p in group['params']:
				matrix = {}
				size = p.size()
				dim_w=1
				for dim in size:
					dim_w*=dim
				device= p.device
				matrix['P'] = group['p0']*torch.eye(dim_w,dtype=torch.float,device=device)
				matrix['EPS'] = group['eps']*torch.eye(dim_w,dtype=torch.float,device=device)
				matrix['R'] = group['lbd']*torch.eye(self.dim_out,dtype=torch.float,device=device)
				matrix['alpha'] =group['alpha']
				matrix['H'] = None
				matrix['dim_w']=dim_w
				matrix['device'] = device
				matrix['hist_P'] = matrix['P']
				matrix['lookahead_k'] = group['lookahead_k']
				matrix['lookahead_alpha'] = group['lookahead_alpha']
				iekf_mat.append(matrix)
			self.state['iekf_groups'].append(iekf_mat)

	def set_iekf_matrix(self,iekf_mats):
		for group_ind in range(len(self.param_groups)):
			group = self.param_groups[group_ind]
			new_iekf_mat = iekf_mats[group_ind]
			for ii in range(len(group['params'])):
				self.state['iekf_groups'][group_ind][ii].update(new_iekf_mat[ii])

	def get_H(self):
		H_groups=[]
		for iekf_mats in self.state['iekf_groups']:
			H_mats=[]
			for iekf_mat in iekf_mats:
				H_mats.append({'H':iekf_mat['H'].detach()})
			H_groups.append(H_mats)
		return H_groups

	def step(self,closure=None,skip_cal_grad=False,err=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		example:
			# y -> observed value, y_hat -> predicted value
			update_t=2
			coordinate_dim=3
			y = y[:update_t].contiguous().view((-1, 1))
			y_hat = y_hat[:update_t].contiguous().view((-1, 1))
			err = (y - y_hat).detach()
			def closure(index=0):
				optimizer.zero_grad()
				retain = index < (update_t * coordinate_dim - 1)
				y_hat[index].backward(retain_graph=retain)
				return err
		"""
		self.state['step'] += 1

		if not skip_cal_grad:
			for y_ind in range(self.state['dim_out']):
				err = closure(y_ind)
				for group_ind in range(len(self.param_groups)):
					group = self.param_groups[group_ind]
					iekf_mat = self.state['iekf_groups'][group_ind]
					for ii, w in enumerate(group['params']):
						if w.grad is None:
							continue
						H_n = iekf_mat[ii]['H']
						grad = w.grad.data.detach()
						if len(w.size())>1:
							grad = grad.transpose(1, 0)
						grad = grad.contiguous().view((1,-1))
						if y_ind ==0:
							H_n=grad
						else:
							H_n = torch.cat([H_n,grad],dim=0)
						self.state['iekf_groups'][group_ind][ii]['H'] = H_n

		err_T = err.transpose(0,1)

		for group_ind in range(len(self.param_groups)):
			group = self.param_groups[group_ind]
			iekf_mat = self.state['iekf_groups'][group_ind]
			momentum = group['momentum']
			momentum_p = group['momentum_p']
			for ii,w in enumerate(group['params']):
				if w.grad is None and not skip_cal_grad:
					continue

				lookahead_k = iekf_mat[ii]['lookahead_k']
				lookahead_alpha = iekf_mat[ii]['lookahead_alpha']
				P_n = iekf_mat[ii]['P']
				EPS = iekf_mat[ii]['EPS']
				H_n = iekf_mat[ii]['H']
				R_n = iekf_mat[ii]['R']
				alpha = iekf_mat[ii]['alpha']
				H_n_T = H_n.transpose(0, 1)
				if group['lbd_decay']:
					miu = 1.0 / min(self.state['step'],group['lbd_max_step'])
					R_n = R_n + miu * (err.mm(err_T) + H_n.mm(P_n).mm(H_n_T) - R_n)
					self.state['iekf_groups'][group_ind][ii]['R']= R_n

				g_n = R_n + H_n.mm(P_n).mm(H_n_T)
				g_n = g_n.inverse()

				K_n = P_n.mm(H_n_T).mm(g_n)
				delta_w = group['lr'] * K_n.mm(err)
				if len(w.size()) > 1:
					delta_w = delta_w.view((w.size(1),w.size(0))).transpose(1,0)
				else:
					delta_w = delta_w.view(w.size())

				if momentum_p>0:
					param_state = self.state[w]
					if 'momentum_buffer_p' not in param_state:
						P_ = P_n
					else:
						P_ = param_state['momentum_buffer_p']
					P_star = (alpha + 1) * (P_ - K_n.mm(H_n).mm(P_) + EPS)
					param_state['momentum_buffer_p'] = torch.clone(P_star).detach()
					new_P=momentum_p*P_n + (1-momentum_p)*P_star
				else:
					new_P = (alpha + 1) * (P_n - K_n.mm(H_n).mm(P_n) + EPS)

				if lookahead_k>0:
					if self.state['step'] % lookahead_k==0:
						P_t_k = iekf_mat[ii]['hist_P']
						new_P = P_t_k + lookahead_alpha*(new_P-P_t_k)
						self.state['iekf_groups'][group_ind][ii]['hist_P'] = new_P

				if momentum>0:
					param_state = self.state[w]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(delta_w).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(delta_w.mul(1-momentum).detach())
					delta_w=buf

				self.state['iekf_groups'][group_ind][ii]['P'] =new_P
				w.data.add_(delta_w)

		return err

