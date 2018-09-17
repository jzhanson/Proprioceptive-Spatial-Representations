import numpy as np
from scipy import sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GCN']


def normalize_adj(A, is_sym=True, exponent=0.5):
	"""
		Normalize adjacency matrix
		is_sym=True: D^{-1/2} A D^{-1/2}
		is_sym=False: D^{-1} A
	"""
	rowsum = np.array(A.sum(1))

	if is_sym:
		r_inv = np.power(rowsum, -exponent).flatten()
	else:
		r_inv = np.power(rowsum, -1.0).flatten()

	r_inv[np.isinf(r_inv)] = 0.

	if sp.isspmatrix(A):
		r_mat_inv = sp.diags(r_inv.squeeze())
	else:
		r_mat_inv = np.diag(r_inv)

	if is_sym:
		return r_mat_inv.dot(A).dot(r_mat_inv)
	else:
		return r_mat_inv.dot(A)


def get_laplacian(adj):
	"""
		Compute Graph Laplacian
		Args:
			adj: shape N X N, adjacency matrix, could be numpy or scipy sparse array
				use symmetric GCN renormalization trick, L = D^{-1/2} ( I + A ) D^{-1/2}

		Returns:
			L: shape N X N, graph Laplacian matrix
	"""

	assert len(adj.shape) == 2 and adj.shape[0] == adj.shape[1]

	if sp.isspmatrix(adj):
		identity_mat = sp.eye(adj.shape[0])
	else:
		identity_mat = np.eye(adj.shape[0])

	return normalize_adj(identity_mat + adj, is_sym=True)


class GCN(nn.Module):
	""" Improved version of Graph Convolutional Networks by Renjie Liao 
		N.B.: the readout function is currently designed to do 
			graph-level regression
	"""

	def __init__(self, config):
		super(GCN, self).__init__()
		self.config = config
		self.input_dim = config.input_dim
		self.hidden_dim = config.hidden_dim
		self.output_dim = config.output_dim
		self.num_layer = config.num_layer
		self.num_edgetype = config.num_edgetype
		self.dropout = config.dropout if hasattr(config, 'dropout') else 0.0

		dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
		self.filter = nn.ModuleList([
				nn.Linear(dim_list[tt] * (self.num_edgetype), dim_list[tt + 1])
				for tt in range(self.num_layer)
		] + [nn.Linear(dim_list[-2], dim_list[-1])])

		# attention
		self.att_func = nn.Sequential(* [nn.Linear(dim_list[-2], 1), nn.Sigmoid()])

		if config.loss == 'CrossEntropy':
			self.loss_func = torch.nn.CrossEntropyLoss()
		elif config.loss == 'MSE':
			self.loss_func = torch.nn.MSELoss()
		elif config.loss == 'L1':
			self.loss_func = torch.nn.L1Loss()
		else:
			raise ValueError("Non-supported loss function!")

		self._init_param()

	def _init_param(self):
		for ff in self.filter:
			if isinstance(ff, nn.Linear):
				nn.init.xavier_uniform_(ff.weight.data)
				if ff.bias is not None:
					ff.bias.data.zero_()

		for ff in self.att_func:
			if isinstance(ff, nn.Linear):
				nn.init.xavier_uniform_(ff.weight.data)
				if ff.bias is not None:
					ff.bias.data.zero_()

	def forward(self, node_feat, L, label=None, mask=None):
		"""
			N.B.:
				If you have multiple small graphs, I recommend to use dense L 
				otherwise you can use sparse

			shape parameters:
				batch size of graphs = B
				maximum # nodes per batch = N
				feature dimension = D
				hidden dim = H
				num edge types = C
				output dimension = P

			Args:
				node_feat: float tensor, shape B X N X D
				L: float tensor, shape B X N X N X C
				label: float tensor, shape B X P
				mask: float tensor, shape B X N, indicates which node is valid

			Returns:
				score: float tensor, shape B X P
		"""
		batch_size = node_feat.shape[0]
		num_node = node_feat.shape[1]
		input_state = node_feat

		# propagation
		state = input_state
		for tt in range(self.num_layer):
			dim_state = state.shape[2]
			msg = []

			for ii in range(self.num_edgetype):
				msg += [torch.bmm(L[:, :, :, ii], state)]  # shape: B X N X D

			msg = torch.cat(msg, dim=2).view(-1, (self.num_edgetype) * dim_state)
			state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
			state = F.dropout(state, self.dropout, training=self.training)

		# output
		state = state.view(batch_size * num_node, -1)
		y = self.filter[-1](state)  # shape: BN X 1
		att_weight = self.att_func(state)  # shape: BN X 1    
		y = (att_weight * y).view(batch_size, num_node, -1)

		score = []
		if mask is not None:
			for bb in range(batch_size):
				score += [torch.mean(y[bb, mask[bb], :], dim=0)]
		else:
			for bb in range(batch_size):
				score += [torch.mean(y[bb, :, :], dim=0)]

		score = torch.stack(score)

		if label is not None:
			return score, self.loss_func(score, label)
		else:
			return score


if __name__ == '__main__':
	from easydict import EasyDict as edict
	config = {
			'input_dim': 2,
			'hidden_dim': [16, 16],  # len should be same as num_layer
			'output_dim': 1,
			'num_layer': 2,
			'num_edgetype': 1,
			'loss': 'MSE'
	}
	config = edict(config)

	# (1) You can increase # edgetype by making adjacency matrix a 3-dim tensor
	#     and the 3rd dimension corresponding to edgetype
	# (2) L is the graph Laplacian matrix
	num_graphs = 2
	num_nodes = 4
	dim_feat = 2
	A = [
			torch.Tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
			torch.Tensor([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
	]
	L = []
	for adj in A:
		L += [torch.from_numpy(get_laplacian(adj)).float().unsqueeze(dim=2)]
	L = torch.stack(L, dim=0)
	node_feat = torch.randn(num_graphs, num_nodes, dim_feat)

	model = GCN(config)
	score = model.forward(node_feat, L)
