import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


class CAPE(nn.Module):
	def __init__(self, embedding_dim, poi_num, vocab_size):
		super(CAPE, self).__init__()
		"""
			embedding_dim: Dimensionality of Embedding Vector
			poi_size: The Number of POIs in dataset
			vocab_size: The Number of Words in dataset
			user_size: The Number of Users in dataste
		"""

		self.embedding_dim = embedding_dim
		self.poi_num = poi_num
		self.vocab_size = vocab_size

		# Embedding Matrix		
		self.poi_embedding_in = nn.Embedding(self.poi_num, self.embedding_dim)
		self.poi_embedding_out = nn.Embedding(self.poi_num, self.embedding_dim)
		self.word_embedding_in = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.word_embedding_out = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.poi_embedding = nn.Embedding(self.poi_num, self.embedding_dim)

		# FF Layer
		self.ff = nn.Linear(2*self.embedding_dim, self.embedding_dim)
		self.init_weights()

	def init_weights(self):
		self.poi_embedding_in.weight.data.uniform_(-1.0, 1.0)
		self.poi_embedding_out.weight.data.uniform_(-1.0, 1.0)
		self.word_embedding_in.weight.data.uniform_(-1.0, 1.0)
		self.word_embedding_out.weight.data.uniform_(-1.0, 1.0)
		self.ff.weight.data.uniform_(-0.1, 0.1)
		self.ff.bias.data.fill_(0.0)
		self.poi_embedding.weight.requires_grad = False

	def get_embedding(self):
		self.poi_embedding.weight.data.copy_(self.poi_embedding_in.weight.data)

	def forward(self, target, context, num_sampled=None):
		'''
			For Check-in Context Layer
			target: Target POI
			context: Context POI
		'''

		# Embedding Lookup
		embedded_poi_in = self.poi_embedding_in(target)
		embedded_poi_out = self.poi_embedding_out(context)

		# =============================
		# Positive Loss
		# =============================
		target_loss = (embedded_poi_in*embedded_poi_out).sum(1).squeeze()


		# =============================
		# Negative Loss
		# =============================

		# Negative Sampling
		batch_size = target.size()[0]
		negative_samples = Variable(torch.cuda.FloatTensor(batch_size, num_sampled).
                         uniform_(0, self.poi_num-1).long())
		embedded_samples = self.poi_embedding_out(negative_samples).neg()

		negative_loss = torch.bmm(embedded_samples, embedded_poi_in.unsqueeze(2))

		return target_loss, negative_loss
	
	def content(self, poi, target, context, num_sampled):
		'''
			For Check-in Content Layer
			poi: Given POI
			target: Target Word
			context: Context Word
		'''

		# Embedding Lookup
		embedded_poi_in = self.poi_embedding_in(poi)
		embedded_word_target = self.word_embedding_in(target)
		embedded_word_context = self.word_embedding_out(context)

		# Concat
		embedded_context = torch.cat([embedded_poi_in, embedded_word_target],1)
		embedded_context = self.ff(embedded_context)

		# =============================
		# Positive Loss
		# =============================
		target_loss = (embedded_context*embedded_word_context).sum(1).squeeze()
		
		# =============================
		# Negative Loss
		# =============================
		# Negative Sampling
		batch_size = target.size()[0]
		negative_samples = Variable(torch.cuda.FloatTensor(batch_size, num_sampled).
                         uniform_(0, self.vocab_size-1).long())
		embedded_samples = self.word_embedding_out(negative_samples).neg()

		negative_loss = torch.bmm(embedded_samples, embedded_context.unsqueeze(2))

		return target_loss, negative_loss
