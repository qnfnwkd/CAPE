from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import geopy
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import utils
import sys
import models
import time
import bottleneck as bn

torch.manual_seed(1)

# ==============================================================
# Parameters

# # Data Path
TRAIN_DATA_CONTEXT_PATH = "./toyset/train_context.txt"
TRAIN_DATA_CONTENT_PATH = "./toyset/top20_tf_idf.txt"

# # Hyperparamters
CONTEXT_SIZE = 2
CONTENT_CONTEXT_SIZE = 2
EMBEDDING_DIM = 100
ALPHA = 0.01

# # Training Parameters
LEARNING_RATE = 0.001
EPOCHS = 20000
BATCH_SIZE = 1024
NUM_SAMPLED = 16


# =============================================================
# Data Load

print("Data Loading..")
train_context, train_content = utils.load_data(TRAIN_DATA_CONTEXT_PATH, TRAIN_DATA_CONTENT_PATH)

print("Data Indexing..")
poi2id, id2poi, word2id, id2word, train_context, train_content = utils.indexing(train_context, train_content)

print("Dataset Processing..")
train_context, train_content = utils.process_data(train_context, train_content, CONTEXT_SIZE, CONTENT_CONTEXT_SIZE)

print("Data Statistics:\nNumber of POIs: {}\nNumber of Words: {}".format(len(poi2id)-1, len(word2id)-1))

# =============================================================
# Training
# # Model Initialize
model = models.CAPE(EMBEDDING_DIM, len(poi2id), len(word2id)).cuda()
criterion = nn.LogSigmoid()

# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer_context = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
optimizer_content = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

"""
    Training...
"""
print("=======================================")
print("Training..\n")

best = [.0,.0,.0,.0,.0,.0,.0,.0,0]
for i in range(EPOCHS):

	# Check-in Context Layer
	batches = utils.train_context_batches(train_context, BATCH_SIZE)
	step = 0
	batch_num = int(len(train_context)/BATCH_SIZE) + 1
	context_loss = .0
	for batch in batches:
		target, context = batch
		input_target = Variable(torch.cuda.LongTensor(target))
		input_context = Variable(torch.cuda.LongTensor(context))

		# Optimizer Initialize
		optimizer_context.zero_grad()
		positive, negative = model(input_target, input_context, NUM_SAMPLED)
		_loss = -(criterion(positive) + criterion(negative).mean()).sum()
		_loss.backward()
		optimizer_context.step()
		context_loss += _loss.cpu().data.numpy()[0]

		# Printing Progress
		step+=1
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Context Layer Epoch: [{}/{}] Batch: [{}/{}]".format(i+1, EPOCHS, step, batch_num))

	batches = utils.train_content_batches(train_content, BATCH_SIZE)
	step = 0
	batch_num = int(len(train_content)/BATCH_SIZE) + 1
	content_loss = .0
	for batch in batches:
		poi, target, context = batch
		input_poi = Variable(torch.cuda.LongTensor(poi))
		input_target = Variable(torch.cuda.LongTensor(target))
		input_context = Variable(torch.cuda.LongTensor(context))

		# Optimizer Initialize
		optimizer_content.zero_grad()
		positive, negative = model.content(input_poi, input_target, input_context, NUM_SAMPLED)
		_loss = -1*ALPHA*(criterion(positive)+criterion(negative).mean()).sum()
		_loss.backward()
		optimizer_content.step()
		content_loss += _loss.cpu().data.numpy()[0]

		# Printing Progress
		step+=1
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Content Layer Epoch: [{}/{}] Batch: [{}/{}]".format(i+1, EPOCHS, step, batch_num))

	sys.stdout.write("\033[F")
	sys.stdout.write("\033[K")
	print("At Epoch: [{}/{}] [Context Loss / Content Loss]: [{}/{}]\n".format(i+1, EPOCHS, context_loss, content_loss))
