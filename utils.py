import random
import numpy as np
import geopy
from geopy.distance import vincenty
import sys


def load_data(train_data_context_path, train_data_content_path):

	"""
		Data format:
			train_context
				<poi_1> \t <poi_2> \t ...

			train_content
				<poi_1> \t <word_1> <word_2> ... <word_20>
	"""
	train_context = [line.strip() for line in open(train_data_context_path).readlines()]
	train_content = [line.strip() for line in open(train_data_content_path).readlines()]

	return train_context, train_content

def indexing(_train_context, _train_content):
	
	"""
		In this function, all the words, users, and POIs are indexed.
		And then, all the train, validation, and test data are converted by index.
	"""
	train_context = []
	train_content = []

	# POI Dictionary
	poi2id = {"<PAD>":0}
	id2poi = ["<PAD>"]

	for line in _train_context:
		tokens = line.split("\t")
		for poi in tokens:
			if poi not in poi2id:
				poi2id[poi] = len(poi2id)
				id2poi.append(poi)
		train_context.append([poi2id[poi] for poi in tokens])

	# Word Dictionary
	word2id = {"<PAD>":0}
	id2word = ["<PAD>"]
	
	for line in _train_content:
		poi, content = line.split("\t")
		words = content.split()
		for word in words:
			if word not in word2id:
				word2id[word] = len(word2id)
				id2word.append(word)
		if poi not in poi2id:
			poi2id[poi] = len(poi2id)
			id2poi.append(poi)

		train_content.append((poi2id[poi], [word2id[word] for word in words]))

	return poi2id, id2poi, word2id, id2word, train_context, train_content

def process_data(_train_context, _train_content, context_size, content_context_size):
	train_context = []

	for line in _train_context:
		pois = [0]*context_size + line + [0]*context_size
		for i in range(context_size, len(pois)-context_size):
			context = pois[i-context_size:i] + pois[i+1:i+context_size+1]
			for poi in context:
				if poi != 0:
					train_context.append((pois[i],poi))

	train_content = []
	for line in _train_content:
		poi, words = line
		for i in range(len(words)):
			for j in range(len(words)):
				if i != j:
					train_content.append((poi, words[i], words[j]))

	return train_context, train_content

def train_context_batches(train_context, batch_size):
	batch_num = int(len(train_context)/batch_size) + 1
	random.shuffle(train_context)
	for i in range(batch_num):
		left = i*batch_size
		right = min(len(train_context), (i+1)*batch_size)
		target, context = [], []
		for line in train_context[left:right]:
			target.append(line[0])
			context.append(line[1])
		
		yield target, context

def train_content_batches(train_content, batch_size):
	batch_num = int(len(train_content)/batch_size) + 1
	random.shuffle(train_content)
	for i in range(batch_num):
		left = i*batch_size
		right = min(len(train_content), (i+1)*batch_size)
		target_poi, target_word, context_word = [], [], []
		for line in train_content[left:right]:
			target_poi.append(line[0])
			target_word.append(line[1])
			context_word.append(line[2])
		
		yield target_poi, target_word, context_word