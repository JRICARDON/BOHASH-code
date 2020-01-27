from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from siamese_net import *
import random
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from base_learner import *
from BoHash import *
import numpy.matlib
import time
from optparse import OptionParser

op = OptionParser()
op.add_option("-b", "--nbits", type=int, default=64, help="number of bits for hashing")
op.add_option("-M", "--nlearners", type=int, default=10, help="number of learners/hash functions")
op.add_option("-e", "--nepochs", type=int, default=10, help="number of epochs")
op.add_option("-p", "--path", type="string", default='data/', help="path for data")

(opts, args) = op.parse_args()
path = opts.path

def build_matrix_S(x_train_s,y_train_s,criterion=1,nsame_class = 50,nother_class = 100):

	n = x_train_s.shape[0]
	S = np.zeros((n,n))
	count_dis = 0
	count_sim = 0
	tr_pairs = []
	tr_labels = []

	if criterion == 1: #chinese criterion

		for i in range(0,n):#for each training example
		
			indices_same_class = [j for j in range(n) if y_train_s[i]==y_train_s[j]]
			indices_other_class = [j for j in range(n) if y_train_s[i]!=y_train_s[j]]
			indices_same_class = np.random.permutation(indices_same_class)
			indices_other_class = np.random.permutation(indices_other_class)
			indices_same_class = indices_same_class[1:nsame_class]
			indices_other_class  = indices_other_class[1:nother_class]

			for j in indices_same_class:
				S[i,j]=1
				S[j,i]=1
				count_sim +=1
				tr_pairs.append((i,j))
				tr_labels.append(1)

			for j in indices_other_class:
				S[i,j]=-1
				S[j,i]=-1
				count_dis +=1
				tr_pairs.append((i,j))
				tr_labels.append(-1)

	else: #old criterion (all pairs)

		for i in range(0,n):
			for j in range(0,n):
				if y_train_s[i] != y_train_s[j]:
					S[i,j]=-1
					S[j,i]=-1
					count_dis +=1
					tr_pairs.append((i,j))
					tr_labels.append(-1)

				else:
					S[i,j]=1
					S[j,i]=1
					count_sim +=1
					tr_pairs.append((i,j))
					tr_labels.append(1)


	print("WARNING. %d similar example pairs and %d dissimilar example pairs"%(count_sim,count_dis))
	print(S.shape)
	
	return S,tr_pairs,tr_labels

print('CIFAR data from %s...'%path) 

t_start_with_reading = time.time()

database_x = np.loadtxt(path+'CIFAR_GIST384_X_DATABASE.csv',delimiter=',')
queries_x = np.loadtxt(path+'CIFAR_GIST384_X_QUERIES.csv',delimiter=',')
database_y = np.loadtxt(path+'CIFAR_GIST384_Y_DATABASE.csv',delimiter=',')
queries_y = np.loadtxt(path+'CIFAR_GIST384_Y_QUERIES.csv',delimiter=',')
training_idx = np.loadtxt(path+'CIFAR_GIST384_TRAINING_INDICES_10000.csv',delimiter=',',dtype='int')

t_start_full_training = time.time()

print(database_x.shape)
print(database_y.shape)
print(queries_x.shape)
print(queries_y.shape)
print(training_idx)

x_train_s = database_x[training_idx]
y_train_s = database_y[training_idx]

print(x_train_s.shape)
print(y_train_s.shape)

N = database_x.shape[0]
n = x_train_s.shape[0]
input_dim = database_x.shape[1]

S,tr_pairs,tr_labels = build_matrix_S(x_train_s,y_train_s,criterion=1,nsame_class=50,nother_class=100)

scaler = StandardScaler()
database_x = scaler.fit_transform(database_x)
x_train_s = scaler.transform(x_train_s)

tr_epochs = opts.nepochs
nbits = opts.nbits
n_learners = opts.nlearners

t0 = time.time()

B=BoHasher(input_dim,nbits)
models = B.fit(n_learners,S,x_train_s,tr_pairs,tr_epochs)

elapsed= (time.time()-t0)/60.0#in minutes
elapsed_full = (time.time()-t_start_full_training)/60.0#in minutes
elapsed_with_reading = (time.time()-t_start_with_reading)/60.0#in minutes 

print(elapsed)
print(elapsed_full)
print(elapsed_with_reading)

import cPickle as pickle
import keras.losses
import siamese_net
keras.losses.hashing_loss = siamese_net.hashing_loss

m_weights = B.get_model_weights()
CIFAR_trained = {'models': models, 'weights': m_weights, 'time': elapsed, 'epochs': tr_epochs, 'nbits': nbits}
fo = open('CIFAR_trained_%dM_%db_%de-REIMPL.pickle'%(n_learners,nbits,tr_epochs), 'wb') 
pickle.dump(CIFAR_trained, fo)

fo_weights = open("BoHash_CIFAR_REIMPL_WEIGHTS.txt", 'wb')
print("Writting Model Weights ..")
for weight in m_weights:
	fo_weights.write("%.12f "%weight)
fo_weights.close()

name_time_file = "BoHash_Training_Time.txt"
fo_time = open(name_time_file, 'a+')
fo_time.write("%.4f, %.4f, %.4f\n"%(elapsed,elapsed_full,elapsed_with_reading))
fo_time.close()

