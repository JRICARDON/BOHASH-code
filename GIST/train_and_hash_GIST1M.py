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
import gc

op = OptionParser()
op.add_option("-b", "--nbits", type=int, default=64, help="number of bits for hashing")
op.add_option("-M", "--nlearners", type=int, default=10, help="number of learners/hash functions")
op.add_option("-e", "--nepochs", type=int, default=10, help="number of epochs")
op.add_option("-p", "--path", type="string", default='data/', help="path for data")
op.add_option("-U", "--max_models", type=int, default=20, help="use at most U models for hashing")
op.add_option("-r", "--reweighting", type="int", default=0, help="reweighting?") #0:resampling, 1:reweighting


(opts, args) = op.parse_args()
path = opts.path

def build_matrix_S(tr_neighbors,criterion=1,n_similar=50,n_dissimilar=100):

	n = tr_neighbors.shape[0]
	S = np.zeros((n,n))
	count_dis = 0
	count_sim = 0
	tr_pairs = []
	tr_labels = []

	print("SHAPE S is %dx%d"%(n,n))
	for i in range(0,n):#for each training example
	
		print("Training item %d, found %d neighbors..."%(i,len(tr_neighbors[i,:])))
		indices_similar = tr_neighbors[i,:n_similar]
		indices_dissimilar = tr_neighbors[i,n_similar:]

		for j in indices_similar:
			S[i,j]=1
			S[j,i]=1
			count_sim +=1
			tr_pairs.append((i,j))
			tr_labels.append(1)

		for j in indices_dissimilar:
			S[i,j]=-1
			S[j,i]=-1
			count_dis +=1
			tr_pairs.append((i,j))
			tr_labels.append(-1)

	print("WARNING. %d similar example pairs and %d dissimilar example pairs"%(count_sim,count_dis))
	print(S.shape)
	
	return S,tr_pairs,tr_labels

print('GIST1M data from %s...'%path) 


def ivecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

t_start_with_reading = time.time()

tr_file = path+'gist_learn.fvecs'
database_file = path+'gist_base.fvecs'

database = fvecs_read(database_file)
print("DATABASE LOADED ... SHAPE MATRIX:")
print(database.shape)

scaler = StandardScaler()
database = scaler.fit_transform(database)
N = database.shape[0]

#del database
#gc.collect()

print("DATABASE CLEARED ...")

original_tr_data = fvecs_read(tr_file)
print("SHAPE ORIGINAL TRAINING DATA MATRIX:")
print(original_tr_data.shape)
tr_indices = np.loadtxt(path+'GIST1M-Train-Indices.txt',delimiter=',',dtype='int')
print("SELECTING %d ..."%len(tr_indices))
training_x = original_tr_data[tr_indices,:]
print("SHAPE FINAL TRAINING DATA MATRIX:")
print(training_x.shape)

training_x = scaler.transform(training_x)

tr_neighbors = np.loadtxt(path+'GIST1M-Train-vs-Train-Neighbors.txt',delimiter=',',dtype='int')

print(tr_neighbors.shape)

t_start_full_training = time.time()

n = training_x.shape[0]
input_dim = training_x.shape[1]

S,tr_pairs,tr_labels = build_matrix_S(tr_neighbors,criterion=1,n_similar=50,n_dissimilar=100)

tr_epochs = opts.nepochs
nbits = opts.nbits
n_learners = opts.nlearners

t0 = time.time()

B=BoHasher(input_dim,nbits)
models = B.fit(n_learners,S,training_x,tr_pairs,tr_epochs)

elapsed= (time.time()-t0)/60.0#in minutes
elapsed_full = (time.time()-t_start_full_training)/60.0#in minutes
elapsed_with_reading = (time.time()-t_start_with_reading)/60.0#in minutes 

print(elapsed)
print(elapsed_full)
print(elapsed_with_reading)

m_weights = B.get_model_weights()

fo_weights = open("BoHash_GIST1M_WEIGHTS.txt", 'wb')
print("Writting Model Weights ..")
for weight in m_weights:
	fo_weights.write("%.12f "%weight)
fo_weights.close()

name_time_file = "BoHash_GIST1M_Training_Time_NMaq-%d-E%d.txt"%(n_learners,tr_epochs)

fo_time = open(name_time_file, 'a+')
fo_time.write("%.4f, %.4f, %.4f\n"%(elapsed,elapsed_full,elapsed_with_reading))
fo_time.close()

#### HASHING
print("HASHING ...")

t0 = time.time()

query_file = path+'gist_query.fvecs'
database_file = path+'gist_base.fvecs'

database_x = database

queries_x = np.loadtxt(path+'GIST1M_X_QUERY.csv',delimiter=',')
queries_groundtruth = np.loadtxt(path+'GIST1M-Query-Groundtruth.txt',delimiter=',',dtype='int')

queries_x = scaler.transform(queries_x)

N = database_x.shape[0]
input_dim = database_x.shape[1]

print(database_x.shape)
print(queries_x.shape)
print(queries_groundtruth.shape)

elapsed_0 = (time.time()-t0)/60.0
print("END READING ... AFTER %f MINS"%elapsed_0)

tr_time = elapsed

ensemble = BoHasher(input_dim,nbits)
ensemble.models = models
ensemble.model_weights = m_weights
ensemble.M = len(models)
nmodels = ensemble.M
maxmodels = nmodels
maxmodels = min(maxmodels,nmodels)

print("@@@@@@@@ HASHING with %d models ..."%ensemble.M)

n_queries = len(queries_x)
n_database = len(database_x)
g_queries = np.zeros(n_database) #QxN matrix
hamm_distance = np.zeros(n_database) #QxN matrix

hash_database = np.zeros((n_database,nbits))
hash_queries = np.zeros((n_queries,nbits))

t0 = time.time()
nblocks = nbits/8
npadding = 0

if nbits % 8 != 0:
	npadding = nbits - nblocks*8   
	nblocks = nblocks + 1
	print("@@@@@@@@ Padding will be: %d bits"%npadding)

for i in range(maxmodels):
	model = models[i]
	hash_queries = model.hash(queries_x)
	hash_database = model.hash(database_x)
	fo = open('BoHash_GIST1M_Database_%dBits_Model%d.txt'%(nbits,i), 'w')
	nbits_with_padding = nbits+npadding
	fo.write("%d %d\n"%(n_database,nbits_with_padding)) 
	for k in range(n_database):
		for l in range(nbits):
			if hash_database[k,l]>0:
				fo.write("1 ")
			else:
				fo.write("0 ")
		if npadding>0:
			for p in range(npadding):
				fo.write("0 ")
		fo.write("\n")
	fo.close()

	fo = open('BoHash_GIST1M_Queries_%dBits_Model%d.txt'%(nbits,i), 'w') 
	fo.write("%d %d\n"%(n_queries,nbits_with_padding)) 
	for k in range(n_queries):
		for l in range(nbits):
			if hash_queries[k,l]>0:
				fo.write("1 ")
			else:
				fo.write("0 ")
		if npadding>0:
			for p in range(npadding):
				fo.write("0 ")
		fo.write("\n")
	fo.close()
	
elapsed_hashing = (time.time()-t0)/60.0
print("ELAPSED HASHING:  %f"%elapsed_hashing )

