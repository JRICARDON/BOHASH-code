from __future__ import absolute_import
#from __future__ import print_function
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
#recuerda dejar data/ en el path
op.add_option("-p", "--path", type="string", default='', help="path for data")
op.add_option("-U", "--max_models", type=int, default=20, help="use at most U models for hashing")
op.add_option("-r", "--reweighting", type="int", default=0, help="reweighting?") #0:resampling, 1:reweighting

(opts, args) = op.parse_args()
path = opts.path

def read_concepts(filename_labels):
    label_master_list=[]
    f=open(filename_labels)
    for line in f:
        label_list = line.strip().split()
        label_list = list(map(int,label_list)) 
        label_master_list.append(list(label_list))
    f.close()
    return list(label_master_list)


def build_matrix_S(x_train_s,y_train_s,criterion=1,nsame_class = 50,nother_class = 100):

	n = x_train_s.shape[0]
	S = np.zeros((n,n))
	count_dis = 0
	count_sim = 0
	tr_pairs = []
	tr_labels = []

	if criterion == 1: #chinese criterion

		for i in range(0,n):#for each training example
		
			indices_same_class = [j for j in range(n) if len(set(y_train_s[i][1:])& set(y_train_s[j][1:]))>=1]
			indices_other_class = [j for j in range(n) if len(set(y_train_s[i][1:])& set(y_train_s[j][1:]))==0]
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


t_start_with_reading = time.time()

print('NUSWIDE data from %s...'%path) 

database_x = np.loadtxt(path+'NUSWIDE_X_DATABASE.csv',delimiter=',',dtype=np.float32)
queries_x = np.loadtxt(path+'NUSWIDE_X_QUERIES.csv',delimiter=',',dtype=np.float32)
#database_y and queries_y contain the list of concepts of each example (note that the first element of that list is not a concept but the length of the concept list) 
database_y = np.array(read_concepts(path+'NUSWIDE_Y_DATABASE.csv'))
queries_y = read_concepts(path+'NUSWIDE_Y_QUERIES.csv')
training_idx = np.loadtxt(path+'NUSWIDE_TRAINING_INDICES_10000.csv',delimiter=',',dtype='int')

print(database_x.shape)
print(queries_x.shape)
print(training_idx)

x_train_s = database_x[training_idx]
y_train_s = database_y[training_idx]

print(x_train_s.shape)

N = database_x.shape[0]
n = x_train_s.shape[0]
input_dim = database_x.shape[1]

t_start_full_training = time.time()

S,tr_pairs,tr_labels = build_matrix_S(x_train_s,y_train_s,criterion=1,nsame_class=50,nother_class=100)

scaler = StandardScaler()
database_x = scaler.fit_transform(database_x)
x_train_s = scaler.transform(x_train_s)

#del database_x
#gc.collect()

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

m_weights = B.get_model_weights()

fo_weights = open("BoHash_NUSWIDE_WEIGHTS.txt", 'wb')
print("Writting Model Weights ..")
for weight in m_weights:
	fo_weights.write("%.12f "%weight)
fo_weights.close()

name_time_file = "BoHash_NUSWIDE_Training_Time_NMaq-%d-E%d.txt"%(n_learners,tr_epochs)

fo_time = open(name_time_file, 'a+')
fo_time.write("%.4f, %.4f, %.4f\n"%(elapsed,elapsed_full,elapsed_with_reading))
fo_time.close()

#### HASHING
print("HASHING ...")

t0 = time.time()

queries_x = scaler.transform(queries_x)
N = database_x.shape[0]
input_dim = database_x.shape[1]

print(database_x.shape)
print(queries_x.shape)

elapsed_0 = (time.time()-t0)/60.0
print("END READING ... AFTER %f MINS"%elapsed_0)

tr_time = elapsed

ensemble = BoHasher(input_dim,nbits)
ensemble.models = models
ensemble.model_weights = m_weights
ensemble.M = len(models)
nmodels = ensemble.M
maxmodels = nmodels

print("@@@@@@@@ HASHING with %d models ..."%ensemble.M)

n_queries = len(queries_x)
n_database = len(database_x)

hash_database = np.zeros((n_database,nbits))
hash_queries = np.zeros((n_queries,nbits))

t0 = time.time()
nblocks = nbits/8
npadding = 0
print("@@@@@@@@ NBITS: %d bits"%nbits)
if nbits % 8 != 0: 
    nblocks = nblocks + 1
    npadding = nblocks*8 - nbits
    print("@@@@@@@@ Padding will be: %d bits"%npadding)

boosting_mode = opts.reweighting #0:resampling, 1:reweighting

for i in range(maxmodels):
    model = models[i]
    hash_queries = model.hash(queries_x)
    hash_database = model.hash(database_x)
    if boosting_mode == 0:
        fo = open('BoHash_NUSWIDE_Database_%dBits_Model%d.txt'%(nbits,i), 'w')
    if boosting_mode == 1:
        fo = open('BoHash_NUSWIDE_Database_%dBits_Model%d-r1.txt'%(nbits,i), 'w')
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
    if boosting_mode == 0:
        fo = open('BoHash_NUSWIDE_Queries_%dBits_Model%d.txt'%(nbits,i), 'w')
    if boosting_mode == 1:
        fo = open('BoHash_NUSWIDE_Queries_%dBits_Model%d-r1.txt'%(nbits,i), 'w')   
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


