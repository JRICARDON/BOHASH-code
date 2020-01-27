import cPickle as pickle
from base_learner import *
from BoHash import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from keras import backend as mTheano
import time
from optparse import OptionParser

def read_concepts(filename_labels):
    label_master_list=[]
    f=open(filename_labels)
    for line in f:
        label_list = line.strip().split()
        label_list = list(map(int,label_list)) 
        label_master_list.append(list(label_list))
    f.close()
    return list(label_master_list)

op = OptionParser()
op.add_option("-F", "--pickle_name", type="string", help="name of the pickle file with trained models")
op.add_option("-M", "--max_models", type=int, default=20, help="use at most M models for hashing")
op.add_option("-p", "--path", type="string", default='data/', help="path for data")
op.add_option("-r", "--reweighting", type="int", default=0, help="reweighting?") #0:resampling, 1:reweighting

(opts, args) = op.parse_args()
mpickle_name = opts.pickle_name

path = opts.path

print('CIFAR data from %s...'%path) 
t0 = time.time()

database_x = np.loadtxt(path+'CIFAR_GIST384_X_DATABASE.csv',delimiter=',')
queries_x = np.loadtxt(path+'CIFAR_GIST384_X_QUERIES.csv',delimiter=',')
#database_y and queries_y contain the list of concepts of each example (note that the first element of that list is not a concept but the length of the concept list) 
database_y = np.array(read_concepts(path+'CIFAR_GIST384_Y_DATABASE.csv'))
queries_y = np.array(read_concepts(path+'CIFAR_GIST384_Y_QUERIES.csv'))

scaler = StandardScaler()
database_x = scaler.fit_transform(database_x)
queries_x = scaler.transform(queries_x)
N = database_x.shape[0]
input_dim = database_x.shape[1]

print(database_x.shape)
print(queries_x.shape)

elapsed_0 = (time.time()-t0)/60.0
print("END READING ... AFTER %f MINS"%elapsed_0)

fi = open(mpickle_name, 'rb')

trained_functions = pickle.load(fi)
models = trained_functions['models']
m_weights = trained_functions['weights']
tr_time = trained_functions['time']
nbits = trained_functions['nbits']

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
        fo = open('BoHash_CIFAR_Database_%dBits_Model%d-REIMPL.txt'%(nbits,i), 'w')
    if boosting_mode == 1:
        fo = open('BoHash_CIFAR_Database_%dBits_Model%d-r1-REIMPL.txt'%(nbits,i), 'w')    
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
        fo = open('BoHash_CIFAR_Queries_%dBits_Model%d-REIMPL.txt'%(nbits,i), 'w') 
    if boosting_mode == 1:
        fo = open('BoHash_CIFAR_Queries_%dBits_Model%d-r1-REIMPL.txt'%(nbits,i), 'w') 
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
    
