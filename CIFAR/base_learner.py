#Implementation of a base learner for the BoHasher
"""
@author: ricky

Observations: Finished on May 23, 17.
		      Tested with tester_baselearner.py, the following methods
		      - constructor ... OK
		      - fit ... OK
		      - hash ... OK, not checked if it gives reasonable results, but it runs
"""

from numpy.random import choice
from siamese_net import *
import numpy as np
import itertools


class BaseLeaner(object):

    def __init__(self, input_dim, nbits,nhidden=0):
        self.nbits = nbits #number of bits for hashing
        self.nhidden = nhidden
        self.net = Siamese_Net(input_dim,nbits,nhidden)
        self.trained =  False

	#train the network on a sample obtained by resampling example pairs according to D
    #mode=0 resampling, mode \neq 0 reweighting
    def fit(self, X, S, D,tr_pairs,nb_epoch=10,mode=0): 
        '''Fit a network according to a training set X, a label matrix S and a distribution D.
        '''
        sampled_tr_pairs = []
        sampled_tr_labels = []

        if mode == 0:
            training_pairs_idx = self.sample_pairs(tr_pairs,D)
        else:
            training_pairs_idx = tr_pairs

        for pair_index in training_pairs_idx:
            i1,i2 = pair_index
            sampled_tr_pairs += [[X[i1,:], X[i2,:]]]
            if mode == 0:
                sampled_tr_labels += [S[i1,i2]]
            else:#reweighting
                sampled_tr_labels += [S[i1,i2]*D[i1,i2]]
            
        sampled_tr_pairs = np.array(sampled_tr_pairs)
        sampled_tr_labels = np.array(sampled_tr_labels)

        print("SHAPE X, Y TRAIN")
        print(sampled_tr_pairs.shape)
        print(sampled_tr_labels.shape)
        print(type(sampled_tr_pairs[0,0,0]))

        self.net.fit(sampled_tr_pairs, sampled_tr_labels,nb_epoch=nb_epoch,batch_size=128)  
        self.trained = True
        del sampled_tr_pairs, sampled_tr_labels, training_pairs_idx

        X_hashed = self.net.hash(X)

        #return 0.5*np.dot(X_hashed,np.transpose(X_hashed))
        return (1.0/float(self.nbits))*np.dot(X_hashed,np.transpose(X_hashed))
        
	#D should be a symmetric matrix because the pair (i,j) should have the same weight that (j,i) 
    def sample_pairs(self,tr_pairs,D):
        '''Sample pairs according to D.
        '''
        n,n = D.shape
        weights = [D[pair] for pair in tr_pairs]
        print "SUM WEIGHTS IS:"
        print sum(weights)
        #all_pairs = list(itertools.combinations_with_replacement(range(n), 2))
        #all_weights = np.array([D[pair[0],pair[1]] for pair in all_pairs])
        selected_idx = choice(a=np.arange(len(tr_pairs)),size=(len(tr_pairs),1),p=weights,replace=True)
        selected_pairs = [tr_pairs[idx] for idx in selected_idx.flatten()]
        
        return selected_pairs

    def hash(self,X):
        if self.trained is False:
            print("**** WARNING. NETWORK HAS NOT BEEN TRAINED ****")
        return self.net.hash(X)

    def getW(self):
        return self.net.get_hashingW()

