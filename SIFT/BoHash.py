# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:39:51 2017

@author: carlosvalle
"""
from __future__ import division
from __future__ import absolute_import
import numpy as np
from base_learner import *
import matplotlib.pyplot as plt
from keras import backend as mTheano
import gc

class BoHasher:

    def __init__(self,input_dim,nbits,nhidden=0):
           
        self.input_dim = input_dim
        self.nbits = nbits
        self.models = []
        self.model_weights = []
        self.nhidden = nhidden
        self.M = 0

    def fit(self,ens_size,S,X_train,tr_pairs,nb_epoch=10,boosting_mode=0):
        '''Train a set of ens_size hash functions using the Bohash approach
        '''
        n = len(X_train)
        weight_uniform = 1.0/float(len(tr_pairs))
        D = np.zeros((n,n))
        for pair in tr_pairs:
            D[pair[0],pair[1]] = weight_uniform
            D[pair[1],pair[0]] = weight_uniform

        weights = [D[pair] for pair in tr_pairs]
        print "INITIALIZATION"
        print np.sum(np.sum(np.triu(D, k=0)))
        print np.sum(weights)
        
        Gcons = np.zeros((n,n))
        
        #LAST MODIFICATION 16ABRIL18: WITH WEIGHTS

        for i in range(ens_size):
            base_learner = BaseLeaner(self.input_dim,self.nbits,self.nhidden)
            g = base_learner.fit(X_train,S,D,tr_pairs,nb_epoch=nb_epoch,mode=boosting_mode)
            
            #EHamm = EHamm + hamm
            edge = sum(sum(np.triu(D*S*g, k=0).astype('float32')))
            print(edge)
            
            beta=0.5*np.log((1.0+edge)/(1.0-edge))  

            Gcons = Gcons + beta*g

            D = np.exp(-1.0*S*Gcons)
            D = self.normalize_weights(D,tr_pairs)
            
            print "WEIGHT NORMALIZATION"
            print np.sum(np.sum(np.triu(D, k=0)))
                  
            self.models.append(base_learner)
            self.model_weights.append(beta)

            print(beta)
            print(self.model_weights) 

            # name = "WEIGHTS-%d.txt"%i
            # np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
            # with open(name, 'w') as f:
            #     f.write(np.array2string(W, separator=', '))
            
        self.M = ens_size
        return self.models

    def getM(self):
        return self.M

    def hash(X_matrix, model_index):
        if model_index > self.M-1:
            raise Exception('Invalid index. There are only: %d models'%self.M) 

        if model_index < 0:
            raise Exception('Invalid index.') 

        desired_model = self.models[model_index]
        return desired_model.hash(X_matrix)
        
    def get_model_weights(self):
        return self.model_weights

    def normalize_weights(self,D,tr_pairs):
        '''Normalize W by dividing the upper matrix and its diagonal values by the sum of these elements. 
        Next, lower matrix values are added in order to make the matrix W symmetric
        '''
        sum_weights = np.sum([D[tr_pair] for tr_pair in tr_pairs])
        D=D/float(sum_weights)
        return D

    #X_database is a Nxd matrix, with N the number of database items and d dimensionality
    #X_queries is a Qxd matrix, with q the number of queries and d dimensionality
    def predict_full(self,X_queries,X_database,maxmodels=-1,use_weights=False,K=10):

        G_queries = np.zeros((len(X_queries),len(X_database))) #QxN matrix
        nmodels = len(self.models)
        
        if maxmodels < 0:
            maxmodels = nmodels

        maxmodels = min(maxmodels,nmodels)
        total_ranking = np.zeros((maxmodels,len(X_queries),K))#MxQxK array

        for i in range(maxmodels):
            model = self.models[i]
            hashX = mTheano.variable(model.hash(X_database))
            hashQ = mTheano.variable(model.hash(X_queries))
            g_queries = mTheano.dot(hashQ, mTheano.transpose(hashX)).eval() #QxN matrix
            if use_weights is not False:
                g_queries = self.model_weights[i]*g_queries #check
            G_queries = G_queries + g_queries

        hamm_distance = 0.5*(self.nbits - G_queries)#qxm matrix, element hamm_distance[i,j] is the distance between query i and candidate j
        hamm_distance = mTheano.variable(hamm_distance)#qxm matrix
        #ranking = np.argsort(hamm_distance,axis=1)
        ranking = mTheano.T.argsort(hamm_distance,axis=1).eval()
        kNN_indices = ranking[:,0:K]

        return kNN_indices

    #X_database is a Nxd matrix, with N the number of database items and d dimensionality
    #X_queries is a Qxd matrix, with q the number of queries and d dimensionality

    def get_candidates_knn_query(self,X_queries,X_database,maxmodels=-1,use_weights=False,K=10,return_G=True):

        G_queries = None
        if return_G is not False:
            G_queries = np.zeros((len(X_queries),len(X_database))) #QxN matrix
        
        nmodels = len(self.models)
        
        if maxmodels < 0:
            maxmodels = nmodels

        maxmodels = min(maxmodels,nmodels)
        total_ranking = np.zeros((maxmodels,len(X_queries),K),dtype=np.int32)#MxQxK array

        for i in range(maxmodels):
            model = self.models[i]
            hashX = mTheano.variable(model.hash(X_database))
            hashQ = mTheano.variable(model.hash(X_queries))
            g_queries = mTheano.dot(hashQ, mTheano.transpose(hashX)).eval() #QxN matrix
            hamm_distance = 0.5*(self.nbits - g_queries) #QxN matrix
            hamm_distance = mTheano.variable(hamm_distance)
            ranking = mTheano.T.argsort(hamm_distance,axis=1).eval() #QxN matrix
            kNN_indices = ranking[:,:K] #QxK matrix
            total_ranking[i] = kNN_indices #QxK matrix
            if return_G is not False:
                if use_weights is not False:
                    g_queries = self.model_weights[i]*g_queries #check
                G_queries = G_queries + g_queries

            del hashX
            del hashQ
            del g_queries
            del hamm_distance
            gc.collect()

        #transpose changes shape from MxQxK to QxMxK
        return total_ranking.transpose(1,0,2).tolist(), G_queries

    def get_candidates_range_query(self,X_queries,X_database,Y_queries,Y_database,maxmodels=-1,use_weights=False,rad_hamm=0.0,return_G=True):

        n_queries = len(X_queries)
        n_database = len(X_database)

        print("QUERY SIZE=%d, DATABASE SIZE=%d"%(n_queries,n_database))
        print("RANGE QUERY WITH RADIUS = %f"%rad_hamm)
        best_hamm_distance = 10*self.nbits*np.ones((n_queries,n_database))

        G_queries = None
        if return_G is not False:
            G_queries = np.zeros((len(X_queries),len(X_database))) #QxN matrix
        
        nmodels = len(self.models)
        
        if maxmodels < 0:
            maxmodels = nmodels

        maxmodels = min(maxmodels,nmodels)
        #retrieved_mask = np.zeros((n_queries,n_database))
        #total_ranking = []
        #index_threshold = np.zeros((n_queries,maxmodels))

        for i in range(maxmodels):
            model = self.models[i]
            hashX = mTheano.variable(model.hash(X_database))
            hashQ = mTheano.variable(model.hash(X_queries))
            g_queries = mTheano.dot(hashQ, mTheano.transpose(hashX)).eval() #QxN matrix
            hamm_distance = 0.5*(self.nbits - g_queries) #QxN matrix
            #count_collisions = (hamm_distance <= rad_hamm).mean()
            #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            #print "Querying Table %d: found %f%% collisions with radius %f"%(i,100*count_collisions,rad_hamm)
            hamm_distance = mTheano.variable(hamm_distance)
            b_hamm = mTheano.variable(best_hamm_distance)
            best_hamm_distance = mTheano.minimum(hamm_distance,b_hamm).eval()

            if return_G is not False:
                if use_weights is not False:
                    g_queries = self.model_weights[i]*g_queries #check
                G_queries = G_queries + g_queries

            #print(limits.shape)
            #index_threshold[:,i] = limits
            del hashX
            del hashQ
            del hamm_distance
            del g_queries
            gc.collect()

        
        #transpose changes shape from MxQxK to QxMxK
        #return retrieved_mask,best_hamm_distance
        return best_hamm_distance, G_queries


    #X_queries is a qxd matrix, with q the number of queries and d dimensionality
    #X_shortlist is a mxd matrix, with m the number of candidates and d dimensionality
    #hamm_distance = 0.5*(nbits - G)
    def get_kNN_G(self,X_queries,X_shortlist,maxmodels=-1,use_weights=False,K=10):

        G_queries = np.zeros((len(X_queries),len(X_shortlist))) #qxm matrix
        nmodels = len(self.models)
        
        if maxmodels < 0:
            maxmodels = nmodels

        maxmodels = min(maxmodels,nmodels)

        for i in range(maxmodels):
            model = self.models[i]
            hashX = mTheano.variable(model.hash(X_shortlist))
            hashQ = mTheano.variable(model.hash(X_queries))
            g_queries = mTheano.dot(hashQ, mTheano.transpose(hashX)).eval()
            if use_weights is not False:
                g_queries = self.model_weights[i]*g_queries #check
            G_queries = G_queries + g_queries

        hamm_distance = 0.5*(self.nbits - G_queries)#qxm matrix, element hamm_distance[i,j] is the distance between query i and candidate j
        hamm_distance = mTheano.variable(hamm_distance)#qxm matrix
        #ranking = np.argsort(hamm_distance,axis=1)
        ranking = mTheano.T.argsort(hamm_distance,axis=1).eval()
        kNN_indices = ranking[:,0:K]

        del hashX
        del hashQ
        del g_queries
        gc.collect()

        return kNN_indices

    def predict(self,X_queries,X_database,Y_queries,Y_database,maxmodels=-1,use_weights=False,opts=None):
        '''Aggregate the individual K-NN list into a single one.
        Mode 1: majority vote.
        Mode 2: consensus
        '''

        K=opts.nn
        R=opts.radius
        global_mode=opts.aggregation_mode
        local_mode=opts.local_mode

        if self.M <= 0:
            raise Exception('Empty ensemble.') 
        
        K_one_table = int(np.sqrt(len(X_database)))
        R_one_table = 0

        neighbours_indices = []

        if maxmodels < 0:
            maxmodels = self.M

        maxmodels = min(maxmodels,self.M)

        if global_mode == 1:# majority voting
            print("Local K is %d"%K_one_table)
            query_neighbours_list, _ = self.get_candidates_knn_query(X_queries,X_database,maxmodels=maxmodels,use_weights=use_weights,K=K_one_table,return_G=False)

            for i in range(len(query_neighbours_list)):
                query_result = query_neighbours_list[i]
                dict_str = {} #dict contains the index of each object selected for at least one hash function and its respectively weight
                for j in range(maxmodels):
                    single_neighbours = query_result[j]
                    for k in single_neighbours:
                        if k in dict_str:
                            dict_str[k]=dict_str[k]+1#self.model_weights[j] #if the object is in the dictionary the learner weight is added
                        else:
                            dict_str[k]=1#self.model_weights[j] #if the object is not in the dictionary, it is added
                neigh_tuples = dict_str.items()
                neigh_tuples.sort(key=lambda tup: tup[1], reverse=True)
                neigh_list = [] #final neighborh list for the current query
                for l in range(K):
                    neigh_list.append(neigh_tuples[l][0])
                neighbours_indices.append(neigh_list)

        elif global_mode == 2: #consensus knn with ranking 
            print("@@@@@@ Global Mode is KNN with K=%d ..."%K)
            av_candidates=0.0
            #### QUERY LOCAL TABLES
            if local_mode == 0:#local knn query
                print("@@@@@@ Local Mode is KNN K=%d..."%K_one_table)
                query_neighbours_list, G_queries = self.get_candidates_knn_query(X_queries,X_database,maxmodels=maxmodels,use_weights=use_weights,K=K_one_table,return_G=True)
            
                for i in range(len(query_neighbours_list)):#for each query
                    query_result = query_neighbours_list[i]#get the list of candidates for that query
                    all_sets=set()
                    for j in range(maxmodels):#for each hash table
                        all_sets = all_sets.union(set(query_result[j]))#delete duplicates
                    
                    jointed_list = list(all_sets) #list of candidates without repetitions e.g. jointed_list [1,5,8,9]
                    av_candidates += float(len(jointed_list))
                    ranking = G_queries[i,jointed_list] #get the ranking function for the candidates 
                                                        #QxN matrix, e.g ranking=[0.2,0.8,0.9,-0.5]
                    selected = np.argsort(ranking)[::-1] #rank candidates, e.g. selected =[2,1,0,3]
                    selected = selected[:K] #pick the indices of the best candidates
                    jointed_list = [jointed_list[i] for i in selected.tolist()] #pick the corresponding indices
                    neighbours_indices.append(jointed_list)#jointed_list=[8,5], con K=2

                del query_neighbours_list
                del query_result

            else: #local range query
                print("@@@@@@ Local Mode is Range Query with R=%d ..."%R_one_table)
                best_hamming,  G_queries = self.get_candidates_range_query(X_queries,X_database,Y_queries,Y_database,maxmodels=maxmodels,use_weights=use_weights,rad_hamm=R_one_table,return_G=True)
            
                for i in range(len(X_queries)):#for each query
                    query_best_hamm = best_hamming[i,:]#list of best hamming distances to database points
                    jointed_list = (query_best_hamm <= R_one_table).nonzero()[0]#points with best distance at least R_one_table
                    av_candidates += float(len(jointed_list))
                    ranking = G_queries[i,jointed_list]
                    selected = np.argsort(ranking)[::-1] #rank candidates, e.g. selected =[2,1,0,3]
                    selected = selected[:K] #pick the indices of the best candidates
                    jointed_list = [jointed_list[i] for i in selected.tolist()] #pick the corresponding indices
                    neighbours_indices.append(jointed_list)#jointed_list=[8,5], con K=2

                del best_hamming
                del query_best_hamm

            print "@@@@@@ AV. NUMBER OF CANDIDATES IS %f"%(av_candidates/float(len(X_queries)))

            del G_queries
            del ranking
            del selected
            gc.collect()

        elif global_mode == 3:#consensus range query

            print("@@@@@@ Global Mode is Range Query with R=%f"%R)
            #### QUERY LOCAL TABLES
            if local_mode == 0:#local knn query
                print("@@@@@@ Local Mode is KNN K=%d..."%K_one_table)
                query_neighbours_list, G_queries = self.get_candidates_knn_query(X_queries,X_database,maxmodels=maxmodels,use_weights=use_weights,K=K_one_table,return_G=True)

                for i in range(len(query_neighbours_list)):#for each query
                    query_result = query_neighbours_list[i]#get the list of candidates for that query
                    all_sets=set()
                    for j in range(maxmodels):#for each hash table
                        all_sets = all_sets.union(set(query_result[j]))#delete duplicates
                    
                    jointed_list = list(all_sets) #list of candidates without repetitions e.g. jointed_list [1,5,8,9]
                    norm_factor = 1.0/maxmodels
                    if use_weights is not False:
                        norm_factor = 1.0/np.sum(self.model_weights[:maxmodels])

                    hamming_candidates = 0.5*(self.nbits - norm_factor*G_queries[i,jointed_list])#consensus hamming distance to candidates
                    selected = (hamming_candidates <=R).nonzero()[0]#candidates with consensus hamming distance lower than R
                    jointed_list = [jointed_list[i] for i in selected.tolist()]#original indices of selected candidates
                    neighbours_indices.append(jointed_list)

                del query_neighbours_list

            else: #local range query
                print("@@@@@@ Local Mode is Range Query with R=%d ..."%R_one_table)
                best_hamming,  G_queries = self.get_candidates_range_query(X_queries,X_database,Y_queries,Y_database,maxmodels=maxmodels,use_weights=use_weights,rad_hamm=R_one_table,return_G=True)
            
                for i in range(len(X_queries)):#for each query
                    query_best_hamm = best_hamming[i,:]#list of best hamming distances to database points
                    jointed_list = (query_best_hamm <= R_one_table).nonzero()[0]#points with best distance at least R_one_table
                    norm_factor = 1.0/maxmodels
                    if use_weights is not False:
                        norm_factor = 1.0/np.sum(self.model_weights[:maxmodels])
                    print("QUERY %d: CANDIDATES ARE %d"%(i,len(jointed_list)))
                    hamming_candidates = 0.5*(self.nbits - norm_factor*G_queries[i,jointed_list])#consensus hamming distance to candidates
                    selected = (hamming_candidates <=R).nonzero()[0]#candidates with consensus hamming distance lower than R
                    print("QUERY %d: SELECTED ARE %d"%(i,len(selected)))
                    jointed_list = [jointed_list[i] for i in selected.tolist()]#original indices of selected candidates
                    neighbours_indices.append(jointed_list)

                del best_hamming
  
            del G_queries
            del query_best_hamm
            del selected
            gc.collect()

        else:
            raise Exception('Invalid mode.') 
        
        print "RETURNING"
        return neighbours_indices
        
        
        