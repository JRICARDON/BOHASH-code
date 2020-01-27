#Implementation of a Siamese Net for Hashing
"""
@author: ricky

Observations: Finished on May 23, 17.
              Tested with tester_siamese.py, the following methods
              - constructor ... OK
              - fit ... OK
              - predict ... OK
              - hash ... OK and gives reasonable results

"""
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop, Adadelta
from keras import backend as K

def hamming_layer_function(vects,nb):
    x, y = vects
    return 0.5*(nb - K.batch_dot(x, y, axes=1))

def hamming_layer_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def hashing_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def create_base_network(input_dim,nhidden,nbits):
    seq = Sequential(name='tied_layer')
    if nhidden > 0:

        print("NHIDDEN TYPE")
        print(type(nhidden))
        print(nhidden)

        seq.add(Dense(nhidden, input_shape=(input_dim,), activation='relu',name='tied_hidden'))
        seq.add(Dense(nbits,activation='tanh',name='tied_hashing'))
    else:
        seq.add(Dense(nbits, input_shape=(input_dim,), activation='tanh',name='tied_hashing'))
    return seq

def sign_layer_function(x):
    return K.sign(x)

def sign_layer_shape(input_shape):
    return input_shape

class Siamese_Net(object):

    def __init__(self,input_dim,nbits,nhidden=0):
        self.input_dim = input_dim
        self.nbits = nbits
        self.nhidden = nhidden
        base_network = create_base_network(input_dim,nhidden,nbits)
        input_a = Input(shape=(self.input_dim,),name='input1')
        input_b = Input(shape=(self.input_dim,),name='input2')
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        hamming_layer = Lambda(hamming_layer_function, output_shape=hamming_layer_shape,
            arguments={'nb': int(self.nbits)},name='hamming_layer')([processed_a, processed_b])
        self.model = Model([input_a, input_b], hamming_layer)
        rms = Adadelta()
        self.model.compile(loss=hashing_loss, optimizer=rms)
        self.model.summary()
        self.hasher = None

    def fit(self,tr_pairs,tr_y,nb_epoch,batch_size):

        print("SHAPE X, Y TRAIN")
        print(tr_pairs.shape)
        print(tr_y.shape)

        self.hasher = None
        self.model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=nb_epoch)
        return self.model

    def predict(self,te_pairs):
        return self.model.predict([te_pairs[:, 0], te_pairs[:, 1]])

    def explore_net(self):
        print([x.name for x in self.model.layers])
        mseq=self.model.layers[2]
        print(mseq)
        print([x.name for x in mseq.layers])
        tied_layer = self.model.get_layer("tied_layer")
        hashing_layer = tied_layer.get_layer("tied_hashing")
        weights = hashing_layer.get_weights()
        print(weights[1].shape)
        print(weights[0].shape)

    def build_hasher_from_net(self):      
        tied_layer = self.model.get_layer("tied_layer")
        self.hasher = None

        if self.nhidden > 0:

            hidden_layer = tied_layer.get_layer("tied_hidden")
            hashing_layer = tied_layer.get_layer("tied_hashing")
            input_x = Input(shape=(self.input_dim,))
            hidden_x = Dense(self.nhidden,activation='relu',weights=hidden_layer.get_weights())(input_x)
            pre_sign = Dense(self.nbits,activation='linear',weights=hashing_layer.get_weights())(hidden_x)
            hash_output = Lambda(sign_layer_function,output_shape=sign_layer_shape)(pre_sign)   
            self.hasher = Model(input_x, hash_output)

        else:

            hashing_layer = tied_layer.get_layer("tied_hashing")
            input_x = Input(shape=(self.input_dim,))
            pre_sign = Dense(self.nbits,activation='linear',weights=hashing_layer.get_weights())(input_x)
            hash_output = Lambda(sign_layer_function,output_shape=sign_layer_shape)(pre_sign)   
            self.hasher = Model(input_x, hash_output)

        return self.hasher

    def hash(self,data):
        if self.hasher == None:
            self.build_hasher_from_net()
        return self.hasher.predict(data) 

    def get_hashingW(self):
        tied_layer = self.model.get_layer("tied_layer")        
        hashing_layer = tied_layer.get_layer("tied_hashing")
        weights=hashing_layer.get_weights()
        return weights[0]