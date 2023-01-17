import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt
import random
import time
import copy
vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
import IPython
from IPython.display import clear_output
import data
tf.multiply(2,3)
from warprnnt_tensorflow import rnnt_loss
clear_output()
import string


class Encoder(tf.keras.Model):
    def __init__(self, num_inputs, encoder_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.embedding = tf.keras.layers.Embedding(num_inputs, encoder_dim)
        self.gru = tf.keras.layers.GRU(encoder_dim, return_sequences=True, dropout=self.dropout)
        self.linear = tf.keras.layers.Dense(encoder_dim)


    def call(self, x):
        x = self.embedding(x)
        x = self.gru(x)
        x = self.linear(x)
        return x



class Predictor(tf.keras.Model):
    def __init__(self, num_outputs, predictor_dim, joiner_dim, NULL_INDEX):
        super(Predictor, self).__init__()
        self.embed = tf.keras.layers.Embedding(num_outputs, predictor_dim)
        self.rnn = tf.keras.layers.GRUCell(predictor_dim)
        self.linear = tf.keras.layers.Dense(predictor_dim)
        self.initial_state = tf.random.normal([predictor_dim])
        self.start_symbol = NULL_INDEX

    def one_step_forward(self, input, previous_state):
        input = self.embed(input)
        output, state = self.rnn(input, tf.cast(previous_state, tf.float32))
        output = self.linear(output)
        return output, state

    def call(self, y):
        batch_size = tf.shape(y)[0]
        # convert batch_size to int
        batch_size = int(batch_size)
        U = tf.shape(y)[1]
        outs = []
        state = tf.stack([self.initial_state] * batch_size)
        
        for u in range(U+1):  # need U+1 to get null output for final time step
            if u == 0:
                decoder_input = tf.stack(tf.Variable([self.start_symbol] * batch_size))                
            else:
                decoder_input = y[:, u-1]
     
            output, state = self.one_step_forward(decoder_input, state)
            outs.append(output)
        outs = tf.stack(outs, axis=1)
        return outs


class Joiner(tf.keras.Model):
    def __init__(self, maxT, maxU, num_outputs, **kwargs):
        super(Joiner, self).__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(num_outputs+1, use_bias=True)
        self.maxU = maxU
        self.maxT = maxT        

    def call(self, inputs):
        encoder_out, predictor_out = inputs
        encoder_out = tf.tile(tf.expand_dims(encoder_out, axis=2), [1, 1, int(self.maxU+1), 1])
        predictor_out = tf.tile(tf.expand_dims(predictor_out, axis=1), [1, int(self.maxT), 1, 1])
        concat = tf.concat([encoder_out, predictor_out], axis=3)
        out = self.linear(concat)
        return out


class TransducerModel(tf.keras.Model):
    def __init__(self, # For Encoder Module
                    num_inputs, 
                    encoder_dim,
                    num_outputs, 
                    # For Predictor Module
                    predictor_dim, 
                    joiner_dim, 
                    NULL_INDEX,
                    # For Joiner Module
                    maxT, 
                    maxU,
                    **kwargs):
        super(TransducerModel, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.encoder_dim = encoder_dim
        self.num_outputs = num_outputs
        self.predictor_dim = predictor_dim
        self.joiner_dim = joiner_dim
        self.NULL_INDEX = NULL_INDEX
        self.maxT = maxT
        self.maxU = maxU

        self.encoder = Encoder(self.num_inputs, self.encoder_dim)
        self.predictor = Predictor(self.num_outputs, self.predictor_dim, self.joiner_dim, self.NULL_INDEX)
        self.joiner = Joiner(self.maxT, self.maxU, self.num_outputs)

    def call(self, inputs):
        x, y, T, U = inputs
        encoder_out = self.encoder(x)
        predictor_out = self.predictor(y)
        out = self.joiner([encoder_out, predictor_out])
        logits = out

        # print("encoder_out.shape: ", encoder_out.shape)
        # print("predictor_out.shape: ", predictor_out.shape)
        # print("joiner.shape: ", out.shape)

        # competitive dtype
        logits = tf.cast(logits, tf.float32)
        y = tf.cast(y, tf.int32)
        T = tf.cast(T, tf.int32)
        U = tf.cast(U, tf.int32)

        losses = rnnt_loss(logits, y, T, U)

        return losses
        

import data
import copy
test_data = False
if test_data == True:
    file_path = "/build/Desktop/RNNT-for-Vowel-Complete/data/war_and_peace.txt"
    data = data.DataWarAndPeace(file_path)
    train_data, test_data = data.train_data, data.test_data
    for i in train_data.take(1):
        # print(i[0].shape)
        # print(i[1].shape)
        # print(i[2])
        # print(i[3])
        break

    x = copy.deepcopy(i[0])
    y = copy.deepcopy(i[1])
    T = copy.deepcopy(i[2])
    U = copy.deepcopy(i[3])
    x.shape, y.shape

    num_inputs = len(vocab)+1
    encoder_dim = 1024
    num_outputs = len(vocab)+1
    predictor_dim = 1024
    joiner_dim = 1024
    NULL_INDEX = 0
    maxT = 71
    maxU = 74


    model = TransducerModel(num_inputs,
                            encoder_dim,
                            num_outputs,
                            predictor_dim,
                            joiner_dim,
                            NULL_INDEX,
                            maxT,
                            maxU)

    testJoin = model((x,y,T,U))
    print("Log values: ", testJoin)
    print_text = "log_probs shape: {}, labels shape: {}".format(testJoin.shape, y.shape)
    print(print_text)
    print(y[0])
else:
    for i in range(10):
        import string
        y_letters = "CAT"
        y = tf.expand_dims(tf.Variable([string.ascii_uppercase.index(l) + 1 for l in y_letters]), axis=0)
        T = tf.Variable([4])
        U = tf.Variable([len(y_letters)])
        B = 1

        # create tensor shape [B, T, joiner_dim] all value = 0.5
        x = tf.ones([int(B), int(T)]) * 0.5
        x.shape, y.shape


        import string
        num_outputs = len(string.ascii_uppercase) + 1
        num_inputs = 1
        encoder_dim = 1024
        predictor_dim = 1024
        joiner_dim = 1024
        NULL_INDEX = 0
        maxT = T
        maxU = U
        maxU, maxT


        model = TransducerModel(num_inputs,
                                encoder_dim,
                                num_outputs,
                                predictor_dim,
                                joiner_dim,
                                NULL_INDEX,
                                maxT,
                                maxU)

        loss = model((x,y, tf.Variable(T), tf.Variable(U)))
        print_text = "log_probs shape: {}, labels shape: {}".format(loss.shape, y.shape)
        print(loss)
        # print(print_text)

    
