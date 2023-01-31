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
 
        # Trước chỗ  này Long để  là tf.concat nên sai ngu v~ :v
        joiner_input = encoder_out + predictor_out
        out = self.linear(joiner_input)
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
        encoder_out = tf.expand_dims(encoder_out, axis=2)
        predictor_out = tf.expand_dims(predictor_out, axis=1)
        out = self.joiner([encoder_out, predictor_out])
        logits = out

        # competitive dtype
        logits = tf.cast(logits, tf.float32)
        y = tf.cast(y, tf.int32)
        T = tf.cast(T, tf.int32)
        U = tf.cast(U, tf.int32)

        return logits
    
    def greedy_decode(self, x, T, U_max):
        y_batch = []
        B = len(x)
        encoder_out = self.encoder(x)
        for b in range(B):
            t = 0
            u = 0
            y = [self.predictor.start_symbol]
            predictor_state = tf.expand_dims(model.predictor.initial_state, axis=0)

            while t < T[b] and u < U_max:
                predictor_input = tf.expand_dims(tf.Variable(y[-1]), axis=0)
                g_u, predictor_state = self.predictor.one_step_forward(predictor_input, predictor_state)
                f_t = tf.expand_dims(encoder_out[0, 0], axis=0)
                h_t_u = model.joiner([f_t, g_u])
                # find max index in h_t_u
                maxarg = tf.argmax(h_t_u, axis=1)
                max_index = int(maxarg[0])
                if max_index == NULL_INDEX:
                    t += 1
                else:  # is label
                    u += 1
                    y.append(max_index)
            y_batch.append(y[1:])   # remove start symbol
        return y_batch           




class RNNTLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(RNNTLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        x = tf.cast(y_true[0], tf.float32)
        y = tf.cast(y_pred, tf.int32)
        T = tf.cast(y_true[1], tf.int32)
        U = tf.cast(y_true[2], tf.int32)

        

        losses = rnnt_loss(x, y, T, U)
        losses = tf.reduce_mean(losses)   

        return losses
    
    def get_config(self):
        return super().get_config()



# def greedy_decode()
    

import data
import copy
test_data = False
if test_data == True:
    file_path = "/build/data/war_and_peace.txt"
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
    maxT = data.max_input_length
    maxU = data.max_output_length


    model = TransducerModel(num_inputs,
                            encoder_dim,
                            num_outputs,
                            predictor_dim,
                            joiner_dim,
                            NULL_INDEX,
                            maxT,
                            maxU)

    testJoin = model((x,y,T,U))
    # print("Log values: ", testJoin)
    print_text = "log_probs shape: {}, labels shape: {}".format(testJoin.shape, y.shape)
    print(len(x))
else:
    for i in range(1):
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

        # prob = model((x,y, tf.Variable(T), tf.Variable(U)))
        # print_text = "log_probs shape: {}, labels shape: {}".format(prob.shape, y.shape)
        # print(prob)
        # print(print_text)
        x = model.greedy_decode(x, T, U)
        print(x)
        # encoder_out = model.encoder(x)
        # y_pred = [model.predictor.start_symbol]
        # predictor_input = tf.expand_dims(tf.Variable(y_pred[-1]), axis=0)
        # predictor_state = tf.expand_dims(model.predictor.initial_state, axis=0)
        # g_u, predictor_state = model.predictor.one_step_forward(predictor_input, predictor_state)
        # f_t = tf.expand_dims(encoder_out[0, 0], axis=0)
        # h_t_u = model.joiner([f_t, g_u])
        # # find max index in h_t_u
        # max_index = tf.argmax(h_t_u, axis=1)


        # print(predictor_input.shape)
        # print(predictor_state.shape)
        # print(g_u.shape)
        # print(f_t.shape)
        # print(h_t_u.shape)
        # print(h_t_u)
        # print(max_index)
    
