import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt
import random
import time
import copy
vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
import data


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
        encoder_out = tf.tile(tf.expand_dims(encoder_out, axis=2), [1, 1, self.maxU+1, 1])
        predictor_out = tf.tile(tf.expand_dims(predictor_out, axis=1), [1, self.maxT, 1, 1])
        concat = tf.concat([encoder_out, predictor_out], axis=3)
        out = self.linear(concat)
        return out


if __name__=='__main__':

    file_path = "/home/adminvbdi/Desktop/RNNT-for-Vowel-Complete/data/war_and_peace.txt"
    data = data.DataWarAndPeace(file_path)
    train_data, test_data = data.train_data, data.test_data
    for i in train_data.take(1):
        print(i[0].shape)
        print(i[1].shape)
        print(i[2])
        print(i[3])

    x = copy.deepcopy(i[0])
    y = copy.deepcopy(i[1])
    x.shape, y.shape

    # test encoder
    num_inputs = len(vocab)+1
    encoder_dim = 64
    encoder = Encoder(num_inputs, encoder_dim)
    testEnc = encoder(x)
    print("Test Encoder: ", testEnc.shape)

    # Test Predictor
    num_outputs = len(vocab)+1
    predictor_dim = 64
    joiner_dim = 64
    predictor = Predictor(num_outputs, predictor_dim, joiner_dim, 0)
    testPred = predictor(y)
    print("Test Predictor: ", testPred.shape)

    # Test Joiner
    joiner = Joiner(71, 74, num_outputs)
    testJoin = joiner([testEnc, testPred])
    print_text = "log_probs shape: {}, labels shape: {}".format(testJoin.shape, y.shape)
    print(print_text)