import tensorflow as tf
import numpy as np
import data
import model
import copy
import os
from model import Encoder, Predictor, Joiner, vocab
from data import DataWarAndPeace
import warprnnt_tensorflow




file_path = "/build/Desktop/RNNT-for-Vowel-Complete/data/war_and_peace.txt"
data = DataWarAndPeace(file_path)
train_data, test_data = data.train_data, data.test_data

for count, batch_data in enumerate(train_data):

    x, y = batch_data[0], batch_data[1]
    input_lengths = batch_data[2]
    output_lengths = batch_data[3]    
    
    # test encoder
    num_inputs = len(vocab)+1
    encoder_dim = 64
    encoder = Encoder(num_inputs, encoder_dim)
    testEnc = encoder(x)
    # print("Test Encoder: ", testEnc.shape)

    # Test Predictor
    num_outputs = len(vocab)+1
    predictor_dim = 64
    joiner_dim = 64
    predictor = Predictor(num_outputs, predictor_dim, joiner_dim, 0)
    testPred = predictor(y)
    # print("Test Predictor: ", testPred.shape)

    # Test Joiner
    joiner = Joiner(71, 74, num_outputs)
    testJoin = joiner([testEnc, testPred])
    print_text = "log_probs shape: {}, labels shape: {}".format(testJoin.shape, y.shape)
    # print(print_text)

    # Convert to softmax format
    testJoin = tf.nn.softmax(testJoin)

    # Fit type for RNNT
    testJoin = tf.cast(testJoin, tf.float32)
    y = tf.cast(y, tf.int32)
    input_lengths = tf.cast(input_lengths, tf.int32)
    output_lengths = tf.cast(output_lengths, tf.int32)


    costs = warprnnt_tensorflow.rnnt_loss(testJoin, y, input_lengths, output_lengths)

    print("costs: ", costs)

    if count == 100:
        break
