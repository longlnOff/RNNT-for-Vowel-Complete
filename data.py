import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt
import random
import time


class DataWarAndPeace():
    def __init__(self, orginal_data_path) -> None:
        self.orginal_data_path = orginal_data_path
        self.data = self.load_data()
        self.NULL_INDEX = 0
        self.vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
        self.max_input_length = 71
        self.max_output_length = 74
        self.batch_size = 4
        self.train_data, self.test_data = self.get_data()

    def get_label_input(self, line):
        return line, line

    def load_data(self):
        with open(self.orginal_data_path, 'r') as f:
            data = f.readlines()
        return data

    def split_data(self):
        train_data = self.data[:int(len(self.data)*0.9)]
        test_data = self.data[int(len(self.data)*0.9):]
        train_data = tf.data.experimental.from_list(
            elements=train_data)
        test_data = tf.data.experimental.from_list(
            elements=test_data)
        return train_data, test_data

    # remove vowels from input
    def remove_vowels(self, line):
        vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        for vowel in vowels:
            line = tf.strings.regex_replace(line, vowel, '')
        return line


    # Encode String
    def encode_string(self, s):
        s = tf.strings.unicode_split(s, "UTF-8")
        vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
        vocab_list = tf.constant(list(vocab))
        table = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                vocab_list, tf.range(1, len(vocab)+1, dtype=tf.int64)
            ), len(vocab))
        return table.lookup(s)

    
    # Padding
    def padding(self, data_object, max_length):
        return tf.pad(data_object, [[0, max_length - tf.shape(data_object)[0]]])

    def get_data(self):
        train_data, test_data = self.split_data()

        # Get label and input
        train_data = train_data.map(self.get_label_input)
        test_data = test_data.map(self.get_label_input)

        # Remove vowels
        train_data = train_data.map(
            lambda input, label: (self.remove_vowels(input), label))
        test_data = test_data.map(
            lambda input, label: (self.remove_vowels(input), label))

        # Encode String
        train_data = train_data.map(
            lambda input, label: (self.encode_string(input), self.encode_string(label)))
        test_data = test_data.map(
            lambda input, label: (self.encode_string(input), self.encode_string(label)))

        train_data = train_data.map(
            lambda input, label: (input, label, tf.size(input), tf.size(label)))
        test_data = test_data.map(
            lambda input, label: (input, label, tf.size(input), tf.size(label)))

        # Padding
        train_data = train_data.map(
            lambda input, label, input_length, label_length: 
                    (self.padding(input, self.max_input_length), 
                    self.padding(label, self.max_output_length), 
                    input_length, 
                    label_length))
        # test_data = test_data.map(
        #     lambda input, label, input_length, label_length:
        #             (self.padding(input, 71),
        #             self.padding(label, 74),
        #             input_length,
        #             label_length))

        train_data = train_data.batch(self.batch_size)

        return train_data, test_data

                            



                            


if __name__=='__main__':
    file_path = "/home/adminvbdi/Desktop/RNNT-for-Vowel-Complete/data/war_and_peace.txt"
    data = DataWarAndPeace(file_path)
    train_data, test_data = data.train_data, data.test_data
    for i in train_data.take(1):
        print(i[0].shape)
        print(i[1].shape)
        print(i[2])
        print(i[3])

