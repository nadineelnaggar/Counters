import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import argparse
from models import NonZeroReLUCounter,LinearBracketCounter




parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str)
parser.add_argument('--train_seq_length',type=int,default=4, help='length of sequences in training set: 4, 8, 16')
# parser.add_argument('--long_seq_length',type=int,default=30,help='length of sequences in long test set (20,30,40,50 tokens)')
parser.add_argument('--runtime',type=str,default='colab',help='colab or local')
parser.add_argument('--num_epochs',type=int)
parser.add_argument('--output_activation',type=str,default='Sigmoid',help='Sigmoid or Clipping')


args = parser.parse_args()


model_name = args.model_name
train_seq_length = args.train_seq_length
# long_test_length = args.long_test_length
runtime = args.runtime
num_epochs = args.num_epochs
output_activation = args.output_activation


input_size = 2
output_size = 1
hidden_size = 2
counter_input_size = 3
counter_output_size = 1
vocab = ['(', ')']



num_classes = 2
n_letters = 2

def read_datasets():
    x = []
    y = []
    if train_seq_length==2:
        if model_name=='NonZeroReLUCounter':
            read_file = 'Dyck1Dataset2Tokens.txt'
        elif model_name=='LinearBracketCounter':
            read_file = 'CounterDataset2Tokens.txt'
    elif train_seq_length==4:
        if model_name=='NonZeroReLUCounter':
            read_file = 'Dyck1Dataset4Tokens.txt'
        elif model_name=='LinearBracketCounter':
            read_file = 'CounterDataset4Tokens.txt'
    elif train_seq_length==8:
        if model_name=='NonZeroReLUCounter':
            read_file = 'Dyck1Dataset8Tokens.txt'
        elif model_name=='LinearBracketCounter':
            read_file = 'CounterDataset8Tokens.txt'

    with open(read_file, 'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            x.append(sentence)
            y.append(label)

    x_20 = []
    y_20 = []
    x_30 = []
    y_30 = []
    x_40 = []
    y_40 = []
    x_50 = []
    y_50 = []

    if model_name=='NonZeroReLUCounter':
        with open('Dyck1Dataset20Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_20.append(sentence)
                y_20.append(label)
        with open('Dyck1Dataset30Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_30.append(sentence)
                y_30.append(label)

        with open('Dyck1Dataset40Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_40.append(sentence)
                y_40.append(label)

        with open('Dyck1Dataset50Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_50.append(sentence)
                y_50.append(label)

    elif model_name == 'LinearBracketCounter':
        with open('CounterDataset20Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_20.append(sentence)
                y_20.append(label)
        with open('CounterDataset30Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_30.append(sentence)
                y_30.append(label)

        with open('CounterDataset40Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_40.append(sentence)
                y_40.append(label)

        with open('CounterDataset50Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_50.append(sentence)
                y_50.append(label)


    return x, y, x_20, y_20, x_30, y_30, x_40, y_40, x_50, y_50




