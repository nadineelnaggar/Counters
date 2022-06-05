import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str)
parser.add_argument('--train_seq_length',type=int,default=4, help='length of sequences in training set: 4, 8, 16')
parser.add_argument('--long_seq_length',type=int,default=30,help='length of sequences in long test set')




