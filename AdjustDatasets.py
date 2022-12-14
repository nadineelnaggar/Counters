import pandas as pd
import sklearn
from sklearn.utils import shuffle

import pandas as pd
import sklearn
from sklearn.utils import shuffle
import random
from random import randint
import math



def removeOversampling(dataset_name):
    x = []
    y = []
    with open(dataset_name,'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            if sentence not in x:
                x.append(sentence)
                y.append(label)
    print('len x = ',len(x))
    print('len y = ',len(y))
    if dataset_name=='CounterDataset2Tokens.txt':
        with open('CounterDataset2TokensNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset4Tokens.txt':
        with open('CounterDataset4TokensNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset8Tokens.txt':
        with open('CounterDataset8TokensNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset16Tokens.txt':
        with open('CounterDataset16TokensNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')

#
# removeOversampling('CounterDataset2Tokens.txt')
# removeOversampling('CounterDataset4Tokens.txt')
# removeOversampling('CounterDataset8Tokens.txt')
# removeOversampling('CounterDataset16Tokens.txt')


def makeDatasetTernary(dataset_name):
    x = []
    y = []
    with open(dataset_name, 'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            if sentence not in x:
                x.append(sentence)
                if sentence.count('(') > sentence.count(')'):
                    y.append('Pos')
                elif sentence.count('(')==sentence.count(')'):
                    y.append('Zero')
                elif sentence.count('(') < sentence.count(')'):
                    y.append('Neg')
                # y.append(label)
    print('len x = ', len(x))
    print('len y = ', len(y))
    if dataset_name=='CounterDataset2Tokens.txt':
        with open('CounterDataset2TokensTernary.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset4Tokens.txt':
        with open('CounterDataset4TokensTernary.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset8Tokens.txt':
        with open('CounterDataset8TokensTernary.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset16Tokens.txt':
        with open('CounterDataset16TokensTernary.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset2TokensNoOversampling.txt':
        with open('CounterDataset2TokensTernaryNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset4TokensNoOversampling.txt':
        with open('CounterDataset4TokensTernaryNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset8TokensNoOversampling.txt':
        with open('CounterDataset8TokensTernaryNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')
    elif dataset_name=='CounterDataset16TokensNoOversampling.txt':
        with open('CounterDataset16TokensTernaryNoOversampling.txt', 'a') as f:
            for i in range(len(x)):
                f.write(x[i] + ',' + y[i] + '\n')

# makeDatasetTernary('CounterDataset2Tokens.txt')
# makeDatasetTernary('CounterDataset4Tokens.txt')
# makeDatasetTernary('CounterDataset8Tokens.txt')
makeDatasetTernary('CounterDataset16Tokens.txt')
# makeDatasetTernary('CounterDataset2TokensNoOversampling.txt')
# makeDatasetTernary('CounterDataset4TokensNoOversampling.txt')
# makeDatasetTernary('CounterDataset8TokensNoOversampling.txt')
makeDatasetTernary('CounterDataset16TokensNoOversampling.txt')