# import pandas as pd
# import sklearn
# from sklearn.utils import shuffle
#
# import pandas as pd
# import sklearn
# from sklearn.utils import shuffle
# import random
# from random import randint
# import math
#
#
#
#
#
#
#
# class Dyck1_Generator(object):
#     def generateParenthesis(self, n):
#         def generate(A = []):
#             if len(A) == 2*n:
#                 if valid(A):
#                     val.append("".join(A))
#                 else:
#                     inval.append("".join(A))
#             else:
#                 A.append('(')
#                 generate(A)
#                 A.pop()
#                 A.append(')')
#                 generate(A)
#                 A.pop()
#
#         def valid(A):
#             bal = 0
#             for c in A:
#                 if c == '(': bal += 1
#                 else: bal -= 1
#                 if bal < 0: return False
#             return bal == 0
#
#         val = []
#         inval = []
#         generate()
#         return val, inval
#
# def generateDataset(n_bracket_pairs_start, n_bracket_pairs_end):
#     gen = Dyck1_Generator()
#     # d1_valid, d1_invalid = gen.generateParenthesis(3)
#     d1_valid = []
#     d1_invalid = []
#     for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
#         x,y = gen.generateParenthesis(i)
#         for elem in x:
#             d1_valid.append(elem)
#         for elem in y:
#             d1_invalid.append(elem)
#     return d1_valid,d1_invalid
#
# class Count_Task_Bracket_Generator(object):
#     def generateParenthesis(self, n):
#         def generate(A = []):
#             if len(A) == 2*n:
#                 if pos(A):
#                     positive.append("".join(A))
#                 else:
#                     zero_neg.append("".join(A))
#             else:
#                 A.append('(')
#                 generate(A)
#                 A.pop()
#                 A.append(')')
#                 generate(A)
#                 A.pop()
#
#         def pos(A):
#             bal = 0
#             for c in A:
#                 if c == '(': bal += 1
#                 else: bal -= 1
#                 # if bal <= 0: return False
#             return bal > 0
#
#         positive = []
#         zero_neg = []
#         generate()
#         return positive, zero_neg
#
#
# def generateCountDataset(n_bracket_pairs_start, n_bracket_pairs_end):
#     gen = Count_Task_Bracket_Generator()
#     # d1_valid, d1_invalid = gen.generateParenthesis(3)
#     pos = []
#     zero_neg = []
#     for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
#         x,y = gen.generateParenthesis(i)
#         for elem in x:
#             pos.append(elem)
#         for elem in y:
#             zero_neg.append(elem)
#     return pos,zero_neg
#
#
# def generateLabelledCountDataset(n_bracket_pairs_start,n_bracket_pairs_end):
#     pos, zero_neg = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
#     dataset = []
#     sentences = []
#     labels = []
#     for elem in pos:
#         entry = (elem, 'Pos')
#         dataset.append(entry)
#         sentences.append(elem)
#         labels.append('Pos')
#     for elem in zero_neg:
#         entry = (elem,'Zero_Neg')
#         dataset.append(entry)
#         sentences.append(elem)
#         labels.append('Zero_Neg')
#     sentences, labels = shuffle(sentences, labels,random_state=0)
#     return sentences, labels

# x = []
# y = []

import random
from random import randint
from sklearn.utils import shuffle

def balanceDataset(dataset):
    x = []
    y = []
    pos_x = []
    neg_x = []
    zero_x = []
    with open(dataset,'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            x.append(sentence)
            y.append(label)
            if label=='Pos':
                pos_x.append(sentence)
            elif label=='Neg':
                neg_x.append(sentence)
            elif label=='Zero':
                zero_x.append(sentence)

    count_pos = len(pos_x)
    count_neg = len(neg_x)
    count_zero = len(zero_x)
    
    majority_length = max(count_neg,count_pos,count_zero)
    count_pos1 = count_pos
    while(count_pos1<majority_length):
        idx = randint(0,count_pos-1)
        x.append(pos_x[idx])
        y.append('Pos')
        count_pos1+=1
    
    count_neg1 = count_neg
    while (count_neg1 < majority_length):
        idx = randint(0,count_neg-1)
        x.append(neg_x[idx])
        y.append('Neg')
        count_neg1 += 1

    count_zero1 = count_zero
    while (count_zero1 < majority_length):
        idx = randint(0,count_zero-1)
        x.append(zero_x[idx])
        y.append('Zero')
        count_zero1 += 1

    # x, y = random.shuffle(x,y)
    x,y = shuffle(x,y)
    dataset_new = dataset[:-4]
    print(dataset_new)
    dataset_new = dataset_new+'OversampledDataset.txt'
    print(dataset_new)
    with open(dataset_new, 'a') as f:
        for i in range(len(x)):
            f.write(x[i] + ',' + y[i] + '\n')

# balanceDataset('CounterDataset2TokensTernary.txt')
# balanceDataset('CounterDataset4TokensTernary.txt')
# balanceDataset('CounterDataset8TokensTernary.txt')
# balanceDataset('CounterDataset10TokensTernary.txt')
# balanceDataset('CounterDataset20TokensTernary.txt')
# balanceDataset('CounterDataset50TokensTernary.txt')
# balanceDataset('CounterDataset30TokensTernary.txt')
# balanceDataset('CounterDataset40TokensTernary.txt')
#
        
    
