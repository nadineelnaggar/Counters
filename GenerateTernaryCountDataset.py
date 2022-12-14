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