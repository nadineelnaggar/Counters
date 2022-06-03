import pandas as pd
import sklearn
from sklearn.utils import shuffle

import pandas as pd
import sklearn
from sklearn.utils import shuffle
import random
from random import randint
import math

class Dyck1_Generator(object):
    def generateParenthesis(self, n):
        def generate(A = []):
            if len(A) == 2*n:
                if valid(A):
                    val.append("".join(A))
                else:
                    inval.append("".join(A))
            else:
                A.append('(')
                generate(A)
                A.pop()
                A.append(')')
                generate(A)
                A.pop()

        def valid(A):
            bal = 0
            for c in A:
                if c == '(': bal += 1
                else: bal -= 1
                if bal < 0: return False
            return bal == 0

        val = []
        inval = []
        generate()
        return val, inval

def generateDataset(n_bracket_pairs_start, n_bracket_pairs_end):
    gen = Dyck1_Generator()
    # d1_valid, d1_invalid = gen.generateParenthesis(3)
    d1_valid = []
    d1_invalid = []
    for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
        x,y = gen.generateParenthesis(i)
        for elem in x:
            d1_valid.append(elem)
        for elem in y:
            d1_invalid.append(elem)
    return d1_valid,d1_invalid

# d1_valid, d1_invalid = generateDataset(6)


# print(d1_valid)
# print('///////////////////////')
# print(d1_invalid)


dataset_valid, dataset_invalid = generateDataset(2,2)
print('dataset valid = ',dataset_valid)
print('len(dataset_valid) = ',len(dataset_valid))
print('dataset invalid = ',dataset_invalid)
print('len(dataset_invalid) = ',len(dataset_invalid))


dataset_valid1, dataset_invalid1 = generateDataset(4,4)
print('dataset valid = ',dataset_valid1)
print('len(dataset_valid) = ',len(dataset_valid1))
print('dataset invalid = ',dataset_invalid1)
print('len(dataset_invalid) = ',len(dataset_invalid1))


dataset_valid2, dataset_invalid2 = generateDataset(8,8)
print('dataset valid = ',dataset_valid2)
print('len(dataset_valid) = ',len(dataset_valid2))
# print('dataset invalid = ',dataset_invalid2)
print('len(dataset_invalid) = ',len(dataset_invalid2))
# add the labels and then create the csv file to complete the dataset

def generateLabelledDataset(n_bracket_pairs_start,n_bracket_pairs_end):
    d1_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
    dataset = []
    sentences = []
    labels = []
    for elem in d1_valid:
        entry = (elem, 'valid')
        dataset.append(entry)
        sentences.append(elem)
        labels.append('valid')
    for elem in d1_invalid:
        entry = (elem,'invalid')
        dataset.append(entry)
        sentences.append(elem)
        labels.append('invalid')
    sentences, labels = shuffle(sentences, labels,random_state=0)
    return sentences, labels

print(generateLabelledDataset(2,2))


#this function generates a balanced labelled dataset consisting of bracket sequences of lengths ranging from
# n_bracket_pairs_start to n_bracket_pairs_end. if a size limit is defined then
def generateBalancedLabelledDistinctDataset(n_bracket_pairs_start,n_bracket_pairs_end, size=15000, size_limit=False):
    d1_valid, d1_invalid = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
    d1_valid = shuffle(d1_valid)
    d1_invalid = shuffle(d1_invalid)
    dataset = []
    sentences = []
    labels = []
    count_valid=0
    count_invalid=0
    for elem in d1_valid:
        entry = (elem, 'valid')
        if elem not in sentences and size_limit==True and count_valid<(size/2):
            dataset.append(entry)
            sentences.append(elem)
            labels.append('valid')
            count_valid+=1
    for elem in d1_invalid:
        if elem not in sentences and size_limit==True and count_valid < (size / 2):
            entry = (elem,'invalid')
            dataset.append(entry)
            sentences.append(elem)
            labels.append('invalid')
            count_invalid+=1
    sentences, labels = shuffle(dataset, sentences, labels,random_state=0)

    return dataset, sentences, labels


def oversampleDyck1Minority(valid, invalid):
    num_valid = len(valid)
    num_invalid = len(invalid)
    difference = len(invalid)-len(valid)
    all_elements = []

    count_valid_all_elements = 0
    count_invalid_all_elements = 0

    for i in range(num_invalid):
        if invalid[i] not in all_elements:
            all_elements.append(invalid[i])
            count_invalid_all_elements+=1

    for i in range(num_valid):
        if valid[i] not in all_elements:
            all_elements.append(valid[i])
            count_valid_all_elements+=1

    # for i in range(difference):
    #     idx = randint(0,num_valid-1)
    #     all_elements.append(valid[idx])
    #     count_valid_all_elements+=1

    num_loops = math.ceil(difference/num_valid)

    for i in range(num_loops):
        for j in range(num_valid):
            if count_invalid_all_elements>count_valid_all_elements:
                all_elements.append(valid[j])
                count_valid_all_elements+=1

    print('count_valid_all_elements = ',count_valid_all_elements)
    print('count_invalid_all_elements = ',count_invalid_all_elements)

    return all_elements

def oversampleMinority(class1, class1_label, class2, class2_label):
    num_class1 = len(class1)
    num_class2 = len(class2)

    # difference = len(invalid)-len(valid)
    all_elements = []
    all_labels = []
    count_class1_all_elements = 0
    count_class2_all_elements = 0

    if num_class1==num_class2:
        for i in range(len(class1)):
            all_elements.append(class1[i])
            all_labels.append(class1_label)
            count_class1_all_elements+=1
            all_elements.append(class2[i])
            all_labels.append(class2_label)
            count_class2_all_elements+=1

    elif num_class1>num_class2:
        difference = class1-class2
        num_loops = math.ceil(difference/class2)
        for i in range(len(class1)):
            all_elements.append(class1[i])
            all_labels.append(class1_label)
            count_class1_all_elements+=1
        for i in range(len(class2)):
            all_elements.append(class2[i])
            all_labels.append(class2_label)
            count_class2_all_elements+=1

        for i in range(num_loops):
            for j in range(len(class2)):
                if count_class2_all_elements<count_class1_all_elements:
                    all_elements.append(class2[j])
                    all_labels.append(class2_label)
                    count_class2_all_elements+=1

    elif num_class2>num_class1:
        difference = len(class2)-len(class1)
        num_loops = math.ceil(difference/len(class1))
        for i in range(len(class1)):
            all_elements.append(class1[i])
            all_labels.append(class1_label)
            count_class1_all_elements+=1
        for i in range(len(class2)):
            all_elements.append(class2[i])
            all_labels.append(class2_label)
            count_class2_all_elements+=1
        for i in range(num_loops):
            for j in range(len(class1)):
                if count_class1_all_elements<count_class2_all_elements:
                    all_elements.append(class1[i])
                    all_labels.append(class1_label)
                    count_class1_all_elements+=1


    all_elements,all_labels = shuffle(all_elements,all_labels,random_state=0)
    return all_elements, all_labels

oversampled =oversampleDyck1Minority(dataset_valid,dataset_invalid)
oversampled1=oversampleDyck1Minority(dataset_valid1,dataset_invalid1)

print(oversampled)
count_valid1 = 0
count_valid2 = 0

for i in range(len(oversampled)):
    if oversampled[i]=='()()':
        count_valid1+=1
    elif oversampled[i]=='(())':
        count_valid2+=1

print(count_valid1)
print(count_valid2)
#
# #generate dataset similar to Suzgun paper
# dataset, sentences, labels = generateBalancedLabelledDistinctDataset(1,25,size=15000,size_limit=True)
#
#
# with open('Dyck1_Dataset_25pairs_balanced.txt','a') as f:
#     for i in range(len(sentences)):
#         f.write(sentences[i]+','+labels[i]+'\n')
#
# dataset_length, sentences_length, labels_length = generateBalancedLabelledDistinctDataset(25,50,size=5000,size_limit=True)
# with open('Dyck1_Dataset_25pairs_balanced_length.txt','a') as f:
#     for i in range(len(sentences_length)):
#         f.write(sentences_length[i]+','+labels_length[i]+'\n')



class Count_Task_Bracket_Generator(object):
    def generateParenthesis(self, n):
        def generate(A = []):
            if len(A) == 2*n:
                if pos(A):
                    positive.append("".join(A))
                else:
                    zero_neg.append("".join(A))
            else:
                A.append('(')
                generate(A)
                A.pop()
                A.append(')')
                generate(A)
                A.pop()

        def pos(A):
            bal = 0
            for c in A:
                if c == '(': bal += 1
                else: bal -= 1
                if bal < 0: return False
            return bal > 0

        positive = []
        zero_neg = []
        generate()
        return positive, zero_neg


def generateCountDataset(n_bracket_pairs_start, n_bracket_pairs_end):
    gen = Count_Task_Bracket_Generator()
    # d1_valid, d1_invalid = gen.generateParenthesis(3)
    pos = []
    zero_neg = []
    for i in range(n_bracket_pairs_start,n_bracket_pairs_end+1):
        x,y = gen.generateParenthesis(i)
        for elem in x:
            pos.append(elem)
        for elem in y:
            zero_neg.append(elem)
    return pos,zero_neg


def generateLabelledCountDataset(n_bracket_pairs_start,n_bracket_pairs_end):
    pos, zero_neg = generateDataset(n_bracket_pairs_start,n_bracket_pairs_end)
    dataset = []
    sentences = []
    labels = []
    for elem in pos:
        entry = (elem, 'Pos')
        dataset.append(entry)
        sentences.append(elem)
        labels.append('Pos')
    for elem in zero_neg:
        entry = (elem,'Zero_Neg')
        dataset.append(entry)
        sentences.append(elem)
        labels.append('Zero_Neg')
    sentences, labels = shuffle(sentences, labels,random_state=0)
    return sentences, labels

# print(generateLabelledCountDataset(2,2))
# print(generateLabelledCountDataset(4,4))


# print(generateLabelledCountDataset(8,8))

seqsCount4TokensPos,seqsCount4TokensZeroNeg = generateCountDataset(2,2)
count_seqs, count_labels = oversampleMinority(seqsCount4TokensPos,'Pos',seqsCount4TokensZeroNeg,'ZeroNeg')
print(count_seqs)
print(count_labels)

#GENERATE THE LONGER SETS BY USING THE SUZGUN SEQUENCES AND THEN ADAPTING THEM TO MAKE THEM INVALID OR WHATEVER, TO MATCH THE DATASETS WE NEED

# print(generateLabelledCountDataset(10,10))
# print(generateLabelledCountDataset(15,15))