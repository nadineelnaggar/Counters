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
                    all_elements.append(class1[j])
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
                # if bal <= 0: return False
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

seqsCount2TokensPos, seqsCount2TokensZeroNeg = generateCountDataset(1,1)

count2_seqs, count2_labels = oversampleMinority(seqsCount2TokensPos,'Pos', seqsCount2TokensZeroNeg,'ZeroNeg')


seqsCount4TokensPos,seqsCount4TokensZeroNeg = generateCountDataset(2,2)
count4_seqs, count4_labels = oversampleMinority(seqsCount4TokensPos,'Pos',seqsCount4TokensZeroNeg,'ZeroNeg')
print(count4_seqs)
print(count4_labels)

seqsCount8TokensPos, seqsCount8TokensZeroNeg = generateCountDataset(4,4)
count8_seqs, count8_labels = oversampleMinority(seqsCount8TokensPos,'Pos', seqsCount8TokensZeroNeg,'ZeroNeg')

seqsCount10TokensPos, seqsCount10TokensZeroNeg = generateCountDataset(5,5)
count10_seqs, count10_labels = oversampleMinority(seqsCount10TokensPos,'Pos', seqsCount10TokensZeroNeg,'ZeroNeg')

seqsCount16TokensPos, seqsCount16TokensZeroNeg = generateCountDataset(8,8)
count16_seqs, count16_labels = oversampleMinority(seqsCount16TokensPos,'Pos', seqsCount16TokensZeroNeg,'ZeroNeg')

# with open('CountDataset2Tokens.txt','a') as f:
#     for i in range(len(count2_seqs)):
#         f.write(count2_seqs[i]+','+count2_labels[i]+'\n')
#
#
# with open('CounterDataset4Tokens.txt','a') as f:
#     for i in range(len(count4_seqs)):
#         f.write(count4_seqs[i] + ',' + count4_labels[i] + '\n')
#
# with open('CounterDataset8Tokens.txt','a') as f:
#     for i in range(len(count8_seqs)):
#         f.write(count8_seqs[i] + ',' + count8_labels[i] + '\n')

count_upto4_seqs = []
count_upto4_labels = []

for i in range(len(count4_seqs)):
    count_upto4_seqs.append(count4_seqs[i])
    count_upto4_labels.append(count4_labels[i])

for i in range(len(count2_seqs)):
    count_upto4_seqs.append(count2_seqs[i])
    count_upto4_labels.append(count2_labels[i])

count_upto4_seqs,count_upto4_labels=shuffle(count_upto4_seqs,count_upto4_labels)

count_upto8_seqs = []
count_upto8_labels = []

for i in range(len(count_upto4_seqs)):
    count_upto8_seqs.append(count_upto4_seqs[i])
    count_upto8_labels.append(count_upto4_labels[i])

for i in range(len(count8_seqs)):
    count_upto8_seqs.append(count8_seqs[i])
    count_upto8_labels.append(count8_labels[i])

count_upto8_seqs,count_upto8_labels = shuffle(count_upto8_seqs,count_upto8_labels)


count_upto16_seqs = []
count_upto16_labels = []

for i in range(len(count_upto4_seqs)):
    count_upto16_seqs.append(count_upto4_seqs[i])
    count_upto16_labels.append(count_upto4_labels[i])

for i in range(len(count8_seqs)):
    count_upto16_seqs.append(count8_seqs[i])
    count_upto16_labels.append(count8_labels[i])

for i in range(len(count16_seqs)):
    count_upto16_seqs.append(count16_seqs[i])
    count_upto16_labels.append(count16_labels[i])

# for i in range(len(count10_seqs)):
#     count_upto16_seqs.append(count10_seqs[i])
#     count_upto16_labels.append(count10_labels[i])

count_upto16_seqs,count_upto16_labels = shuffle(count_upto16_seqs,count_upto16_labels)

# with open('CounterDataset2Tokens.txt','w') as f:
#     f.write('')

# with open('CounterDataset4Tokens.txt','w') as f:
#     f.write('')
#
# with open('CounterDataset8Tokens.txt','w') as f:
#     f.write('')


# with open('CounterDataset10Tokens.txt','w') as f:
#     f.write('')

# with open('Dyck1Dataset2Tokens.txt','w') as f:
#     f.write('')
#
# with open('Dyck1Dataset4Tokens.txt','w') as f:
#     f.write('')
#
# with open('Dyck1Dataset8Tokens.txt','w') as f:
#     f.write('')

# with open('Dyck1Dataset10Tokens.txt','w') as f:
#     f.write('')

with open('Dyck1Dataset16Tokens.txt','w') as f:
    f.write('')

# with open('CounterDataset2Tokens.txt','a') as f:
#     for i in range(len(count2_seqs)):
#         f.write(count2_seqs[i]+','+count2_labels[i]+'\n')


# with open('CounterDataset4Tokens.txt','a') as f:
#     for i in range(len(count_upto4_seqs)):
#         f.write(count_upto4_seqs[i] + ',' + count_upto4_labels[i] + '\n')
#
# with open('CounterDataset8Tokens.txt','a') as f:
#     for i in range(len(count_upto8_seqs)):
#         f.write(count_upto8_seqs[i] + ',' + count_upto8_labels[i] + '\n')


# with open('CounterDataset10Tokens.txt','a') as f:
#     for i in range(len(count10_seqs)):
#         f.write(count10_seqs[i]+','+count10_labels[i]+'\n')


with open('CounterDataset16Tokens.txt','a') as f:
    for i in range(len(count_upto16_seqs)):
        f.write(count_upto16_seqs[i] + ',' + count_upto16_labels[i] + '\n')

###############

dyck_seqs2_valid, dyck_seqs2_invalid = generateDataset(1,1)
dyck_seqs_2, dyck_labels_2 = oversampleMinority(dyck_seqs2_valid,'valid',dyck_seqs2_invalid,'invalid')

# with open('Dyck1Dataset2Tokens.txt','a') as f:
#     for i in range(len(dyck_seqs_2)):
#         f.write(dyck_seqs_2[i]+','+dyck_labels_2[i]+'\n')


dyck_seqs4_valid, dyck_seqs4_invalid = generateDataset(2,2)
dyck_seqs_4, dyck_labels_4 = oversampleMinority(dyck_seqs4_valid,'valid', dyck_seqs4_invalid,'invalid')

dyck_upto4_seqs = []
dyck_upto4_labels = []
for i in range(len(dyck_seqs_2)):
    dyck_upto4_seqs.append(dyck_seqs_2[i])
    dyck_upto4_labels.append(dyck_labels_2[i])

for i in range(len(dyck_seqs_4)):
    dyck_upto4_seqs.append(dyck_seqs_4[i])
    dyck_upto4_labels.append(dyck_labels_4[i])

dyck_upto4_seqs,dyck_upto4_labels = shuffle(dyck_upto4_seqs,dyck_upto4_labels)

# with open('Dyck1Dataset4Tokens.txt','a') as f:
#     for i in range(len(dyck_upto4_seqs)):
#         f.write(dyck_upto4_seqs[i]+','+dyck_upto4_labels[i]+'\n')


dyck_upto8_seqs = []
dyck_upto8_labels = []


dyck_seqs8_valid, dyck_seqs_8_invalid = generateDataset(4,4)
dyck_seqs_8, dyck_labels_8 = oversampleMinority(dyck_seqs8_valid,'valid', dyck_seqs_8_invalid,'invalid')

for i in range(len(dyck_upto4_seqs)):
    dyck_upto8_seqs.append(dyck_upto4_seqs[i])
    dyck_upto8_labels.append(dyck_upto4_labels[i])

for i in range(len(dyck_seqs_8)):
    dyck_upto8_seqs.append(dyck_seqs_8[i])
    dyck_upto8_labels.append(dyck_labels_8[i])

dyck_upto8_seqs,dyck_upto8_labels = shuffle(dyck_upto8_seqs,dyck_upto8_labels)

# with open('Dyck1Dataset8Tokens.txt','a') as f:
#     for i in range(len(dyck_upto8_seqs)):
#         f.write(dyck_upto8_seqs[i]+','+dyck_upto8_labels[i]+'\n')



dyck_seqs10_valid, dyck_seqs_10_invalid = generateDataset(5,5)
dyck_seqs_10, dyck_labels_10 = oversampleMinority(dyck_seqs10_valid,'valid', dyck_seqs_10_invalid,'invalid')


# with open('Dyck1Dataset10Tokens.txt','a') as f:
#     for i in range(len(dyck_seqs_10)):
#         f.write(dyck_seqs_10[i]+','+dyck_labels_10[i]+'\n')

#GENERATE THE LONGER SETS BY USING THE SUZGUN SEQUENCES AND THEN ADAPTING THEM TO MAKE THEM INVALID OR WHATEVER, TO MATCH THE DATASETS WE NEED
#WRITE DATASETS TO TEXT FILES AND WRITE PYTORCH DATASETS TO READ THESE txt file DATASETS


# print(generateLabelledCountDataset(10,10))
# print(generateLabelledCountDataset(15,15))


