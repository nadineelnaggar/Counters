from sklearn.utils import shuffle
import math
import random
from random import randint

seqs_10 = []
seqs_20 = []
seqs_30 = []
seqs_40 = []
seqs_50 = []

with open('Dyck1_Dataset_Suzgun_train_.txt', 'r') as f:
    for line in f:
        line = line.split(",")
        sentence = line[0].strip()
        label = line[1].strip()
        if len(sentence)==10:
            seqs_10.append(sentence)
        elif len(sentence)==20:
            seqs_20.append(sentence)
        elif len(sentence)==50:
            seqs_50.append(sentence)
        elif len(sentence)==30:
            seqs_30.append(sentence)
        elif len(sentence)==40:
            seqs_40.append(sentence)

print(len(seqs_10))
print(len(seqs_20))
print(len(seqs_30))
print(len(seqs_40))
print(len(seqs_50))


def make_incomplete(seqs):
    new_seqs = []

    for i in range(len(seqs)):
        close_bracket_indices = [pos for pos, char in enumerate(seqs[i]) if char == ')']
        num_close_brackets = len(close_bracket_indices)
        num_changed_brackets = randint(1,num_close_brackets)
        changed_bracket_indices = []
        for j in range(num_changed_brackets):
            idx = randint(0,num_close_brackets-1)
            if close_bracket_indices[idx] not in changed_bracket_indices:
                changed_bracket_indices.append(close_bracket_indices[idx])


        seq = seqs[i]
        print('seq before changing = ',seq)
        print('change bracket indices = ',changed_bracket_indices)
        for j in range(len(changed_bracket_indices)):
            idx = changed_bracket_indices[j]
            seq = seq[:idx]+'('+seq[idx+1:]
        print('seq after changing = ',seq)
        new_seqs.append(seq)
        # print(num_close_brackets)
        # print(num_changed_brackets)
        print('num open brackets = ', seq.count('('))
        print('num close brackets = ', seq.count(')'))
    return new_seqs





seq_test = ['((()))', '()()(())', '(()()(()))']
print(make_incomplete(seq_test))



def make_invalid_excess_close(seqs):
    new_seqs = []

    for i in range(len(seqs)):
        open_bracket_indices = [pos for pos, char in enumerate(seqs[i]) if char == '(']
        num_open_brackets = len(open_bracket_indices)
        num_changed_brackets = randint(1,num_open_brackets)
        changed_bracket_indices = []
        for j in range(num_changed_brackets):
            idx = randint(0,num_open_brackets-1)
            if open_bracket_indices[idx] not in changed_bracket_indices:
                changed_bracket_indices.append(open_bracket_indices[idx])

        changed_bracket_indices.sort()

        seq = seqs[i]
        print('seq before changing = ',seq)
        print('change bracket indices = ',changed_bracket_indices)
        for j in range(len(changed_bracket_indices)):
            idx = changed_bracket_indices[j]
            seq = seq[:idx]+')'+seq[idx+1:]
        print('seq after changing = ',seq)
        new_seqs.append(seq)
        # print('num open brackets = ',num_open_brackets)
        # print('num close brackets = ',num_changed_brackets)
        print('num open brackets = ', seq.count('('))
        print('num close brackets = ', seq.count(')'))
    return new_seqs

print('***************************')
seq_test = ['((()))', '()()(())', '(()()(()))']
print(make_invalid_excess_close(seq_test))



def get_timestep_depths(x):
    max_depth=0
    current_depth=0
    timestep_depths = []
    for i in range(len(x)):

        if x[i] == '(':
            current_depth += 1
            timestep_depths.append(current_depth)
            if current_depth > max_depth:
                max_depth = current_depth
        elif x[i] == ')':
            current_depth -= 1
            timestep_depths.append(current_depth)
    return timestep_depths

def make_invalid_wrong_order(seqs):
    new_seqs = []

    for i in range(len(seqs)):
        seq = seqs[i]
        timestep_depths = get_timestep_depths(seq)
        zero_depth_indices = []
        # zero_depth_indices.append(-1)
        for j in range(len(timestep_depths)):
            if timestep_depths[j]==0:
                zero_depth_indices.append(j)
        print('timestep depths = ',timestep_depths)
        print('zero depth indices = ',zero_depth_indices)

        num_zeros = len(zero_depth_indices)

        num_changed_brackets = randint(1,num_zeros)
        changed_indices = []
        for j in range(num_changed_brackets):
            idx = randint(0,num_zeros-1)
            if zero_depth_indices[idx] not in changed_indices:
                changed_indices.append(zero_depth_indices[idx])



        changed_indices.sort()

        # seq = seqs[i]
        print('seq before changing = ',seq)
        print('change bracket indices = ',changed_indices)
        # print('change close bracket indices = ', changed_close_bracket_indices)
        for j in range(len(changed_indices)):
            idx = changed_indices[j]
            if idx!=(len(seq)-1):
                print('seq[:idx+1] = ',seq[:idx+1])
                print('seq[idx+3:]',seq[idx+3:])
                # seq = seq[:idx+1]+')('+seq[idx+3:]
                seq = seq[:idx + 1] + ')' + seq[idx + 1:len(seq) - 1]
                print('num opening brackets = ', seq.count('('))
                print('num closing brackets = ', seq.count(')'))
                # print('seq after changing first = ',seq)
            elif idx==len(seq)-1:
                # seq = ')'+seq[:len(seq)-1]
                seq = seq[1:] + '('
            # elif idx == -1:
            #     seq = ')' + seq[:len(seq) - 1]
        print('seq after changing = ',seq)
        print('num opening brackets = ',seq.count('('))
        print('num closing brackets = ',seq.count(')'))
        new_seqs.append(seq)
        # print(num_open_brackets)
        print(num_changed_brackets)
    return new_seqs

print('***************************')
seq_test = ['((()))', '()()(())', '(()()(()))']
print(make_invalid_wrong_order(seq_test))


"""
150 seqs total for dyck 1 classification task

75 valid
25 invalid excess
25 invalid wrong order
25 incomplete

invalid excess, invalid wrong order, and incomplete will all be labelled as invalid
"""

dyck_seqs_20_valid = seqs_20[:75]
dyck_seqs_20_invalid_wrong_order = seqs_20[75:100]
dyck_seqs_20_invalid_excess_close = seqs_20[100:125]
dyck_seqs_20_incomplete = seqs_20[125:150]

print('///////////////////////////////////////')

dyck_seqs_20_invalid_wrong_order=make_invalid_wrong_order(dyck_seqs_20_invalid_wrong_order)
for seq in dyck_seqs_20_invalid_wrong_order:
    ts_depths = get_timestep_depths(seq)
    neg_flag = False
    if len(seq)!=20:
        print('len seq != 20, len(seq) = ',len(seq),', ',seq.count('('),'opening, ',seq.count(')'), 'closing')
        # print(len(seq))
    else:
        for depth in ts_depths:
            if depth<0:
                neg_flag=True
        print('len(seq) = 20 ',', ',seq.count('('),'opening, ',seq.count(')'), 'closing, Neg Flag = ',neg_flag)

print('````````````````````````````````````````````````')

dyck_seqs_20_invalid_excess_close=make_invalid_excess_close(dyck_seqs_20_invalid_excess_close)
for seq in dyck_seqs_20_invalid_excess_close:
    ts_depths = get_timestep_depths(seq)
    neg_flag = False
    if len(seq)!=20:
        print('len seq != 20, len(seq) = ',len(seq),', ',seq.count('('),'opening, ',seq.count(')'), 'closing')
        # print(len(seq))
    else:
        for depth in ts_depths:
            if depth<0:
                neg_flag=True
        print('len(seq) = 20 ',', ',seq.count('('),'opening, ',seq.count(')'), 'closing, Neg Flag = ',neg_flag)

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
dyck_seqs_20_incomplete=make_incomplete(dyck_seqs_20_incomplete)
for seq in dyck_seqs_20_incomplete:
    ts_depths = get_timestep_depths(seq)
    neg_flag = False
    if len(seq)!=20:
        print('len seq != 20, len(seq) = ',len(seq),', ',seq.count('('),'opening, ',seq.count(')'), 'closing')
        # print(len(seq))
    else:
        for depth in ts_depths:
            if depth<0:
                neg_flag=True
        print('len(seq) = 20 ',', ',seq.count('('),'opening, ',seq.count(')'), 'closing, Neg Flag = ',neg_flag)

print(len(dyck_seqs_20_valid))
print(len(dyck_seqs_20_incomplete))
print(len(dyck_seqs_20_invalid_excess_close))
print(len(dyck_seqs_20_invalid_wrong_order))

dyck_seqs_30_valid = seqs_30[:75]
dyck_seqs_30_invalid_wrong_order = seqs_30[75:100]
dyck_seqs_30_invalid_excess_close = seqs_30[100:125]
dyck_seqs_30_incomplete = seqs_30[125:150]



dyck_seqs_40_valid = seqs_40[:75]
dyck_seqs_40_invalid_wrong_order = seqs_40[75:100]
dyck_seqs_40_invalid_excess_close = seqs_40[100:125]
dyck_seqs_40_incomplete = seqs_40[125:150]



dyck_seqs_50_valid = seqs_50[:75]
dyck_seqs_50_invalid_wrong_order = seqs_50[75:100]
dyck_seqs_50_invalid_excess_close = seqs_50[100:125]
dyck_seqs_50_incomplete = seqs_50[125:150]

###################################

"""
150 seqs total for the counting task

75 incomplete
25 valid
25 invalid excess close
25 invalid wrong order

incomplete will be labelled final count positive
valid, invalid excess close and invalid wrong order will be labelled final count zero or negative

"""

count_seqs_20_valid = []
count_seqs_20_invalid_wrong_order = []
count_seqs_20_invalid_excess_close = []
count_seqs_20_incomplete = []


count_seqs_30_valid = []
count_seqs_30_invalid_wrong_order = []
count_seqs_30_invalid_excess_close = []
count_seqs_30_incomplete = []



count_seqs_40_valid = []
count_seqs_40_invalid_wrong_order = []
count_seqs_40_invalid_excess_close = []
count_seqs_40_incomplete = []



count_seqs_50_valid = []
count_seqs_50_invalid_wrong_order = []
count_seqs_50_invalid_excess_close = []
count_seqs_50_incomplete = []



#
# def make_incomplete(indices, seqs, labels):
#     """
#     input a valid sequence and distort it to make it potentially valid by
#         - replacing one or more closing brackets in a random location with an opening bracket
#     :return:
#     """
#
#     for i in range(len(indices)):
#         # seq = seqs[indices[i]]
#         seq = seqs[i]
#         count_zeros = 0
#         indices_zeros = []
#         label = labels[i]
#
#         for j, char in enumerate(label):
#             if char == '0':
#                 count_zeros += 1
#                 indices_zeros.append(j)
#
#         print('length of original seq = ', len(seq))
#         print('initial number of opening brackets = ', seq.count('('))
#         print('initial number of closing brackets = ', seq.count(')'))
#         print('count_zeros for seq ', seq, ' = ', count_zeros)
#         print('indices_zeros for seq ', seq, ' = ', indices_zeros)
#         rand_idx = randint(0, len(indices_zeros))
#         print('rand_idx for seq', seq, ' = ', rand_idx)
#         if rand_idx == len(indices_zeros):
#             changed_idx = 0
#         elif rand_idx < len(indices_zeros):
#             changed_idx = indices_zeros[rand_idx]
#         print('changed idx for seq ', seq, ' = ', changed_idx)
#
#         if changed_idx == 0:
#             seq = '(' + seq[0:len(seq) - 1]
#         elif changed_idx > 0:
#             seq = seq[:changed_idx] + '(' + seq[changed_idx + 1:]
#
#         print('changed seq = ', seq)
#         print('length of changed sequence = ', len(seq))
#         print('final number of opening brackets = ', seq.count('('))
#         print('final number of closing brackets = ', seq.count(')'))
#         print('incomplete')
#         print('***************************')
#         # seqs[indices[i]] = seq
#         seqs[i] = seq
#
#
#     return seqs
