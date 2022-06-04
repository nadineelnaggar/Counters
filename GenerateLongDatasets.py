from sklearn.utils import shuffle
import math


seqs_10 = []
seqs_20 = []
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

print(len(seqs_10))
print(len(seqs_20))
print(len(seqs_50))