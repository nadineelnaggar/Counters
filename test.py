import torch
n_letters = 2
vocab=['(', ')']
labels = ['valid', 'invalid']
from sklearn.utils import shuffle
import math
from models import TernaryLinearBracketCounter
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns



def encode_sentence(sentence):

    rep = torch.zeros(len(sentence),1,n_letters)



    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        rep[index][0][pos]=1
    rep.requires_grad_(True)
    return rep

def encode_labels(label):
    return torch.tensor([labels.index(label)], dtype=torch.float32)
print(encode_sentence('(())'))

print(encode_labels('invalid'))



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


seqsCount4TokensPos,seqsCount4TokensZeroNeg = generateCountDataset(2,2)
count4_seqs, count4_labels = oversampleMinority(seqsCount4TokensPos,'Pos',seqsCount4TokensZeroNeg,'ZeroNeg')
print(count4_seqs)
print(count4_labels)

for i in range(len(count4_seqs)):

    print(count4_seqs[i], count4_labels[i], (count4_seqs[i].count('(') - count4_seqs[i].count(')')))

seqsCount8TokensPos, seqsCount8TokensZeroNeg = generateCountDataset(4,4)
count8_seqs, count8_labels = oversampleMinority(seqsCount8TokensPos,'Pos', seqsCount8TokensZeroNeg,'ZeroNeg')

for i in range(len(count8_seqs)):

    print(count8_seqs[i], count8_labels[i], (count8_seqs[i].count('(') - count8_seqs[i].count(')')))




seqsCount2TokensPos, seqsCount2TokensZeroNeg = generateCountDataset(1,1)

count2_seqs, count2_labels = oversampleMinority(seqsCount2TokensPos,'Pos', seqsCount2TokensZeroNeg,'ZeroNeg')

for i in range(len(count2_seqs)):

    print(count2_seqs[i], count2_labels[i], (count2_seqs[i].count('(') - count2_seqs[i].count(')')))



labels=['Neg', 'Zero', 'Pos']
task='TernaryBracketCounting'
# task=''
def encode_labels(label):
    if task=='TernaryBracketCounting':
        out = torch.zeros((1,len(labels)))
        out[0][labels.index(label)]=1
        return out
    else:
        return torch.tensor([labels.index(label)], dtype=torch.float32)

print(encode_labels('Neg'))
print(encode_labels('Zero'))
print(encode_labels('Pos'))

def classFromOutput(output):

    if task=='Dyck1Classification' or task=='BracketCounting':
        if output.item() > 0.5:
            category_i = 1
        else:
            category_i = 0
    elif task == 'TernaryBracketCounting':
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        # return labels[category_i], category_i
    return labels[category_i], category_i

print(classFromOutput(torch.tensor([0.5,0.3,0.2], dtype=torch.float32)))
print(classFromOutput(torch.tensor([0.1,0.7,0.2], dtype=torch.float32)))
print(classFromOutput(torch.tensor([0.6,0.1,0.2], dtype=torch.float32)))
print(classFromOutput(torch.tensor([0.1,0.4,0.5], dtype=torch.float32)))


model = TernaryLinearBracketCounter(counter_input_size=3, counter_output_size=1, output_size=1, output_activation='Softmax', initialisation='ranodm')


a = [1,2,3,1,1,2,3,3,2]
b = [1,2,1,1,3,2,1,3,1]

conf = confusion_matrix(a,b)
print(conf)
plt.subplots()
# b, t = plt.ylim()
# b+=1
# t-=1
# plt.ylim(bottom=150, top=100)
heat=sns.heatmap(conf, annot=True, xticklabels=[1,2,3], yticklabels=[1,2,3], cmap='Blues', cbar=False)
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predicted')
plt.ylabel('actual')
plt.show()
# plot_confusion_matrix(conf)
# plt.show()
plt.close()



filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_TEST_LOG 50_TOKENS.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output function = ' in line:
            predictions.append(line[-3:-2])
        elif 'target = ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 50_TOKENS_ALL.png')
plt.show()
plt.close()

print(len(targets))
for i in range(10):
    targets_per_run=targets[150*i:((150*(i+1))-1)]
    predictions_per_run=predictions[150*i:((150*(i+1))-1)]
    print(targets_per_run)
    print(predictions_per_run)
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 50_TOKENS_RUN_'+str(i)+'.png')
    plt.close()
# filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_correct_initialisation_100epochs_TEST_LOG 20_TOKENS.txt'
# targets = []
# predictions = []

filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_TEST_LOG 20_TOKENS.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output function = ' in line:
            predictions.append(line[-3:-2])
        elif 'target = ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 20_TOKENS_ALL.png')
plt.show()
plt.close()

print(len(targets))
for i in range(10):
    targets_per_run=targets[150*i:((150*(i+1))-1)]
    predictions_per_run=predictions[150*i:((150*(i+1))-1)]
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 20_TOKENS_RUN_'+str(i)+'.png')
    plt.close()



filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_TEST_LOG 50_TOKENS.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output function = ' in line:
            predictions.append(line[-3:-2])
        elif 'target = ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 50_TOKENS_ALL.png')
plt.show()
plt.close()

print(len(targets))
for i in range(10):
    targets_per_run=targets[150*i:((150*(i+1))-1)]
    predictions_per_run=predictions[150*i:((150*(i+1))-1)]
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 50_TOKENS_RUN_'+str(i)+'.png')
    plt.close()

# filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_correct_initialisation_100epochs_TEST_LOG 20_TOKENS.txt'
# targets = []
# predictions = []

filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_TEST_LOG 20_TOKENS.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output function = ' in line:
            predictions.append(line[-3:-2])
        elif 'target = ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 20_TOKENS_ALL.png')
plt.show()
plt.close()

print(len(targets))
for i in range(10):
    targets_per_run=targets[150*i:((150*(i+1))-1)]
    predictions_per_run=predictions[150*i:((150*(i+1))-1)]
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 20_TOKENS_RUN_'+str(i)+'.png')

    plt.close()


filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_TEST_LOG 50_TOKENS.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output function = ' in line:
            predictions.append(line[-3:-2])
        elif 'target = ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 50_TOKENS_ALL.png')
plt.show()
plt.close()

print(len(targets))
for i in range(10):
    targets_per_run=targets[150*i:((150*(i+1))-1)]
    predictions_per_run=predictions[150*i:((150*(i+1))-1)]
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 50_TOKENS_RUN_'+str(i)+'.png')
    plt.close()

# filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_correct_initialisation_100epochs_TEST_LOG 20_TOKENS.txt'
# targets = []
# predictions = []

filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_TEST_LOG 20_TOKENS.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output function = ' in line:
            predictions.append(line[-3:-2])
        elif 'target = ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 20_TOKENS_ALL.png')
plt.show()
plt.close()

print(len(targets))
for i in range(10):
    targets_per_run=targets[150*i:((150*(i+1))-1)]
    predictions_per_run=predictions[150*i:((150*(i+1))-1)]
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX 20_TOKENS_RUN_'+str(i)+'.png')
    plt.close()



filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_TRAIN LOG_RAW.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output in train function =' in line:
            predictions.append(line[-3:-2])
        elif 'target in train function =' in line:
            targets.append(line[-3:-2])

print('training raw')
print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX TRAIN_ALL.png')
plt.show()
plt.close()

print(len(targets))
print(len(predictions))
for i in range(10):
    targets_per_run=targets[354*i:((354*(i+1)))]
    predictions_per_run=predictions[354*i:((354*(i+1)))]
    print(len(targets_per_run))
    print(len(predictions_per_run))
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX TRAIN_RUN_'+str(i)+'.png')
    plt.close()



filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_TRAIN LOG_RAW.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output ' in line:
            predictions.append(line[-3:-2])
        elif 'target ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX TRAIN_ALL.png')
plt.show()
plt.close()

print(len(predictions))
print(len(targets))
for i in range(10):
    targets_per_run=targets[6*i:((6*(i+1)))]
    predictions_per_run=predictions[6*i:((6*(i+1)))]
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX TRAIN_RUN_'+str(i)+'.png')
    plt.close()




filename = '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_TRAIN LOG_RAW.txt'
targets = []
predictions = []

with open(filename,'r') as f:
    for line in f:
        if 'rounded output ' in line:
            predictions.append(line[-3:-2])
        elif 'target ' in line:
            targets.append(line[-3:-2])

print(targets)
print(predictions)
conf2 = confusion_matrix(targets,predictions)
heat=sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues', cbar=False, fmt='d')
bottom1, top1 = heat.get_ylim()
heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
        # plt.show()
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.savefig('/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX TRAIN_ALL.png')
plt.show()
plt.close()

print(len(targets))
for i in range(10):
    targets_per_run=targets[28*i:((28*(i+1)))]
    predictions_per_run=predictions[28*i:((28*(i+1)))]
    conf2 = confusion_matrix(targets_per_run, predictions_per_run)
    heat = sns.heatmap(conf2, annot=True, xticklabels=['ZeroNeg', 'Pos'], yticklabels=['ZeroNeg', 'Pos'], cmap='Blues',
                       cbar=False, fmt='d')
    bottom1, top1 = heat.get_ylim()
    heat.set_ylim(bottom1 + 0.5, top1 - 0.5)
    # plt.show()
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.savefig(
        '/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/BracketCounting/Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_CONFUSION_MATRIX TRAIN_RUN_'+str(i)+'.png')
    plt.close()
