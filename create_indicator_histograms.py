import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import argparse
from models import NonZeroReLUCounter,LinearBracketCounter, TernaryLinearBracketCounter
import torch.optim as optim
import pandas as pd
import time
import math
from os.path import exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str)
parser.add_argument('--task',type=str,default='Dyck1Classification', help='Dyck1Classification or BracketCounting')
parser.add_argument('--train_seq_length',type=int,default=4, help='length of sequences in training set: 4, 8, 16')
# parser.add_argument('--long_seq_length',type=int,default=30,help='length of sequences in long test set (20,30,40,50 tokens)')
parser.add_argument('--runtime',type=str,default='colab',help='colab or local')
parser.add_argument('--num_epochs',type=int)
parser.add_argument('--output_activation',type=str,default='Sigmoid',help='Sigmoid or Clipping')
parser.add_argument('--initialisation',type=str,default='random',help='random or correct')
parser.add_argument('--oversampling',type=str,default='OversampledDataset', help='OversampledDataset or NonOversampledDataset')


args = parser.parse_args()


model_name = args.model_name
train_seq_length = args.train_seq_length
# long_test_length = args.long_test_length
runtime = args.runtime
num_epochs = args.num_epochs
output_activation = args.output_activation
task = args.task
initialisation=args.initialisation
oversampling = args.oversampling

# num_epochs = 50
num_runs = 10
learning_rate = 0.005
checkpoint_step=1

use_optimiser='Adam'

if task=='Dyck1Classification':
    labels = ['valid','invalid']
elif task=='BracketCounting':
    labels = ['ZeroNeg', 'Pos']



if runtime=='local':
    path = "/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/"+str(task)+"/"
elif runtime=='colab':
    path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Counters/"+str(task)+"/"

prefix = path+output_activation+"_activation_"+str(train_seq_length)+'train_seq_length_'+initialisation+"_initialisation_"+str(num_epochs)+"epochs"+"_"+oversampling

# file_name = prefix+'.txt'
# train_log = prefix+'_TRAIN LOG.txt'
# train_log_raw = prefix+'_TRAIN LOG_RAW.txt'
# test_20_log = prefix+'_TEST_LOG 20_TOKENS.txt'
# test_30_log = prefix+'_TEST_LOG 30_TOKENS.txt'
# test_40_log = prefix+'_TEST_LOG 40_TOKENS.txt'
# test_50_log = prefix+'_TEST_LOG 50_TOKENS.txt'
read_excel_name = prefix+'.xlsx'
modelname = prefix+"_MODEL_"
# optimname = prefix+"_OPTIMISER_"
checkpoint = prefix+'_CHECKPOINT_'
# plt_name = prefix+'_PLOT_'
excel_name_indicators = path+''+model_name+'_INDICATORS.xlsx'

input_size = 2

if model_name=='TernaryLinearBracketCounter':
    output_size = 3
else:
    output_size = 1
hidden_size = 2
counter_input_size = 3
counter_output_size = 1
vocab = ['(', ')']


def select_model():
    if task=='Dyck1Classification':
        model = NonZeroReLUCounter(counter_input_size=counter_input_size,counter_output_size=counter_output_size,output_size=output_size, initialisation=initialisation, output_activation=output_activation)
    elif task=='BracketCounting':
        model = LinearBracketCounter(counter_input_size=counter_input_size,counter_output_size=counter_output_size,output_size=output_size, initialisation=initialisation, output_activation=output_activation)
    elif task=='TernaryBracketCounting':
        model = TernaryLinearBracketCounter(counter_input_size=counter_input_size,counter_output_size=counter_output_size,output_size=output_size, initialisation=initialisation, output_activation=output_activation)

    return model.to(device)

a_b_ratio = []
recurrent_weights = []

def read_sheets():
    sheet_names = []
    for i in range(num_runs):
        sheet_name = "run"+str(i)
        sheet_names.append(sheet_name)
    df = pd.read_excel(read_excel_name,sheet_name=sheet_names)
    dfs = []
    for i in range(num_runs):
        dfs.append(df.get(sheet_names[i]))
    return dfs

dfs=read_sheets()

for run in range(num_runs):
    # for epoch in range(num_epochs):
        # if epoch%checkpoint_step==0:
        #     checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"
    model = select_model()
    mdl = modelname + 'run' + str(run) + '.pth'

    model.load_state_dict(torch.load(mdl))

    model.to(device)
    # print(model.parameters)
    print('*********************************')
    print(model.counter.weight)
    print(model.out.weight)
    print('if input = (, sum = ',model.counter.weight[0][0].item())
    print('if input = ), sum = ',model.counter.weight[0][1].item())
    print('recurrent weight = ',model.counter.weight[0][2].item())
    print('ratio of a and b = ',model.counter.weight[0][0].item()/model.counter.weight[0][1].item())
    a_b_ratio.append(model.counter.weight[0][0].item()/model.counter.weight[0][1].item())
    recurrent_weights.append(model.counter.weight[0][2].item())




plt.subplots()
plt.hist(a_b_ratio)
plt.savefig(prefix+'_PLOT_INDICATOR_AB_RATIO.png')
plt.show()

plt.hist(recurrent_weights)
plt.savefig(prefix+'_PLOT_INDICATOR_RECURRENT_WEIGHT.png')
plt.show()


# writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
#
#     df1.to_excel(writer, index=False)
#     writer.save()