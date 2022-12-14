import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import argparse
from models import NonZeroReLUCounter,LinearBracketCounter, TernaryLinearBracketCounter, TernaryRegressionLinearBracketCounter
import torch.optim as optim
import pandas as pd
import time
import math
from sklearn.metrics import confusion_matrix



seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str)
parser.add_argument('--task',type=str,default='Dyck1Classification', help='Dyck1Classification or BracketCounting or TernaryBracketCounting')
parser.add_argument('--train_seq_length',type=int,default=4, help='length of sequences in training set: 4, 8, 16')
# parser.add_argument('--long_seq_length',type=int,default=30,help='length of sequences in long test set (20,30,40,50 tokens)')
parser.add_argument('--runtime',type=str,default='colab',help='colab or local')
parser.add_argument('--num_epochs',type=int)
parser.add_argument('--output_activation',type=str,default='Sigmoid',help='Sigmoid or Clipping or Softmax')
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
elif task=='TernaryBracketCounting':
    labels=['Neg', 'Zero', 'Pos']

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




if runtime=='local':
    path = "/Users/nadineelnaggar/Google Drive/PhD/EXPT_LOGS/Counters/"+str(task)+"/"
elif runtime=='colab':
    path = "/content/drive/MyDrive/PhD/EXPT_LOGS/Counters/"+str(task)+"/"


prefix = path+output_activation+"_activation_"+str(train_seq_length)+'train_seq_length_'+initialisation+"_initialisation_"+str(num_epochs)+"epochs"+"_"+oversampling

file_name = prefix+'.txt'
train_log = prefix+'_TRAIN LOG.txt'
train_log_raw = prefix+'_TRAIN LOG_RAW.txt'
test_20_log = prefix+'_TEST_LOG 20_TOKENS.txt'
test_30_log = prefix+'_TEST_LOG 30_TOKENS.txt'
test_40_log = prefix+'_TEST_LOG 40_TOKENS.txt'
test_50_log = prefix+'_TEST_LOG 50_TOKENS.txt'
excel_name = prefix+'.xlsx'
modelname = prefix+"_MODEL_"
optimname = prefix+"_OPTIMISER_"
checkpoint = prefix+'_CHECKPOINT_'
plt_name = prefix+'_PLOT_'



input_size = 2
output_size = 1
hidden_size = 2
counter_input_size = 3
counter_output_size = 1
vocab = ['(', ')']
if task=='TernaryBracketCounting':
        output_size=3




with open(file_name,'w') as f:
    f.write('')

with open(train_log,'w') as f:
    f.write('')

with open(train_log_raw,'w') as f:
    f.write('')

with open(test_20_log,'w') as f:
    f.write('')

with open(test_30_log,'w') as f:
    f.write('')

with open(test_40_log,'w') as f:
    f.write('')

with open(test_50_log,'w') as f:
    f.write('')

num_classes = 2
n_letters = 2

def read_datasets():
    x = []
    y = []
    if train_seq_length==2:
        if model_name=='NonZeroReLUCounter':
            read_file = 'Dyck1Dataset2Tokens.txt'
        elif model_name=='LinearBracketCounter' and oversampling=='OversampledDataset':
            read_file = 'CounterDataset2Tokens.txt'
        elif model_name=='LinearBracketCounter' and oversampling=='NonOversampledDataset':
            read_file = 'CounterDataset2TokensNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='NonOversampledDataset':
            read_file = 'CounterDataset2TokensTernaryNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='OversampledDataset':
            read_file = 'CounterDataset2TokensTernary.txt'
    elif train_seq_length==4:
        if model_name=='NonZeroReLUCounter':
            read_file = 'Dyck1Dataset4Tokens.txt'
        elif model_name=='LinearBracketCounter' and oversampling=='OversampledDataset':
            read_file = 'CounterDataset4Tokens.txt'
        elif model_name=='LinearBracketCounter' and oversampling=='NonOversampledDataset':
            read_file = 'CounterDataset4TokensNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='NonOversampledDataset':
            read_file = 'CounterDataset4TokensTernaryNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='OversampledDataset':
            read_file = 'CounterDataset4TokensTernary.txt'
    elif train_seq_length==8:
        if model_name=='NonZeroReLUCounter':
            read_file = 'Dyck1Dataset8Tokens.txt'
        elif model_name=='LinearBracketCounter' and oversampling=='OversampledDataset':
            read_file = 'CounterDataset8Tokens.txt'
        elif model_name=='LinearBracketCounter' and oversampling=='NonOversampledDataset':
            read_file = 'CounterDataset8TokensNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='NonOversampledDataset':
            read_file = 'CounterDataset8TokensTernaryNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='OversampledDataset':
            read_file = 'CounterDataset8TokensTernary.txt'
    elif train_seq_length == 16:
        # if model_name == 'NonZeroReLUCounter':
        #     read_file = 'Dyck1Dataset8Tokens.txt'
        # elif model_name == 'LinearBracketCounter' and oversampling == 'OversampledDataset':
        if model_name == 'LinearBracketCounter' and oversampling == 'OversampledDataset':
            read_file = 'CounterDataset16Tokens.txt'
        elif model_name == 'LinearBracketCounter' and oversampling == 'NonOversampledDataset':
            read_file = 'CounterDataset16TokensNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='NonOversampledDataset':
            read_file = 'CounterDataset16TokensTernaryNoOversampling.txt'
        elif model_name=='TernaryLinearBracketCounter' and oversampling=='OversampledDataset':
            read_file = 'CounterDataset16TokensTernary.txt'

    with open(read_file, 'r') as f:
        for line in f:
            line = line.split(",")
            sentence = line[0].strip()
            label = line[1].strip()
            x.append(sentence)
            y.append(label)

    x_20 = []
    y_20 = []
    x_30 = []
    y_30 = []
    x_40 = []
    y_40 = []
    x_50 = []
    y_50 = []

    if model_name=='NonZeroReLUCounter':
        with open('Dyck1Dataset20Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_20.append(sentence)
                y_20.append(label)
        with open('Dyck1Dataset30Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_30.append(sentence)
                y_30.append(label)

        with open('Dyck1Dataset40Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_40.append(sentence)
                y_40.append(label)

        with open('Dyck1Dataset50Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_50.append(sentence)
                y_50.append(label)

    elif model_name == 'LinearBracketCounter':
        with open('CounterDataset20Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_20.append(sentence)
                y_20.append(label)
        with open('CounterDataset30Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_30.append(sentence)
                y_30.append(label)

        with open('CounterDataset40Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_40.append(sentence)
                y_40.append(label)

        with open('CounterDataset50Tokens.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_50.append(sentence)
                y_50.append(label)
    elif model_name == 'TernaryLinearBracketCounter':
        with open('CounterDataset20TokensTernary.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_20.append(sentence)
                y_20.append(label)
        with open('CounterDataset30TokensTernary.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_30.append(sentence)
                y_30.append(label)

        with open('CounterDataset40TokensTernary.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_40.append(sentence)
                y_40.append(label)

        with open('CounterDataset50TokensTernary.txt', 'r') as f:
            for line in f:
                line = line.split(",")
                sentence = line[0].strip()
                label = line[1].strip()
                x_50.append(sentence)
                y_50.append(label)


    return x, y, x_20, y_20, x_30, y_30, x_40, y_40, x_50, y_50

#
# def encode_sentence(sentence):
#     rep = torch.zeros(len(sentence),1,input_size)
#     for index, char in enumerate(sentence):
#         pos = vocab.index(char)
#         if pos == 0:
#             rep[index][0] = 1
#         elif pos == 1:
#             rep[index][0] = -1
#     rep.requires_grad_(True)
#     return rep

def encode_sentence(sentence):

    rep = torch.zeros(len(sentence),1,n_letters)



    for index, char in enumerate(sentence):
        pos = vocab.index(char)
        rep[index][0][pos]=1
    rep.requires_grad_(True)
    return rep

# def encode_labels(label):
#     # return torch.tensor(labels.index(label), dtype=torch.float32)
#     if label=='valid' or label=='ZeroNeg':
#         return torch.tensor(0,dtype=torch.float32)
#     elif label =='invalid' or label=='Pos':
#         return torch.tensor(1,dtype=torch.float32)

def encode_labels(label):
    if task=='TernaryBracketCounting':
        outt = torch.zeros((len(labels)))
        outt[labels.index(label)]=1
        return outt
    else:
        return torch.tensor([labels.index(label)], dtype=torch.float32)

def encode_dataset(sentences, labels):
    encoded_sentences = []
    encoded_labels = []
    for sentence in sentences:
        encoded_sentences.append(encode_sentence(sentence))
    for label in labels:
        encoded_labels.append(encode_labels(label))
    return encoded_sentences, encoded_labels


X_train, y_train, X_20, y_20, X_30, y_30, X_40, y_40, X_50, y_50 = read_datasets()

X_train_notencoded = X_train
y_train_notencoded = y_train
X_train, y_train = encode_dataset(X_train, y_train)

X_20_notencoded = X_20
y_20_notencoded = y_20
X_20, y_20 = encode_dataset(X_20, y_20)

X_30_notencoded = X_30
y_30_notencoded = y_30
X_30, y_30 = encode_dataset(X_30, y_30)

X_40_notencoded = X_40
y_40_notencoded = y_40
X_40, y_40 = encode_dataset(X_40, y_40)

X_50_notencoded = X_50
y_50_notencoded = y_50
X_50, y_50 = encode_dataset(X_50, y_50)


def select_model():
    if task=='Dyck1Classification':
        model = NonZeroReLUCounter(counter_input_size=counter_input_size,counter_output_size=counter_output_size,output_size=output_size, initialisation=initialisation, output_activation=output_activation)
    elif task=='BracketCounting':
        model = LinearBracketCounter(counter_input_size=counter_input_size,counter_output_size=counter_output_size,output_size=output_size, initialisation=initialisation, output_activation=output_activation)
    elif task == 'TernaryBracketCounting':
        model = TernaryLinearBracketCounter(counter_input_size=counter_input_size, counter_output_size=counter_output_size,
                                     output_size=output_size, initialisation=initialisation,
                                     output_activation=output_activation)

    return model.to(device)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    # return '%dm %ds' % (m, s)
    return m, s


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    # return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    return asMinutes(s), asMinutes(rs)


def train(model, X, X_notencoded, y, y_notencoded, run=0):

    start = time.time()

    # criterion = nn.MSELoss()
    if output_activation=='Sigmoid':
        criterion = nn.BCELoss()
    elif output_activation=='Clipping':
        criterion=nn.MSELoss()
    elif output_activation=='Softmax':
        criterion=nn.CrossEntropyLoss()
    # learning_rate = args.learning_rate
    # optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    optimiser.zero_grad()
    losses = []
    correct_arr = []
    accuracies = []
    epochs = []
    all_epoch_incorrect_guesses = []
    df1 = pd.DataFrame()
    print_flag = False
    num_samples = len(X)
    confusion_matrices = []

    train_val_accuracies = []
    train_val_losses = []


    print(model)
    num_timesteps = 0

    for elem in X:
        num_timesteps+=len(elem)
    print('num_timesteps = ',num_timesteps)



    for epoch in range(num_epochs):
        model.train()
        num_correct = 0
        confusion = torch.zeros(num_classes, num_classes)
        predicted_classes = []
        expected_classes = []
        num_correct_timesteps = 0
        total_loss = 0
        epoch_incorrect_guesses = []
        epoch_correct_guesses = []
        epochs.append(epoch)
        if epoch==num_epochs-1:
            print_flag=True
        if print_flag == True:
            with open(train_log_raw, 'a') as f:
                f.write('\nEPOCH ' + str(epoch) + '\n')
        for i in range(len(X)):
            model.zero_grad()
            # input_seq = Dyck.lineToTensor(X[i])
            # target_seq = Dyck.lineToTensorSigmoid(y[i])
            input_seq = X[i]
            target_seq = y[i]
            len_seq = len(input_seq)

            # output_seq = torch.zeros(target_seq.shape)

            input_seq.to(device)
            target_seq.to(device)
            # output_seq.to(device)

            input_tensor = X[i]
            class_tensor = y[i]
            input_sentence = X_notencoded[i]
            class_category = y_notencoded[i]

            # hidden = model.init_hidden()
            if model.model_name=='LinearBracketCounter' or model.model_name=='TernaryLinearBracketCounter':
                previous_state = torch.tensor([0],dtype=torch.float32)
            elif model.model_name=='NonZeroReLUCounter':
                opening_brackets = torch.tensor([0],dtype=torch.float32)
                closing_brackets = torch.tensor([0],dtype=torch.float32)
                excess_closing_brackets = torch.tensor([0],dtype=torch.float32)


            for j in range(len_seq):
                if model.model_name=='LinearBracketCounter':
                    out, previous_state = model(X[i][j].squeeze().to(device), previous_state)
                elif model.model_name=='NonZeroReLUCounter':
                    out, opening_brackets,closing_brackets,excess_closing_brackets = model(X[i][j].squeeze().to(device),opening_brackets,closing_brackets,excess_closing_brackets)
                elif model.model_name=='TernaryLinearBracketCounter':
                    out, previous_state = model(X[i][j].squeeze().to(device), previous_state)

                # output_seq[j]=out

            if print_flag == True:
                with open(train_log_raw, 'a') as f:
                    f.write('////////////////////////////////////////\n')
                    f.write('input sentence = ' + str(X[i]) + '\n')
                    f.write('encoded sentence = '+str(input_seq)+'\n')

            loss = criterion(out, target_seq)
            total_loss += loss.item()
            loss.backward()
            optimiser.step()

            if print_flag == True:
                with open(train_log_raw, 'a') as f:
                    f.write('actual output in train function = ' + str(out) + '\n')

            out_np = np.int_(out.detach().cpu().numpy() > 0.5)
            target_np = np.int_(target_seq.detach().cpu().numpy())

            guess, guess_i = classFromOutput(out)
            class_i = labels.index(class_category)
            confusion[class_i][guess_i] += 1
            # current_loss += loss
            expected_classes.append(class_i)
            predicted_classes.append(guess_i)
            if guess == class_category:
                num_correct += 1
                epoch_correct_guesses.append(X[i])
                if print_flag == True:
                    with open(train_log_raw, 'a') as f:
                        f.write('CORRECT' + '\n')
            else:
                epoch_incorrect_guesses.append(input_sentence)
                if print_flag == True:
                    with open(train_log_raw, 'a') as f:
                        f.write('INCORRECT' + '\n')

            if print_flag == True:
                with open(train_log_raw, 'a') as f:
                    f.write('rounded output in train function = ' + str(out_np) + '\n')
                    f.write('target in train function = ' + str(target_np) + '\n')

            #
            #
            # if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
            #     num_correct += 1
            #     # correct_arr.append(X[i])
            #     epoch_correct_guesses.append(X[i])
            #     if print_flag == True:
            #         with open(train_log, 'a') as f:
            #             f.write('CORRECT' + '\n')
            # else:
            #     epoch_incorrect_guesses.append(X[i])
            #     if print_flag == True:
            #         with open(train_log, 'a') as f:
            #             f.write('INCORRECT' + '\n')



        accuracy = num_correct/len(X)*100
        # print('Accuracy for epoch ', epoch, '=', accuracy, '%')
        time_mins, time_secs = timeSince(start, epoch + 1 / num_epochs * 100)
        losses.append(total_loss/len(X))

        # train_val_acc, train_val_loss = validate(model, X, X_notencoded, y, y_notencoded, criterion)
        # train_val_accuracies.append(train_val_acc)
        # train_val_losses.append(train_val_loss)

        with open(train_log, 'a') as f:
            f.write('Accuracy for epoch ' + str(epoch) + '=' + str(round(accuracy, 2)) + '%, avg train loss = ' +
                    str(total_loss / len(X)) +
                    ', num_correct = ' + str(num_correct) + ', time = ' + str(
                time_mins[0]) + 'm ' + str(round(time_mins[1], 2)) + 's \n')

        print('Accuracy for epoch ', epoch, '=', round(accuracy, 2), '%, avg train loss = ',
              total_loss / len(X),
              ', num_correct = ', num_correct, ', time = ', time_mins[0], 'm ', round(time_mins[1], 2), 's')

        accuracies.append(accuracy)

        all_epoch_incorrect_guesses.append(epoch_incorrect_guesses)
        correct_arr.append(epoch_correct_guesses)
        conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)
        confusion_matrices.append(conf_matrix)



        if epoch == num_epochs - 1:
            # print('\n////////////////////////////////////////////////////////////////////////////////////////\n')
            print('num_correct = ',num_correct)
            print('Final training accuracy = ', num_correct / len(X) * 100, '%')
            # print('**************************************************************************\n')



        if epoch%checkpoint_step==0:
            checkpoint_path = checkpoint+'run'+str(run)+"_epoch"+str(epoch)+".pth"
            torch.save({'run':run,
                        'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimiser_state_dict':optimiser.state_dict(),
                        'loss':loss},checkpoint_path)
            checkpoint_loss_plot = modelname+'run'+str(run)+'_epoch'+str(epoch)+'_losses.png'
            checkpoint_accuracy_plot = modelname+'run'+str(run)+'_epoch'+str(epoch)+'_accuracies.png'
            checkpoint_lr_plot = modelname+'run'+str(run)+'_epoch'+str(epoch)+'_lrs.png'
            fig_loss, ax_loss = plt.subplots()
            plt.plot(epochs,losses, label='avg train loss')
            # plt.plot(epochs,validation_losses, label='avg validation loss')
            # plt.plot(long_validation_losses, label='avg long validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(checkpoint_loss_plot)
            plt.close()
            fig_acc, ax_acc = plt.subplots()
            plt.plot(epochs, accuracies, label='train accuracies')
            # plt.plot(epochs, validation_accuracies,label='validation accuracies')
            # plt.plot(epochs, long_validation_accuracies,label='long validation accuracies')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(checkpoint_accuracy_plot)
            plt.close()
            # fig_lr, ax_lr = plt.subplots()
            # plt.plot(epochs, lrs, label='learning rate')
            # plt.xlabel('Epoch')
            # plt.ylabel('Learning rate')
            # plt.legend()
            # plt.savefig(checkpoint_lr_plot)
            # plt.close()

    # plt.subplots()
    # plt.plot(epochs,losses,label='Average Training Losses')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig(plt_name+'')
    # print('len epochs = ',len(epochs))
    # print('len losses = ',len(losses))


    df1['epoch'] = epochs
    df1['Training accuracies'] = accuracies
    df1['Average training losses'] = losses
    # df1['Average training accuracies (after training)'] = train_val_accuracies
    # df1['Average training losses (after training)'] = train_val_losses
    # df1['Average validation losses'] = validation_losses
    # df1['Validation accuracies'] = validation_accuracies
    # df1['Average long validation losses'] = long_validation_losses
    # df1['Long validation accuracies'] = long_validation_accuracies
    # df1['learning rates'] = lrs
    df1['epoch correct guesses'] = correct_arr
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses
    # df1['epoch error indices'] = error_indices
    # df1['epoch error seq lengths'] = error_seq_lengths

    df1['epoch'] = epochs
    df1['accuracies'] = accuracies
    df1['Total epoch losses'] = losses
    df1['epoch correct guesses'] = correct_arr
    df1['epoch incorrect guesses'] = all_epoch_incorrect_guesses
    df1['confusion matrices'] = confusion_matrices

    optm = optimname + 'run' + str(run) + '.pth'
    mdl = modelname + 'run' + str(run) + '.pth'
    torch.save(model.state_dict(), mdl)
    torch.save(optimiser.state_dict(), optm)

        # print(accuracies)
        # print(accuracy)
    return accuracy, df1

def validate(model, X, X_notencoded, y, y_notencoded, criterion):
    model.eval()

    # if dataset=='test_20':
    #     log_file = test_20_log
    # elif dataset=='test_30':
    #     log_file=test_30_log
    # elif dataset=='test_40':
    #     log_file = test_40_log
    # elif dataset=='test_50':
    #     log_file=test_50_log


    num_samples = len(X)


    # print(model)

    num_correct = 0
    confusion = torch.zeros(num_classes, num_classes)
    predicted_classes = []
    expected_classes = []
    total_loss = 0


    for i in range(len(X)):
        # model.zero_grad()
        # input_seq = Dyck.lineToTensor(X[i])
        # target_seq = Dyck.lineToTensorSigmoid(y[i])
        input_seq = X[i]
        target_seq = y[i]
        len_seq = len(input_seq)

        # output_seq = torch.zeros(target_seq.shape)

        input_seq.to(device)
        target_seq.to(device)
        # output_seq.to(device)

        input_tensor = X[i]
        class_tensor = y[i]
        input_sentence = X_notencoded[i]
        class_category = y_notencoded[i]

        # hidden = model.init_hidden()
        if model.model_name == 'LinearBracketCounter':
            previous_state = torch.tensor([0], dtype=torch.float32)
        elif model.model_name == 'NonZeroReLUCounter':
            opening_brackets = torch.tensor([0], dtype=torch.float32)
            closing_brackets = torch.tensor([0], dtype=torch.float32)
            excess_closing_brackets = torch.tensor([0], dtype=torch.float32)

        for j in range(len_seq):
            if model.model_name == 'LinearBracketCounter':
                out, previous_state = model(X[i][j].squeeze().to(device), previous_state)
            elif model.model_name == 'NonZeroReLUCounter':
                out, opening_brackets, closing_brackets, excess_closing_brackets = model(X[i][j].squeeze().to(device), opening_brackets, closing_brackets, excess_closing_brackets)
            elif model.model_name == 'TernaryLinearBracketCounter':
                out, previous_state = model(X[i][j].squeeze().to(device), previous_state)

            # output_seq[j]=out


        # with open(log_file, 'a') as f:
        #     f.write('////////////////////////////////////////\n')
        #     f.write('input sentence = ' + str(X[i]) + '\n')
        #     f.write('encoded sentence = ' + str(input_seq) + '\n')




        # with open(log_file, 'a') as f:
        #     f.write('actual output in train function = ' + str(out) + '\n')

        out_np = np.int_(out.detach().cpu().numpy() > 0.5)
        target_np = np.int_(target_seq.detach().cpu().numpy())

        total_loss+=criterion(out, y[i]).item()

        guess, guess_i = classFromOutput(out)
        class_i = labels.index(class_category)
        confusion[class_i][guess_i] += 1
        # current_loss += loss
        expected_classes.append(class_i)
        predicted_classes.append(guess_i)
        if guess == class_category:
            num_correct += 1

        #     with open(log_file, 'a') as f:
        #         f.write('CORRECT' + '\n')
        # else:
        #
        #     with open(log_file, 'a') as f:
        #         f.write('INCORRECT' + '\n')


        # with open(log_file, 'a') as f:
        #     f.write('rounded output function = ' + str(out_np) + '\n')
        #     f.write('target = ' + str(target_np) + '\n')

            #
            #
            # if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
            #     num_correct += 1
            #     # correct_arr.append(X[i])
            #     epoch_correct_guesses.append(X[i])
            #     if print_flag == True:
            #         with open(train_log, 'a') as f:
            #             f.write('CORRECT' + '\n')
            # else:
            #     epoch_incorrect_guesses.append(X[i])
            #     if print_flag == True:
            #         with open(train_log, 'a') as f:
            #             f.write('INCORRECT' + '\n')

        accuracy = num_correct / len(X) * 100
        # print('Accuracy for epoch ', epoch, '=', accuracy, '%')




        conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)





    # print(accuracies)
    # print(accuracy)
    return accuracy, total_loss/len(X)



def test(model, X, X_notencoded, y, y_notencoded, dataset):
    model.eval()

    if dataset=='test_20':
        log_file = test_20_log
    elif dataset=='test_30':
        log_file=test_30_log
    elif dataset=='test_40':
        log_file = test_40_log
    elif dataset=='test_50':
        log_file=test_50_log


    num_samples = len(X)


    print(model)

    num_correct = 0
    confusion = torch.zeros(num_classes, num_classes)
    predicted_classes = []
    expected_classes = []


    for i in range(len(X)):
        # model.zero_grad()
        # input_seq = Dyck.lineToTensor(X[i])
        # target_seq = Dyck.lineToTensorSigmoid(y[i])
        input_seq = X[i]
        target_seq = y[i]
        len_seq = len(input_seq)

        # output_seq = torch.zeros(target_seq.shape)

        input_seq.to(device)
        target_seq.to(device)
        # output_seq.to(device)

        input_tensor = X[i]
        class_tensor = y[i]
        input_sentence = X_notencoded[i]
        class_category = y_notencoded[i]

        # hidden = model.init_hidden()
        if model.model_name == 'LinearBracketCounter':
            previous_state = torch.tensor([0], dtype=torch.float32)
        elif model.model_name == 'NonZeroReLUCounter':
            opening_brackets = torch.tensor([0], dtype=torch.float32)
            closing_brackets = torch.tensor([0], dtype=torch.float32)
            excess_closing_brackets = torch.tensor([0], dtype=torch.float32)

        for j in range(len_seq):
            if model.model_name == 'LinearBracketCounter':
                out, previous_state = model(X[i][j].squeeze().to(device), previous_state)
            elif model.model_name == 'NonZeroReLUCounter':
                out, opening_brackets, closing_brackets, excess_closing_brackets = model(X[i][j].squeeze().to(device), opening_brackets, closing_brackets, excess_closing_brackets)
            elif model.model_name == 'TernaryLinearBracketCounter':
                out, previous_state = model(X[i][j].squeeze().to(device), previous_state)

            # output_seq[j]=out


        with open(log_file, 'a') as f:
            f.write('////////////////////////////////////////\n')
            f.write('input sentence = ' + str(X[i]) + '\n')
            f.write('encoded sentence = ' + str(input_seq) + '\n')




        with open(log_file, 'a') as f:
            f.write('actual output in train function = ' + str(out) + '\n')

        out_np = np.int_(out.detach().cpu().numpy() > 0.5)
        target_np = np.int_(target_seq.detach().cpu().numpy())

        guess, guess_i = classFromOutput(out)
        class_i = labels.index(class_category)
        confusion[class_i][guess_i] += 1
        # current_loss += loss
        expected_classes.append(class_i)
        predicted_classes.append(guess_i)
        if guess == class_category:
            num_correct += 1

            with open(log_file, 'a') as f:
                f.write('CORRECT' + '\n')
        else:

            with open(log_file, 'a') as f:
                f.write('INCORRECT' + '\n')


        with open(log_file, 'a') as f:
            f.write('rounded output function = ' + str(out_np) + '\n')
            f.write('target = ' + str(target_np) + '\n')

            #
            #
            # if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
            #     num_correct += 1
            #     # correct_arr.append(X[i])
            #     epoch_correct_guesses.append(X[i])
            #     if print_flag == True:
            #         with open(train_log, 'a') as f:
            #             f.write('CORRECT' + '\n')
            # else:
            #     epoch_incorrect_guesses.append(X[i])
            #     if print_flag == True:
            #         with open(train_log, 'a') as f:
            #             f.write('INCORRECT' + '\n')

        accuracy = num_correct / len(X) * 100
        # print('Accuracy for epoch ', epoch, '=', accuracy, '%')




        conf_matrix = sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)





    # print(accuracies)
    # print(accuracy)
    return accuracy



def main():
    # output_activation = 'Sigmoid'
    #
    # if task == 'TernaryClassification':
    #     num_classes = 3
    #     output_activation = 'Softmax'
    # elif task == 'BinaryClassification' or task == 'NextTokenPrediction':
    #     num_classes = 2
    #     output_activation = 'Sigmoid'

    input_size = n_letters

    with open(file_name, 'a') as f:
        f.write('Output activation = ' + output_activation + '\n')
        f.write('Optimiser used = ' + use_optimiser + '\n')
        f.write('Learning rate = ' + str(learning_rate) + '\n')
        # f.write('Batch size = ' + str(batch_size) + '\n')
        f.write('Number of runs = ' + str(num_runs) + '\n')
        f.write('Number of epochs in each run = ' + str(num_epochs) + '\n')
        # f.write('LR scheduler step = ' + str(lr_scheduler_step) + '\n')
        # f.write('LR Scheduler Gamma = ' + str(lr_scheduler_gamma) + '\n')
        f.write('Checkpoint step = ' + str(checkpoint_step) + '\n')
        f.write('Saved model name prefix = ' + modelname + '\n')
        f.write('Saved optimiser name prefix = ' + optimname + '\n')
        f.write('Excel name = ' + excel_name + '\n')
        f.write('Train log name = ' + train_log + '\n')
        f.write('Raw train log name = ' + train_log_raw + '\n')
        # f.write('Validation log name = ' + validation_log + '\n')
        # f.write('Long Validation log name = ' + long_validation_log + '\n')
        f.write('Test 20 log name = ' + test_20_log + '\n')
        f.write('Test 30 log name = ' + test_30_log + '\n')
        f.write('Test 40 log name = ' + test_40_log + '\n')
        f.write('Test 50 log name = ' + test_50_log + '\n')

        f.write('Plot name prefix = ' + plt_name + '\n')
        f.write('Checkpoint name prefix = ' + checkpoint + '\n')
        f.write('Checkpoint step = ' + str(checkpoint_step) + '\n')

        f.write('///////////////////////////////////////////////////////////////\n')
        f.write('\n')

    train_accuracies = []
    test_20_accuracies = []
    test_30_accuracies = []
    test_40_accuracies = []
    test_50_accuracies = []
    train_dataframes = []
    runs = []
    for i in range(num_runs):
        seed = i
        # seed = i
        torch.manual_seed(seed)
        np.random.seed(seed)
        with open(train_log, 'a') as f:
            f.write('random seed for run ' + str(i) + ' = ' + str(seed) + '\n')
        # model = select_model(counter_input_size=counter_input_size,counter_output_size=counter_output_size,output_size=output_size, initialisation=initialisation, output_activation=output_activation)
        # print(model.model_name)
        model=select_model()
        model.to(device)

        # log_dir="logs"
        # log_dir = "/content/drive/MyDrive/PhD/EXPT_LOGS/Dyck1_" + str(task) + "/Minibatch_Training/" + model_name + "/logs/run"+str(i)
        log_dir = path + "logs/run" + str(i)
        # sum_writer = SummaryWriter(log_dir)
        # modelname = modelname+"run"+str(i)+".pth"

        runs.append('run' + str(i))
        print('****************************************************************************\n')
        print('random seed = ', seed)
        train_accuracy, df = train(model, X_train, X_train_notencoded, y_train, y_train_notencoded, i)
        train_accuracies.append(train_accuracy)
        train_dataframes.append(df)
        # test_accuracy = test_model(model, test_loader, 'short')
        test_20_accuracy = test(model, X_20,X_20_notencoded,y_20,y_20_notencoded, 'test_20')
        test_20_accuracies.append(test_20_accuracy)
        test_30_accuracy = test(model, X_30, X_30_notencoded, y_30, y_30_notencoded, 'test_30')
        test_30_accuracies.append(test_30_accuracy)
        test_40_accuracy = test(model, X_40, X_40_notencoded, y_40, y_40_notencoded, 'test_40')
        test_40_accuracies.append(test_40_accuracy)
        test_50_accuracy = test(model, X_50, X_50_notencoded, y_50, y_50_notencoded, 'test_50')
        test_50_accuracies.append(test_50_accuracy)
        # long_test_accuracy = test_model(model, long_loader, 'long')
        # long_test_accuracies.append(long_test_accuracy)

        df.plot(x='epoch', y=['Average training losses'])
        plt.savefig(plt_name + '_losses_run' + str(i) + '.png')
        df.plot(x='epoch', y=['Training accuracies'])
        plt.savefig(plt_name + '_accuracies_run' + str(i) + '.png')
        # df.plot(x='epoch', y='learning rates')
        # plt.savefig(plt_name + '_learning_rates_run' + str(i) + '.png')
        # plt.savefig(plt_name+'_run')

        with open(file_name, "a") as f:
            # f.write('Saved model name for run '+str(i)+' = ' + modelname + '\n')
            f.write('train accuracy for run ' + str(i) + ' = ' + str(train_accuracy) + '%\n')
            f.write('test 20 accuracy for run ' + str(i) + ' = ' + str(test_20_accuracy) + '%\n')
            f.write('test 30 accuracy for run ' + str(i) + ' = ' + str(test_30_accuracy) + '%\n')
            f.write('test 40 accuracy for run ' + str(i) + ' = ' + str(test_40_accuracy) + '%\n')
            f.write('test 50 accuracy for run ' + str(i) + ' = ' + str(test_50_accuracy) + '%\n')

    dfs = dict(zip(runs, train_dataframes))
    writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')

    for sheet_name in dfs.keys():
        dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()

    max_train_accuracy = max(train_accuracies)
    min_train_accuracy = min(train_accuracies)
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    std_train_accuracy = np.std(train_accuracies)

    max_test_20_accuracy = max(test_20_accuracies)
    min_test_20_accuracy = min(test_20_accuracies)
    avg_test_20_accuracy = sum(test_20_accuracies) / len(test_20_accuracies)
    std_test_20_accuracy = np.std(test_20_accuracies)

    max_test_30_accuracy = max(test_30_accuracies)
    min_test_30_accuracy = min(test_30_accuracies)
    avg_test_30_accuracy = sum(test_30_accuracies) / len(test_30_accuracies)
    std_test_30_accuracy = np.std(test_30_accuracies)

    max_test_40_accuracy = max(test_40_accuracies)
    min_test_40_accuracy = min(test_40_accuracies)
    avg_test_40_accuracy = sum(test_40_accuracies) / len(test_40_accuracies)
    std_test_40_accuracy = np.std(test_40_accuracies)

    max_test_50_accuracy = max(test_50_accuracies)
    min_test_50_accuracy = min(test_50_accuracies)
    avg_test_50_accuracy = sum(test_50_accuracies) / len(test_50_accuracies)
    std_test_50_accuracy = np.std(test_50_accuracies)



    with open(file_name, "a") as f:
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum train accuracy = ' + str(max_train_accuracy) + '%\n')
        f.write('Minimum train accuracy = ' + str(min_train_accuracy) + '%\n')
        f.write('Average train accuracy = ' + str(avg_train_accuracy) + '%\n')
        f.write('Standard Deviation for train accuracy = ' + str(std_train_accuracy) + '\n')
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum test 20 accuracy = ' + str(max_test_20_accuracy) + '%\n')
        f.write('Minimum test 20 accuracy = ' + str(min_test_20_accuracy) + '%\n')
        f.write('Average test 20 accuracy = ' + str(avg_test_20_accuracy) + '%\n')
        f.write('Standard Deviation for test 20 accuracy = ' + str(std_test_20_accuracy) + '\n')
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum test 30 accuracy = ' + str(max_test_30_accuracy) + '%\n')
        f.write('Minimum test 30 accuracy = ' + str(min_test_30_accuracy) + '%\n')
        f.write('Average test 30 accuracy = ' + str(avg_test_30_accuracy) + '%\n')
        f.write('Standard Deviation for test 30 accuracy = ' + str(std_test_30_accuracy) + '\n')
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum test 40 accuracy = ' + str(max_test_40_accuracy) + '%\n')
        f.write('Minimum test 40 accuracy = ' + str(min_test_40_accuracy) + '%\n')
        f.write('Average test 40 accuracy = ' + str(avg_test_40_accuracy) + '%\n')
        f.write('Standard Deviation for test 40 accuracy = ' + str(std_test_40_accuracy) + '\n')
        f.write('/////////////////////////////////////////////////////////////////\n')
        f.write('Maximum test 50 accuracy = ' + str(max_test_50_accuracy) + '%\n')
        f.write('Minimum test 50 accuracy = ' + str(min_test_50_accuracy) + '%\n')
        f.write('Average test 50 accuracy = ' + str(avg_test_50_accuracy) + '%\n')
        f.write('Standard Deviation for test 50 accuracy = ' + str(std_test_50_accuracy) + '\n')
        f.write('/////////////////////////////////////////////////////////////////\n')



if __name__=='__main__':
    main()

