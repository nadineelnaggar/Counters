path_binary_nobias = "/content/drive/MyDrive/PhD/EXPT_LOGS/Counters/BracketCounting/"

from models import LinearBracketCounter
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# % matplotlib
# inline

models_trained_2 = []
models_trained_4 = []
models_trained_8 = []

models_2_a_values = []
models_2_b_values = []

models_2_ab_ratio = []
models_2_u_value = []

models_4_a_values = []
models_4_b_values = []

models_4_ab_ratio = []
models_4_u_value = []

models_8_a_values = []
models_8_b_values = []

models_8_ab_ratio = []
models_8_u_value = []

num_runs = 10

prefix_2 = path_binary_nobias + 'Sigmoid_activation_2train_seq_length_random_initialisation_100epochs_'
prefix_4 = path_binary_nobias + 'Sigmoid_activation_4train_seq_length_random_initialisation_100epochs_'
prefix_8 = path_binary_nobias + 'Sigmoid_activation_8train_seq_length_random_initialisation_100epochs_'
model_prefix_2 = prefix_2 + 'MODEL_run'
model_prefix_4 = prefix_4 + 'MODEL_run'
model_prefix_8 = prefix_8 + 'MODEL_run'
# Clipping_activation_2train_seq_length_random_initialisation_100epochs_MODEL_run1.pth

for i in range(num_runs):
    model_2 = LinearBracketCounter(counter_input_size=3, counter_output_size=1, output_size=1)
    model_path_2 = model_prefix_2 + '' + str(i) + '.pth'
    model_2.load_state_dict(torch.load(model_path_2))

    model_4 = LinearBracketCounter(counter_input_size=3, counter_output_size=1, output_size=1)
    model_path_4 = model_prefix_4 + '' + str(i) + '.pth'
    model_4.load_state_dict(torch.load(model_path_4))

    model_8 = LinearBracketCounter(counter_input_size=3, counter_output_size=1, output_size=1)
    model_path_8 = model_prefix_8 + '' + str(i) + '.pth'
    model_8.load_state_dict(torch.load(model_path_8))

    model_2_a_value = model_2.counter.weight[0][0].item()
    model_2_b_value = model_2.counter.weight[0][1].item()
    model_2_u_value = model_2.counter.weight[0][2].item()
    model_2_ab_ratio = model_2_a_value / model_2_b_value

    models_2_a_values.append(model_2_a_value)
    models_2_b_values.append(model_2_b_value)
    models_2_ab_ratio.append(model_2_ab_ratio)
    models_2_u_value.append(model_2_u_value)

    model_4_a_value = model_4.counter.weight[0][0].item()
    model_4_b_value = model_4.counter.weight[0][1].item()
    model_4_u_value = model_4.counter.weight[0][2].item()
    model_4_ab_ratio = model_4_a_value / model_4_b_value

    models_4_a_values.append(model_4_a_value)
    models_4_b_values.append(model_4_b_value)
    models_4_ab_ratio.append(model_4_ab_ratio)
    models_4_u_value.append(model_4_u_value)

    model_8_a_value = model_8.counter.weight[0][0].item()
    model_8_b_value = model_8.counter.weight[0][1].item()
    model_8_u_value = model_8.counter.weight[0][2].item()
    model_8_ab_ratio = model_8_a_value / model_8_b_value

    models_8_a_values.append(model_8_a_value)
    models_8_b_values.append(model_8_b_value)
    models_8_ab_ratio.append(model_8_ab_ratio)
    models_8_u_value.append(model_8_u_value)

print(len(models_2_ab_ratio))
print(len(models_2_u_value))

abr_bins = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0]
u_bins = [0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

plt.rcParams['font.size'] = '14'
plt.subplots()
plt.rcParams['font.size'] = '14'
plt.hist(models_2_ab_ratio, histtype='step', linestyle=('dashed'), bins=abr_bins, label='TSL 2')
plt.hist(models_4_ab_ratio, histtype='step', linestyle=('dotted'), color='g', bins=abr_bins, label='TSL 4')
plt.hist(models_8_ab_ratio, histtype='step', linestyle=('dashed'), color='r', bins=abr_bins, label='TSL 8')
plt.legend()
plt.xlabel('AB Ratio')
plt.show()

plt.subplots()
plt.rcParams['font.size'] = '14'
plt.hist(models_2_u_value, histtype='step', linestyle=('dashed'), bins=u_bins, label='TSL 2')
plt.hist(models_4_u_value, histtype='step', linestyle=('dotted'), color='g', bins=u_bins, label='TSL 4')
plt.hist(models_8_u_value, histtype='step', linestyle=('dashed'), color='r', bins=u_bins, label='TSL 8')
plt.legend()
plt.xlabel('Recurrent Weight U')
plt.show()

print(models_2_ab_ratio)
print(models_2_u_value)