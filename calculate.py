import os
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def collapse_phn(char):
  collapse_dict = {"b":"b", "bcl":"h#", "d":"d", "dcl":"h#", "g":"g", "gcl":"h#", "p":"p", "pcl":"h#", "t":"t", "tcl":"h#", "k":"k", "kcl":"h#", "dx":"dx", "q":"q", "jh":"jh", "ch":"ch", "s":"s", "sh":"sh", "z":"z", "zh":"sh", 
    "f":"f", "th":"th", "v":"v", "dh":"dh", "m":"m", "n":"n", "ng":"ng", "em":"m", "en":"n", "eng":"ng", "nx":"n", "l":"l", "r":"r", "w":"w", "y":"y", 
    "hh":"hh", "hv":"hh", "el":"l", "iy":"iy", "ih":"ih", "eh":"eh", "ey":"ey", "ae":"ae", "aa":"aa", "aw":"aw", "ay":"ay", "ah":"ah", "ao":"aa", "oy":"oy",
    "ow":"ow", "uh":"uh", "uw":"uw", "ux":"uw", "er":"er", "ax":"ah", "ix":"ih", "axr":"er", "ax-h":"ah", "pau":"h#", "epi":"h#", "h#": "h#"}
  return collapse_dict[char]

# generates number of phonemes in complete training set
def count_train_phns():
  source_path = os.path.join('TIMIT', 'TRAIN')
  phn_count = {}
  curr_phn = 1

  # extracts all instances of non-silence phonemes
  for dir_name, subdir_list, file_list in os.walk(source_path):
    for file in file_list:
      if file.endswith('.PHN'):
        # load phoneme description
        lines = []
        with open(os.path.join(dir_name, file)) as f:
          [lines.append(line.rstrip().split(' ')) for line in f.readlines()]

        for line in lines:
          col_phone = collapse_phn(line[2])

          # ignore silence
          if(col_phone == 'h#'):
            continue

          # record phoneme count
          if col_phone not in phn_count:
            phn_count[col_phone] = 1
          else:
            phn_count[col_phone] += 1

          print('Extracted phoneme {} out of {}'.format(curr_phn, 141203), end='\r')
          curr_phn += 1

  sorted_phn = sorted(phn_count.items(), key=operator.itemgetter(1), reverse=True)
  with open('data/train_phn_occurrence.txt', 'w+') as f:
    [f.write(phn[0] + ' ' + str(phn[1]) + '\n') for phn in sorted_phn]

# calculates phoneme accuracy of model during stratification
def strat_accuracy_pct():
  strat_phn_occurrence = {}
  phn_occurrence = {}

  with open('config/phn_occurrence.txt') as f:
    for line in f.readlines():
      phn_occurrence[line.split()[0]] = int(line.split()[1])

  with open('config/strat_phn_occurrence.txt') as f:
    for line in f.readlines():
      strat_phn_occurrence[line.split()[0]] = int(line.split()[1])

  acc = {}

  for phn, num in phn_occurrence.items():
    if phn not in strat_phn_occurrence:
      acc[phn] = 0
    else:
      acc[phn] = strat_phn_occurrence[phn] / phn_occurrence[phn]

  sorted_acc = sorted(acc.items(), key=operator.itemgetter(1), reverse=True)
  print(sorted_acc)

  with open('data/accuracy.txt', 'w+') as f:
    [f.write(tuple[0] + ' ' + str(tuple[1]) + '\n') for tuple in sorted_acc]

# calculates average predicted phoneme sequence length
def calc_avg_len():
  length = 0

  with open('data/original1/output_len.txt') as f:
    for line in f.readlines():
      length += float(line.split(' ')[1])
  # with open('data/stratified2/output_len.txt') as f:
  #   for line in f.readlines():
  #     length += float(line.split(' ')[1])
  print(length / 55)

# averages data together
def average_data():
  df = pd.DataFrame()
  df = df.append(pd.read_csv('data/original1/original1.csv', index_col=0))
  df = df.append(pd.read_csv('data/original2/original2.csv', index_col=0))
  data_mean = df.groupby(level=0).mean()
  data_mean.to_csv('data/data_mean_original.csv')

# calculate percentage of time in which neither of the two phonemes were predicted in a mixed phoneme sample (error rate)
def mixed_acc():
  acc = {}
  with open('data/original1/not_present.txt') as f:
    lines = f.readlines()
    for line in lines:
      # comment out if duplicates are wanted
      #if(line.split(': ')[0].split('_')[0] != line.split(': ')[0].split('_')[1]):
        acc[line.split(': ')[0]] = int(line.split(': ')[1])

  
  with open('data/original1/not_present.txt') as f:
    lines = f.readlines()
    for line in lines:
      # comment out if duplicates are wanted
      #if(line.split(': ')[0].split('_')[0] != line.split(': ')[0].split('_')[1]):
        acc[line.split(': ')[0]] += int(line.split(': ')[1])
  
  # separated by combinations
  # sorted_acc = sorted(acc.items(), key=operator.itemgetter(1), reverse=False)
  # print(sorted_acc)

  #total accuracy
  total = 0
  for key,val in acc.items():
    total += val
  print(total / (55 * 2000))

# calculates percentage of time in which a phoneme was predicted for all of its mixings with other phonemes
def calc_phn_acc(path):
  df = pd.read_csv('data/' + path + '.csv', index_col=0)
  acc = dict.fromkeys(['ow', 'ey', 'ah', 'ay', 'er', 's', 't', 'aa', 'ih', 'eh'], 0)

  for index,row in df.iterrows():
    acc[index.split('_')[0]] += row[index.split('_')[0]] / 10 / (1000)
    if index.split('_')[0] != index.split('_')[1]:
      acc[index.split('_')[1]] += row[index.split('_')[1]] / 10 / (1000)
  return acc
  
def graph_scatter():
  phns = ['ow', 'ey', 'ah', 'ay', 'er', 's', 't', 'aa', 'ih', 'eh']
  accuracy = []
  complete_pred_rate = []
  strat_pred_rate = []

  with open('data/accuracy.txt') as f:
    for line in f.readlines():
      accuracy.append(float(line.split(' ')[1]))
  accuracy = accuracy[1:11]
  
  complete_pred_rate = list(calc_phn_acc('data_mean_original').values())
  strat_pred_rate = list(calc_phn_acc('data_mean_stratified').values())

  os.makedirs('graphs', exist_ok=True)

  fig = plt.figure()

  for i in range(len(phns)):
    x = accuracy[i]

    y = complete_pred_rate[i]
    #y = strat_pred_rate[i]

    name = phns[i]
    plt.plot(x, y, 'bo', ms = 5)
    plt.text(x + 0.007, y + 0.005, name, fontdict={'fontsize': 10, 'fontweight': 'medium'})

  ax = fig.add_subplot(1, 1, 1)
  ax.set_xlim(0.1, 0.8)
  ax.set_ylim(0.0, 0.6)
  
  ax.set_title('Complete Phoneme Set', fontdict={'fontsize': 10, 'fontweight': 'medium'})
  #ax.set_title('Stratified Phoneme Set', fontdict={'fontsize': 10, 'fontweight': 'medium'})

  plt.grid(True)
  plt.setp(ax.spines.values(), linewidth=2)

  plt.xticks(np.arange(0.1, 0.9, 0.1))
  plt.yticks(np.arange(0.0, 0.7, 0.1))
  plt.xlabel('Stratification Accuracy')
  plt.ylabel('Average Prediction Rate')

  plt.savefig('graphs/complete.pdf', bbox_inches = 'tight', pad_inches = 0)
  #plt.savefig('graphs/stratified.pdf', bbox_inches = 'tight', pad_inches = 0)

def average_phn_size():
  phns = ['ow', 'ey', 'ah', 'ay', 'er', 's', 't', 'aa', 'ih', 'eh']
  for phn in phns:
    path = 'strat_phoneme_set/' + phn
    print(sum(os.path.getsize(path + '/' + f) for f in os.listdir(path)) / len(os.listdir(path)))


#count_train_phns()
#strat_accuracy_pct()
#calc_avg_len()
#average_data()
#mixed_acc()
#graph_scatter()
average_phn_size()