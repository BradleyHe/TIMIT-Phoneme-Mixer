import os
import sox
import torch
import yaml
import random
import operator
import itertools
import numpy as np
import pandas as pd

# if this is removed, pytorch will output harmless warning messages that may be irritating
import warnings
warnings.filterwarnings("ignore")

from util.preprocess_functions import preprocess_dataset,normalize,set_type
from util.timit_dataset import create_dataloader
from util.functions import test_file
from six.moves import cPickle 

def calculate_tir(target, interference):
  return 10 * np.log10(target ** 2 / interference ** 2) 

def tir_factor(ratio, target, interference):
  return 10 ** ((ratio - calculate_tir(target, interference)) / 20)

phn_occurrence = {}
#test_set = 'strat_phoneme_set'
test_set = 'phoneme_set'

# load phoneme information
with open('config/strat_phn_occurrence.txt') as f:
  for line in f.readlines():
    phn_occurrence[line.split()[0]] = int(line.split()[1])

# chooses two random phonemes defined by arguments and mixes them
def mix_phonemes(phn1, phn2):
  phn1_list = [os.path.join(test_set, phn1, file) for file in os.listdir(os.path.join(test_set, phn1))]
  phn2_list = [os.path.join(test_set, phn2, file) for file in os.listdir(os.path.join(test_set, phn2))]

  file1 = phn1_list[random.randint(0, phn_occurrence[phn1] - 1)]
  file2 = phn2_list[random.randint(0, phn_occurrence[phn2] - 1)]

  rms1 = sox.file_info.stat(file1)['RMS     amplitude']
  rms2 = sox.file_info.stat(file2)['RMS     amplitude']
  factor = tir_factor(0, rms1, rms2)

  cbn = sox.Combiner()
  cbn.set_input_format(file_type=['wav', 'wav'])
  cbn.build([file1, file2], 'test/new.wav', 'mix', [1, 1 / factor])

  pred = test_mixed('test/new.wav')
  os.remove('test/new.wav')
  return pred

def test_mixed(path):
  data_type = 'float32'

  mean_val = np.loadtxt('config/mean_val.txt')
  std_val = np.loadtxt('config/std_val.txt')

  x, y = preprocess_dataset(path)

  x = normalize(x, mean_val, std_val)
  x = set_type(x, data_type)

  test_set = create_dataloader(x, y, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)

  for batch_index,(batch_data,batch_label) in enumerate(test_set):
    pred,true = test_file(batch_data, batch_label, listener, speller, optimizer, **conf['model_parameter'])
    return pred

# test all combinations of phonemes
def test_all():
  occurrence = {}
  output_len = {}
  not_present = {}
  mix_count = 1000
  folder_name = 'original2'

  all_phonemes = os.listdir('phoneme_set')

  # only phonemes (besides d) that were recognized by LAS more than 200 times are included here
  test_phonemes = ['ah', 's', 'er', 'ih', 't', 'ow', 'ey', 'ay', 'aa', 'eh']
 
  ''' Many vowel phonemes phonemes listed here are interpreted by the LAS model as having a 'd' phoneme before it.
      For example, an evaluation of an 'ah' sound file would return as "h# d ah h#".
      We hypothesise that this might occur since the LAS model is not trained to recognize singular phonemes like this, 
      so it gives a more "word based" guess by inserting a constanant before it. However we cannot confirm this.
      Regardless, this leader phoneme must be removed after evaluation in order to provide insightful results
      This is the reason why the 'd' phoneme is not included in the phoneme mixing procedure, as it would be 
      impossible to tell if the LAS model is predicting a 'd' or is placing a 'd' before a vowel.
  '''
  remove_phonemes = ['h#', 'd']

  mix_phn_list = list(itertools.combinations_with_replacement(test_phonemes, 2))

  # create data dictionaries
  for phns in mix_phn_list:
    occurrence['{}_{}'.format(phns[0], phns[1])] = {}
    output_len['{}_{}'.format(phns[0], phns[1])] = 0
    not_present['{}_{}'.format(phns[0], phns[1])] = 0
    for phn in all_phonemes:
      occurrence['{}_{}'.format(phns[0], phns[1])][phn] = 0

  for phns in mix_phn_list:
    total_length = 0
    for x in range(mix_count):
      print('Mixing {} and {} #{}'.format(phns[0], phns[1], x + 1), end='\r')
      pred = mix_phonemes(phns[0], phns[1])
      pred = [phn for phn in pred if phn not in remove_phonemes]
      pred = list(set(pred)) #remove duplicates

      # count occurrence of phns
      for phn in pred:
        occurrence['{}_{}'.format(phns[0], phns[1])][phn] += 1

      # count num of outputs that did not have original phonemes in them 
      if phns[0] not in pred and phns[1] not in pred:
        not_present['{}_{}'.format(phns[0], phns[1])] += 1

      total_length += len(pred)

    output_len['{}_{}'.format(phns[0], phns[1])] = total_length / mix_count
    print('average output length: {}'.format(total_length / mix_count))

    df = pd.DataFrame.from_dict(occurrence, orient='index')
    df.to_csv('data/' + folder_name + '/' + folder_name + '.csv')

  with open('data/' + folder_name + '/output_len.txt', 'w+') as f:
    [f.write(phns + ': ' + str(length) + '\n') for phns,length in output_len.items()]

  with open('data/' + folder_name + '/not_present.txt', 'w+') as f:
    [f.write(phns + ': ' + str(num) + '\n') for phns,num in not_present.items()]

  df = pd.DataFrame.from_dict(occurrence, orient='index')
  df.to_csv('data/' + folder_name + '/' + folder_name + '.csv')

# load LAS model
config_path = 'config/las_example_config.yaml'
conf = yaml.load(open(config_path,'r'))

listener = torch.load(conf['training_parameter']['pretrained_listener_path'], map_location=lambda storage, loc: storage)
speller = torch.load(conf['training_parameter']['pretrained_speller_path'], map_location=lambda storage, loc: storage)
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], lr=conf['training_parameter']['learning_rate'])
os.makedirs('data', exist_ok=True)
test_all()