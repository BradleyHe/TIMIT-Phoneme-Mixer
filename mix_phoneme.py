import os
import sox
import torch
import yaml
import random
import operator
import numpy as np

from preprocess_functions import preprocess_dataset,normalize,set_type
from util.timit_dataset import create_dataloader
from util.functions import test_file
from six.moves import cPickle 

def calculate_tir(target, interference):
  return 10 * np.log10(target ** 2 / interference ** 2) 

def tir_factor(ratio, target, interference):
  return 10 ** ((ratio - calculate_tir(target, interference)) / 20)

phn_occurence = {}

# load phoneme information
with open('phn_occurence.txt') as f:
  for line in f.readlines():
    phn_occurence[line.split()[0]] = int(line.split()[1])

# chooses two random phonemes defined by arguments and mixes them
def mix_phonemes(phn1, phn2):
  # sort phoneme audio files by length
  phn1_list = []
  for path, dirs, files in os.walk('phoneme_set/' + phn1):
    phn1_list.extend([(os.path.join(path, file), os.path.getsize(os.path.join(path, file))) for file in files])
  phn1_list.sort(key=operator.itemgetter(1))

  phn2_list = []
  for path, dirs, files in os.walk('phoneme_set/' + phn2):
    phn2_list.extend([(os.path.join(path, file), os.path.getsize(os.path.join(path, file))) for file in files])
  phn2_list.sort(key=operator.itemgetter(1))

  id1 = random.randint(int(phn_occurence[phn1]/2), phn_occurence[phn1])
  id2 = random.randint(int(phn_occurence[phn2]/2), phn_occurence[phn2])
  file1 = phn1_list[id1][0]
  file2 = phn2_list[id2][0]

  rms1 = sox.file_info.stat(file1)['RMS     amplitude']
  rms2 = sox.file_info.stat(file2)['RMS     amplitude']
  factor = tir_factor(0, rms1, rms2)

  cbn = sox.Combiner()
  cbn.set_input_format(file_type=['wav', 'wav'])
  cbn.build([file1, file2], 'new.wav', 'mix', [1, 1 / factor])

  tfn = sox.Transformer()
  tfn.pad(0.1)
  tfn.build('new.wav', 'new1.wav')
  test_mixed('new1.wav')

def test_mixed(path):
  data_type = 'float32'

  mean_val = np.loadtxt('config/mean_val.txt')
  std_val = np.loadtxt('config/std_val.txt')

  x, y = preprocess_dataset(path)

  x = normalize(x, mean_val, std_val)
  x = set_type(x, data_type)

  config_path = 'config/las_example_config.yaml'
  conf = yaml.load(open(config_path,'r'))

  test_set = create_dataloader(x, y, **conf['model_parameter'], **conf['training_parameter'], shuffle=False)
  listener = torch.load(conf['training_parameter']['pretrained_listener_path'], map_location=lambda storage, loc: storage)
  speller = torch.load(conf['training_parameter']['pretrained_speller_path'], map_location=lambda storage, loc: storage)
  optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], lr=conf['training_parameter']['learning_rate'])

  for batch_index,(batch_data,batch_label) in enumerate(test_set):
    pred,true = test_file(batch_data, batch_label, listener, speller, optimizer, **conf['model_parameter'])
    print(pred)

mix_phonemes('ih', 'ih')
