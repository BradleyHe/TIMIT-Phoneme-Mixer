import os
import sox
import torch
import yaml
import random
import operator
import itertools
import numpy as np

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

phn_occurence = {}


# load phoneme information
with open('config/strat_phn_occurence.txt') as f:
  for line in f.readlines():
    phn_occurence[line.split()[0]] = int(line.split()[1])

# chooses two random phonemes defined by arguments and mixes them
def mix_phonemes(phn1, phn2):
  phn1_list = [os.path.join('strat_phoneme_set', phn1, file) for file in os.listdir('strat_phoneme_set/' + phn1)]
  phn2_list = [os.path.join('strat_phoneme_set', phn2, file) for file in os.listdir('strat_phoneme_set/' + phn2)]

  file1 = phn1_list[random.randint(0, phn_occurence[phn1] - 1)]
  file2 = phn2_list[random.randint(0, phn_occurence[phn2] - 1)]

  rms1 = sox.file_info.stat(file1)['RMS     amplitude']
  rms2 = sox.file_info.stat(file2)['RMS     amplitude']
  factor = tir_factor(0, rms1, rms2)


  cbn = sox.Combiner()
  cbn.set_input_format(file_type=['wav', 'wav'])
  cbn.build([file1, file2], 'new.wav', 'mix', [1, 1 / factor])

  tfn = sox.Transformer()
  pred = test_mixed('new.wav')
  print(pred)

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

def test_all():
  mix_count = 500 
  phonemes = ['ah', 's', 'er', 'ih', 't', 'ow', 'ey', 'ay', 'aa', 'eh']

  ''' Many vowel phonemes phonemes listed here are interpreted by the LAS model as having a 'd' phoneme before it.
      For example, an evaluation of an 'ah' sound file would return as "#h d ah #h".
      We hypothesise that this might occur since the LAS model is not trained to recognize singular phonemes like this, 
      so it gives a more "word based" guess by inserting a constanant before it. However we cannot confirm this.
      Regardless, this leader phoneme must be removed after evaluation in order to provide insightful results
      This is the reason why the 'd' phoneme is not included in the phoneme mixing procedure, as it would be 
      impossible to tell if the LAS model is predicting a 'd' or is placing a 'd' before a vowel.
  '''
  mix_phn_list = list(itertools.combinations_with_replacement(phonemes, 2))

  for phns in mix_phn_list:
    mix_phonemes(phns[0], phns[1])

# load LAS model
config_path = 'config/las_example_config.yaml'
conf = yaml.load(open(config_path,'r'))

listener = torch.load(conf['training_parameter']['pretrained_listener_path'], map_location=lambda storage, loc: storage)
speller = torch.load(conf['training_parameter']['pretrained_speller_path'], map_location=lambda storage, loc: storage)
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], lr=conf['training_parameter']['learning_rate'])

# mix_phonemes('ow', 't') 
test_all()