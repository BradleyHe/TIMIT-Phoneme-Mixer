import os
import sox
import torch
import yaml
import shutil
import operator
import numpy as np

from util.preprocess_functions import preprocess_dataset,normalize,set_type
from util.timit_dataset import create_dataloader
from util.functions import test_file
from six.moves import cPickle 

# if this is removed, pytorch will output harmless warning messages that may be irritating
import warnings
warnings.filterwarnings("ignore")

phn_occurrence = {}

# load phoneme information
with open('config/phn_occurrence.txt') as f:
  for line in f.readlines():
    phn_occurrence[line.split()[0]] = int(line.split()[1])

phonemes = ['ih', 'n', 'iy', 'l', 's', 'r', 'ah', 'aa', 'er', 'k', 'm', 't', 'eh', 'ae', 'z', 'd', 'q', 'w', 'dh', 'p', 
						'dx', 'f', 'b', 'sh', 'ay', 'ey', 'ow', 'g', 'uw', 'hh', 'v', 'y', 'ng', 'jh', 'th', 'oy', 'ch', 'uh', 'aw']
strat_phn_count = {}

def get_pred(path):
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

# load LAS model
config_path = 'config/las_example_config.yaml'
conf = yaml.load(open(config_path,'r'))

listener = torch.load(conf['training_parameter']['pretrained_listener_path'], map_location=lambda storage, loc: storage)
speller = torch.load(conf['training_parameter']['pretrained_speller_path'], map_location=lambda storage, loc: storage)
optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], lr=conf['training_parameter']['learning_rate'])

for phn in phonemes:
	i = 0
	for file in os.listdir(os.path.join('phoneme_set', phn)):
		print('Testing {} {} out of {}'.format(phn, str(i), str(phn_occurrence[phn])), end='\r')
		test = os.path.join('phoneme_set', phn, file)
		pred = get_pred(test)

		if phn in pred:
			if phn not in strat_phn_count:
				strat_phn_count[phn] = 1
			else:
				strat_phn_count[phn] += 1
			os.makedirs(os.path.join('strat_phoneme_set', phn), exist_ok=True)
			shutil.copy(test, os.path.join('strat_phoneme_set', phn, phn + str(strat_phn_count[phn]) + '.wav'))

		i += 1
	print()


sorted_phn = sorted(strat_phn_count.items(), key=operator.itemgetter(1), reverse=True)
with open('config/strat_phn_occurrence.txt', 'w+') as f:
  [f.write(phn[0] + ' ' + str(phn[1]) + '\n') for phn in sorted_phn]

