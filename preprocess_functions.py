# reference : https://github.com/Faur/TIMIT
# 			  https://github.com/jameslyons/python_speech_features/issues/53
import os
import wave
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle 
import numpy as np
import scipy.io.wavfile as wav

import python_speech_features as features
# a python package for speech features at https://github.com/jameslyons/python_speech_features

# 61 different phonemes
phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
	"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
	"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
	"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

phonemes2index = {k:v for v,k in enumerate(phonemes)}

def get_total_duration(file):
	"""Get the length of the phoneme file, i.e. the 'time stamp' of the last phoneme"""
	for line in reversed(list(open(file))):
		[_, val, _] = line.split()
		return int(val)

def create_mfcc(filename):
	"""Perform standard preprocessing, as described by Alex Graves (2012)
	http://www.cs.toronto.edu/~graves/preprint.pdf
	Output consists of 12 MFCC and 1 energy, as well as the first derivative of these.
	[1 energy, 12 MFCC, 1 diff(energy), 12 diff(MFCC)
	"""
	(rate,sample) = wav.read(filename)

	mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
	preemph=0.97, appendEnergy=True)
	d_mfcc = features.delta(mfcc, 2)
	a_mfcc = features.delta(d_mfcc, 2)

	out = np.concatenate([mfcc, d_mfcc, a_mfcc], axis=1)

	return out, out.shape[0]

def calc_norm_param(X):
	"""Assumes X to be a list of arrays (of differing sizes)"""
	total_len = 0
	mean_val = np.zeros(X[0].shape[1])
	std_val = np.zeros(X[0].shape[1])
	for obs in X:
		obs_len = obs.shape[0]
		mean_val += np.mean(obs,axis=0)*obs_len
		std_val += np.std(obs, axis=0)*obs_len
		total_len += obs_len
	
	mean_val /= total_len
	std_val /= total_len

	return mean_val, std_val, total_len

def normalize(X, mean_val, std_val):
	for i in range(len(X)):
		X[i] = (X[i] - mean_val)/std_val
	return X

def set_type(X, type):
	for i in range(len(X)):
		X[i] = X[i].astype(type)
	return X

def preprocess_dataset(fname):
	X = []
	Y = []

	phn_fname = fname
	wav_fname = fname[0:-4] + '.wav'

	total_duration = 1
	fr = ['0 1 ih']

	X_val, total_frames = create_mfcc(wav_fname)
	total_frames = int(total_frames)

	X.append(X_val)

	y_val = np.zeros(total_frames) - 1
	start_ind = 0
	for line in fr:
		[start_time, end_time, phoneme] = line.rstrip('\n').split()
		start_time = int(start_time)
		end_time = int(end_time)

		phoneme_num = phonemes2index[phoneme] if phoneme in phonemes2index else -1
		end_ind = int(np.round((end_time)/total_duration*total_frames))
		y_val[start_ind:end_ind] = phoneme_num

		start_ind = end_ind

	if -1 in y_val:
		print('WARNING: -1 detected in TARGET')

	Y.append(y_val.astype('int32'))

	return X, Y