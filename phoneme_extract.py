import sox
import os
import operator

source_path = os.path.join('TIMIT', 'TEST')

# sample rate of TIMIT is 16khz
sample_rate = 16000

# total of 51368 non-silence phonemes in TIMIT testing dataset
total_phn = 51368
curr_phn = 1

# collapse 61 phn to 39 phn 
# http://cdn.intechopen.com/pdfs/15948/InTech-Phoneme_recognition_on_the_timit_database.pdf
def collapse_phn(char):
  collapse_dict = {"b":"b", "bcl":"h#", "d":"d", "dcl":"h#", "g":"g", "gcl":"h#", "p":"p", "pcl":"h#", "t":"t", "tcl":"h#", "k":"k", "kcl":"h#", "dx":"dx", "q":"q", "jh":"jh", "ch":"ch", "s":"s", "sh":"sh", "z":"z", "zh":"sh", 
    "f":"f", "th":"th", "v":"v", "dh":"dh", "m":"m", "n":"n", "ng":"ng", "em":"m", "en":"n", "eng":"ng", "nx":"n", "l":"l", "r":"r", "w":"w", "y":"y", 
    "hh":"hh", "hv":"hh", "el":"l", "iy":"iy", "ih":"ih", "eh":"eh", "ey":"ey", "ae":"ae", "aa":"aa", "aw":"aw", "ay":"ay", "ah":"ah", "ao":"aa", "oy":"oy",
    "ow":"ow", "uh":"uh", "uw":"uw", "ux":"uw", "er":"er", "ax":"ah", "ix":"ih", "axr":"er", "ax-h":"ah", "pau":"h#", "epi":"h#", "h#": "h#"}
  return collapse_dict[char]

tfn = sox.Transformer()
phn_count = {}

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

        # cut section of audio that contains phoneme
        os.makedirs(os.path.join('phoneme_set', col_phone), exist_ok=True)
        tfn.trim(float(line[0]) / sample_rate, float(line[1]) / sample_rate)
        tfn.pad(0.1)
        tfn.build(os.path.join(dir_name, file[:-4] + '.wav'), os.path.join('phoneme_set', col_phone, col_phone + str(phn_count[col_phone]) + '.wav'))

        print('Extracted phoneme {} out of {}'.format(curr_phn, total_phn), end='\r')
        curr_phn += 1

        # reset transformer
        tfn.clear_effects()


sorted_phn = sorted(phn_count.items(), key=operator.itemgetter(1), reverse=True)
with open('config/phn_occurence.txt', 'w+') as f:
  [f.write(phn[0] + ' ' + str(phn[1]) + '\n') for phn in sorted_phn]



