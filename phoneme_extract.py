import sox
import os
import operator

source_path = os.path.join('TIMIT', 'TEST')

# sample rate of TIMIT is 16khz
sample_rate = 16000

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

for dirName, subdirList, fileList in os.walk(source_path):
  for file in fileList:
    if file.endswith('.PHN'):
      # load phoneme description
      lines = []
      with open(dirName + '/' + file) as f:
        [lines.append(line.rstrip().split(' ')) for line in f.readlines()]

      for line in lines:
        col_phone = collapse_phn(line[2])

        # ignore silence
        if(col_phone == 'h#'):
          continue

        os.makedirs(os.path.join('phoneme_set', col_phone), exist_ok=True)

        if col_phone not in phn_count:
          phn_count[col_phone] = 1
        else:
          phn_count[col_phone] += 1

sorted_phn = sorted(phn_count.items(), key=operator.itemgetter(1), reverse=True)
with open('phn_occurence.txt', 'w+') as f:
  [f.write(phn[0] + ' ' + str(phn[1]) + '\n') for phn in sorted_phn]



