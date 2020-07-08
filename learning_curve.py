"""
1. Pooling training set by various portion.
2. Creating HMM model for each pooled training set.
3. Accumulating the results and plotting it.
"""

import numpy as np
import subprocess
import matplotlib
import matplotlib.pyplot as plt

TAG_FILE = "ptb.2-21.tgs"
TOKEN_FILE = "ptb.2-21.txt"

select_grid = 100
starting_portion = 0.01
ending_portion = 1
step = 0.01
X = np.arange(1, 101, 1)


with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
    tags = tag_file.read().split('\n')[:-1]
    tokens = token_file.read().split('\n')[:-1]

training_size = len(tags)
shuffler = np.arange(training_size)
np.random.shuffle(shuffler)

np_tags = []
np_tokens = []
for i in range(training_size):
    np_tags.append(tags[shuffler[i]])
    np_tokens.append(tokens[shuffler[i]])
np_tags = np.asarray(np_tags)
np_tokens = np.asarray(np_tokens)

subprocess.call('mkdir training_pool', shell=True)

for i in range(select_grid):
    select_size = int((starting_portion + i*step) * training_size)
    np.savetxt("training_pool/ptb.2-21-%d.tgs" % (i+1), np_tags[:select_size], fmt="%s")
    np.savetxt("training_pool/ptb.2-21-%d.txt" % (i+1), np_tokens[:select_size], fmt="%s")
    print(i)

subprocess.call('mkdir model', shell=True)
subprocess.call('mkdir output', shell=True)
result_word = []
result_sentence = []
for i in range(select_grid):
    subprocess.call('python3 train_hmm.py training_pool/ptb.2-21-%d.tgs training_pool/ptb.2-21-%d.txt '
                    'model/model_%d.hmm' % (i+1, i+1, i+1), shell=True)
    subprocess.call('python3 viterbi.py model/model_%d.hmm ptb.22.txt '
                    'output/ptb.22-%d.tags.txt' % (i+1, i+1), shell=True)
    result = subprocess.check_output('python3 tag_acc.py ptb.22.tgs output/ptb.22-%d.tags.txt' % (i+1), shell=True)
    result = result.split()
    r_word = float(result[4])
    r_sentence = float(result[14])
    result_word.append(r_word)
    result_sentence.append(r_sentence)

Y = result_word

fig, ax = plt.subplots()
ax.plot(X, Y)

ax.set(xlabel='Amount of training data(%)', ylabel='Error rate',
       title='Error rate by Words')
ax.grid()

fig.savefig("ER_by_Words.png")
plt.show()

Y = result_sentence
fig, ax = plt.subplots()
ax.plot(X, Y)

ax.set(xlabel='Amount of training data(%)', ylabel='Error rate',
       title='Error rate by Words')
ax.grid()

fig.savefig("ER_by_Sentences.png")
plt.show()


print("stop")
