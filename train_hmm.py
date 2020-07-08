"""
Changed OOV handling
"""

import sys
import re

from collections import defaultdict

# Files
TAG_FILE = sys.argv[1]
TOKEN_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

# Vocabulary
vocab = {}
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

# Transition and emission probabilities
emissions = {}
transitions = {}
transitions_total = defaultdict(lambda: 0)
emissions_total = defaultdict(lambda: 0)

with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
    for tag_string, token_string in zip(tag_file, token_file):
        tags = re.split("\s+", tag_string.rstrip())
        tokens = re.split("\s+", token_string.rstrip())
        pairs = zip(tags, tokens)

        prevtag = INIT_STATE

        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            if tag not in emissions:
                emissions[tag] = defaultdict(lambda: 0)
            if prevtag not in transitions:
                transitions[prevtag] = defaultdict(lambda: 0)

            if token not in vocab:
                vocab[token] = 1
                emissions[tag][OOV_WORD] += 1
            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissions_total[tag] += 1

            transitions[prevtag][tag] += 1
            transitions_total[prevtag] += 1

            prevtag = tag

        # don't forget the stop probability for each sentence
        if prevtag not in transitions:
            transitions[prevtag] = defaultdict(lambda: 0)

        transitions[prevtag][FINAL_STATE] += 1
        transitions_total[prevtag] += 1

# Write output to output_file
with open(OUTPUT_FILE, "w") as f:
    for prevtag in transitions:
        for tag in transitions[prevtag]:
            f.write("trans {} {} {}\n"
                    .format(prevtag, tag, transitions[prevtag][tag] / transitions_total[prevtag]))

    for tag in emissions:
        for token in emissions[tag]:
            f.write("emit {} {} {}\n"
                    .format(tag, token, emissions[tag][token] / emissions_total[tag]))



