"""
Tagging Accuracy Checker
Modified to get one more argument, and to calculate the differences
"""

import sys
import re

from itertools import zip_longest

# Get files
GOLD_TAGS = sys.argv[1]
MY_TAGS = sys.argv[2]
MY_OWN_TAGS = sys.argv[3]


def evaluate(Y, X):
    # Stats
    num_sentences = 0
    num_sentence_errors = 0
    num_tokens = 0
    num_token_errors = 0

    # Iterate over both files
    gold_tags = []
    with open(Y, "r") as gold_tags, open(X, "r") as my_tags:

        # zip_longest allows us to iterate over the length of the longer list
        for (gold_tag_line, my_tag_line) in zip_longest(gold_tags, my_tags):

            # Terminate loop if more lines in my_tags than gold_tags
            if not gold_tag_line:
                break

            # If missing line, add entire missing line to error num stats
            num_sentences += 1
            if not my_tag_line:
                num_sentence_errors += 1
                gold_tag = re.split("\s+", gold_tags.rstrip())
                num_tokens += len(gold_tag)
                num_token_errors += len(gold_tag)
                continue

            # Otherwise, compare both lines token by token
            sentence_errors = 0
            for (gold_tag, my_tag) in zip_longest(re.split("\s+", gold_tag_line.rstrip()),
                                                  re.split("\s+", my_tag_line.rstrip())):

                # Terminate line if my_tag_line longer than gold_tag_line
                if not gold_tag:
                    break

                num_tokens += 1
                if gold_tag != my_tag:
                    num_token_errors += 1
                    sentence_errors += 1

            if sentence_errors > 0:
                num_sentence_errors += 1
    return num_token_errors, num_tokens, num_sentence_errors, num_sentences


model_1 = evaluate(GOLD_TAGS, MY_TAGS)
model_2 = evaluate(GOLD_TAGS, MY_OWN_TAGS)

file = open("eval.txt", "w")
file.writelines("1. Original\n")
file.writelines("Error rate by word: {} ({} errors out of {})\n".format(model_1[0] / model_1[1], model_1[0], model_1[1]))
file.writelines("Error rate by sentence: {} ({} errors out of {})\n".format(model_1[2] / model_1[3], model_1[2], model_1[3]))
file.writelines("2. My Own HMM Tagger\n")
file.writelines("Error rate by word: {} ({} errors out of {})\n".format(model_2[0] / model_2[1], model_2[0], model_2[1]))
file.writelines("Error rate by sentence: {} ({} errors out of {})\n".format(model_2[2] / model_2[3], model_2[2], model_2[3]))
file.writelines("3. Improvement\n")
file.writelines("Error rate by word: {} improved ({} less errors out of {})\n"
                .format((model_1[0] - model_2[0]) / model_2[1], model_1[0] - model_2[0], model_2[1]))
file.writelines("Error rate by sentence: {} improved ({} less errors out of {})\n"
                .format((model_1[2] - model_2[2]) / model_2[3], model_1[2] - model_2[2], model_2[3]))

file.close()
