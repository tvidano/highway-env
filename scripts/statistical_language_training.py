"""This script trains statistical language models to the sequence of states
observed in each scene. In this way, it is believed that the entropy of each
scene will correlate with the number of collisions observed in the scene."""

import numpy as np
import json
import os.path as op
from nltk.util import everygrams, bigrams
from nltk.lm.preprocessing import flatten
from nltk.lm import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Useful resources:
# https://web.stanford.edu/~jurafsky/slp3/3.pdf
# https://www.kaggle.com/code/alvations/n-gram-language-model-with-nltk.


def unpacked_ngrams(text, n):
    """Returns ngrams as a list of ngrams (tuples) from |text|."""
    unpacked_ngrams = []
    for sentence in text:
        ngrams = list(everygrams(sentence, max_len=n))
        for ngram in ngrams:
            unpacked_ngrams.append(ngram)
    return unpacked_ngrams


# Collect the states from the state record of an experiment.
experiment = "experiment_009"
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "..",)
experiments_dir = op.join(local_highway_env, "scripts",
                          "sensor_scheduling_experiments")
path = op.join(experiments_dir, experiment, "raw_data.json")
with open(path, "r") as f:
    exp = json.load(f)
    state_record = exp["state_record"]

# Extract scenes from state record.
scenes = list(state_record.values())

# Convert states from integers to strings to support nltk models.
tokenized_scenes = [[str(state) for state in scene] for scene in scenes]

# Separate out test and train scenes. Assume no correlation in order of scenes.
i_split = int(np.floor(len(tokenized_scenes)*0.9))
train_scenes = tokenized_scenes[:i_split]
test_scenes = tokenized_scenes[i_split:]

# # Construct ngrams and flattened scenes iterators.
# n = 3
# training_ngrams = [everygrams(list(scene), max_len=n)
#                    for scene in train_scenes]
# # Add a word outside the dictionary so that "<UNK>" is seen once. This prevents
# # discounting models from assigning inf perplexity to data containing unseen
# # words.
# training_ngrams.append((tuple("0"), ))
# flat_train_scenes = flatten(train_scenes)

# # Train maximum likelihood estimator on experiment.
# # KneserNeyInterpolated fails for some reason. The fix above solves the inf
# # perplexity problem for the other interpolated models, but not this one. It
# # takes a very long time to compute perplexity so it is likely that there is
# # some issue with how I'm trying to use it, or in how it is implemented.
# model = KneserNeyInterpolated(n)
# model.fit(training_ngrams, flat_train_scenes)
# print("\n" + "#"*10 + "Summary" + "#"*10)
# print(f'Length of vocab: {len(model.vocab)}')
# print(f'Model.counts = {model.counts}')
# print(f'The probability of "<UNK>" is: {model.score("<UNK>")}')

# # Test how well the model predicts on training data.
# unpacked_train_ngrams = unpacked_ngrams(train_scenes, n)
# print(f'Training perplexity: {model.perplexity(unpacked_train_ngrams)}')

# # Test how well the model predicts on test data.
# test_ngrams = unpacked_ngrams(test_scenes, n)
# print(f'Testing perplexity: {model.perplexity(test_ngrams)}')

# Compare different models.
models = [Laplace, WittenBellInterpolated, AbsoluteDiscountingInterpolated]
model_names = ["Laplace", "WittenBell", "AbsoluteDiscounting"]
ns = np.arange(1, 4)
plt.figure()
for model, name in zip(models, model_names):
    for n in ns:
        flat_train_scenes = flatten(train_scenes)
        training_ngrams = [everygrams(list(scene), max_len=n)
                           for scene in train_scenes]
        training_ngrams.append((tuple("0"), ))
        m = model(n)
        m.fit(training_ngrams, flat_train_scenes)
        test_ngrams = unpacked_ngrams(test_scenes, n)
        perplexity = m.perplexity(test_ngrams)
        print(f"{name},{n}: {perplexity}")
        plt.scatter(n, perplexity, label=name+':'+str(n))
plt.legend()
plt.show()

# TODO: Use the best fit model to compute the entropy rate (or the entropy of
# each state). Additionally, may want to compute the entropy of the experiment
# or of the scene. This should be analogous to computing the entropy of each
# sentence or the entropy of a corpus.

# TODO: See if the experiment, scene, or state has an entropy that correlates
# with the number of collisions.
