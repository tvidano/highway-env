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
experiment = "experiment_006"
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "..",)
experiments_dir = op.join(local_highway_env, "scripts",
                          "sensor_scheduling_experiments")
path = op.join(experiments_dir, experiment, "raw_data.json")
with open(path, "r") as f:
    exp = json.load(f)
    state_record = exp["state_record"]
    collisions_count_record = exp["collisions_count"]

# Extract scenes from state record.
scenes = list(state_record.values())

# Convert states from integers to strings to support nltk models.
# tokenized_scenes = [[str(state) for state in scene] for scene in scenes]
tokenized_state_record = {}
for seed, record in state_record.items():
    if len(record) < 10:
        collisions_count_record.pop(seed)
        continue
    tokenized_state_record[int(seed)] = [str(state) for state in record]

# Separate out test and train scenes. Assume no correlation in order of scenes.
start_seed = min(tokenized_state_record.keys())
cutoff_seed = int(0.9 * len(state_record) + start_seed)
train_record = {seed: scene for seed, scene in tokenized_state_record.items()
                if seed <= cutoff_seed}
test_record = {seed: scene for seed, scene in tokenized_state_record.items()
               if seed > cutoff_seed}
# i_split = int(np.floor(len(tokenized_scenes)*0.9))
# train_scenes = tokenized_scenes[:i_split]
# test_scenes = tokenized_scenes[i_split:]

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
# models = [Laplace, WittenBellInterpolated, AbsoluteDiscountingInterpolated]
# model_names = ["Laplace", "WittenBell", "AbsoluteDiscounting"]
# ns = np.arange(1, 6)
# plt.figure()
# for model, name in zip(models, model_names):
#     for n in ns:
#         flat_train_scenes = flatten(train_record.values())
#         training_ngrams = [everygrams(list(scene), max_len=n)
#                            for scene in train_record.values()]
#         training_ngrams.append((tuple("0"), ))
#         m = model(n)
#         m.fit(training_ngrams, flat_train_scenes)
#         test_ngrams = unpacked_ngrams(test_record.values(), n)
#         perplexity = m.perplexity(test_ngrams)
#         print(f"{name},{n}: {perplexity}")
#         plt.scatter(n, perplexity, label=name+':'+str(n))
# plt.legend()
# plt.show()

# Use the best fit model to compute the entropy rate (or the entropy of
# each state). Additionally, may want to compute the entropy of the experiment
# or of the scene. This should be analogous to computing the entropy of each
# sentence or the entropy of a corpus.
n = 7
model = AbsoluteDiscountingInterpolated(n)
flat_scenes = flatten(tokenized_state_record.values())
ngrams = [everygrams(list(scene), max_len=n)
          for scene in tokenized_state_record.values()]
ngrams.append((tuple("0"), ))
model.fit(ngrams, flat_scenes)
scene_entropy_rate = {}
for seed, test_scene in tokenized_state_record.items():
    scene_entropy = []
    for i, state in enumerate(test_scene):
        scene_entropy.append(-model.logscore(state, test_scene[:i]))
    scene_entropy_rate[seed] = np.mean(scene_entropy)
plt.scatter(collisions_count_record.values(), scene_entropy_rate.values())
plt.show()

# TODO: It appears that there is no obvious correlation between scene
# entropy rate and the number of collisions. It also appears that there is
# little correlation between the entropy of the next state and if that next
# state produces a collision with the ego. However, this needs to be better
# investigated to be ruled out. Intuitively, this scheme should predict ego
# vehicle collisions, however, it may not predict collisions in general.
# We can try a few things now:
#   1. Investigate step-by-step entropy as a predictor of ego collisions.
#   2. Eliminate the first 10 or so states using the assumption that the
#   simulation doesn't resemble a real road until then. Retry scene-by-scene
#   entropy rate as a predictor of ego collisions. Then retry step-by-step
#   entropy as a predictor.
#   3. Train the language model on all experiments to capture probabilities of
#   general driving. Then use that model to measure entropy, step-by-step or
#   scene-by-scene.
