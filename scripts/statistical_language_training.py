"""This script trains statistical language models to the sequence of states
observed in each scene. In this way, it is believed that the entropy of each
scene will correlate with the number of collisions observed in the scene."""

import numpy as np
import json
import sys
import os.path as op
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline, flatten
from nltk.lm import MLE

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
observed_vocab = []
for scene in scenes:
    for state in scene:
        if state in observed_vocab:
            continue
        else:
            observed_vocab.append(state)

# TODO: Separate out a train and test set. Maybe a validation set?

# Train bigram model with sentence padding.
training_ngrams, padded_scenes = padded_everygram_pipeline(1, scenes)
print(len(list(training_ngrams)))
print(len(list(padded_scenes)))

# Train bigram model without sentence padding.
# TODO: There should be a way to create training_ngrams without sentence
# padding. Look into padded_everygram_pipeline.
training_ngrams = everygrams(scenes[0],max_len=2)
padded_scenes = flatten(scenes)
print(len(list(training_ngrams)))
print(len(list(padded_scenes)))

# TODO: Train maximum likelihood estimator on experiment.

# TODO: Test how well the model predicts on training data. A useful resource:
# https://www.kaggle.com/code/alvations/n-gram-language-model-with-nltk.

# TODO: Test how well the model predicts on test data.

# TODO: Ultimately we want to fit the best model to the data at this point, 
# so we need a way to compare different ngrams.

# TODO: Use the best fit model to compute the entropy rate (or the entropy of 
# each state). Additionally, may want to compute the entropy of the experiment
# or of the scene. This should be analogous to computing the entropy of each 
# sentence or the entropy of a corpus.

# TODO: See if the experiment, scene, or state has an entropy that correlates
# with the number of collisions.