#!/usr/bin/env python

# enable to import from modules in the qtl-covid-classification folder when running from the command line
import argparse
import os
import random
import math
import pickle
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm

import src.utils
from src.model import MODEL
from src.utils import *

def lists_to_numbers(input):
    list_int = []
    for lst in input:
        list_int.append(int(''.join(map(str, lst)), 2))
    return list_int

def draw_plot(points, bins, filename):
    plt.figure()
    counts, borders = np.histogram(points, bins=bins, range=(0, 2097151))
    counts_norm = counts / sum(counts)
    plt.ylim(0.0, 1.0)
    plt.bar(x=range(bins), height=counts_norm, edgecolor='black', linewidth=2, width=0.6)
    xticks = []
    #start = str(int(borders[0]))
    start = 0
    start_num = "{:.2}e6".format(start / 1e6)
    for end in borders[1:-1]:
            end_num ="{:.2}e6".format((math.ceil(end)-1)/ 1e6)
            xticks.append(start_num + " - \n" + end_num)
            start = math.ceil(end)
            start_num = "{:.2}e6".format(start/ 1e6)
    end_num = "{:.2}e6".format(2097151/ 1e6)
    xticks.append(start_num + " - \n" + end_num)
    plt.xticks(ticks=range(bins), labels=xticks)
    plt.ylabel('proportion')

    plt.savefig(filename + '.pdf')
    plt.show()

DATASET_PATH = 'src/datasets/l_o7_c5_d3_p200_v1.npy'

seed = 77

random.seed(seed)
np.random.seed(seed)
print("Seed is " + str(seed))

bins = 7

dataset = import_dataset(DATASET_PATH)
data, labels = split_dataset_labels(dataset)
encoded_data, bits_input_vector, num_features = MODEL.binary_encode_data(data, use_folding=True)
encoded_data_int = lists_to_numbers(encoded_data.astype(int))
# max value with 21 bit is 2097151
draw_plot(encoded_data_int, bins, "probs_dataset")

samples = pickle.load(open('Advantage_system4.1_samples.pickle', 'rb'))
samples_list = [list(sample.values())[:21] for sample in samples]
samples_num = lists_to_numbers(samples_list)
draw_plot(samples_num, bins, "probs_samples")
print(samples)