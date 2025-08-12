#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from time import time
import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.rbm import RBM
from src.utils import *

parser = argparse.ArgumentParser(
    description='Generate clustered datasets with outliers.')
parser.add_argument('-hn', '--hnodes',
                    metavar='INT',
                    help='Amount of hidden units for RBM model',
                    default=157,
                    type=int)

parser.add_argument('-e', '--epochs',
                    metavar='INT',
                    help='Epochs for training',
                    default=13,  # best fit for dataset l_o7_c5_d3_p200_v1
                    type=int)

parser.add_argument('-b', '--batchsize',
                    metavar='INT',
                    help='Batchsize for training',
                    default=10,  # best fit for dataset l_o7_c5_d3_p200_v1
                    type=int)

parser.add_argument('-s', '--seed',
                        metavar='INT',
                        help='Seed for RNG',
                        default=77,  # best fit for dataset l_o7_c5_d3_p200_v1
                        type=int)

flags = parser.parse_args(['-hn','6','-b','7','-e','16'])
########## CONFIGURATION ##########
EPOCHS = flags.epochs
HIDDEN_UNITS = flags.hnodes
CD_K = 2
SEED = flags.seed
QUANTILE = 0.95
BATCH_SIZE = flags.batchsize

CLUSTER=5
DATASET_PATH= 'C:/Users/ge84gac/datasets/good_datasets/l_o8_c5_d2_v0.35_p190_4.npy'


start = time()
########## LOADING CUSTOM DATA ##########
li=list(np.arange(0,200,1))
s=random.sample(li,k=15)

fs=[]
rs=[]
ps=[]
for i in s:
    np.random.seed(i)
    random.seed(i)

    data = import_dataset(DATASET_PATH)
    # print(np.amin(data)) #TODO: Fix bug negative datapoints for variance 1.0 in generator

    training_dataset, testing_dataset = split_data(data, CLUSTER)

    training_data, training_labels = split_dataset_labels(training_dataset)

    testing_data,testing_labels=split_dataset_labels(testing_dataset)

    ########## TRAINING RBM ##########
    print('Training RBM...')

    rbm = RBM(training_data, HIDDEN_UNITS, CD_K, SEED, epochs=EPOCHS,momentum_coefficient=0.5276710798696586, trained=False, quantile=QUANTILE)

    rbm.train_model(BATCH_SIZE,learning_rate=0.7028980904704654)

    #save_weights(model = rbm, weight_csv='weights', title="Weights", type_quantum=False)

    ########## GENERATING CUSTOM TRAIN DATA FOR LOGISTIC REGRESSION ##########

    tensor_training_data = torch.from_numpy(RBM.binary_encode_data(training_data)[0])
    tensor_training_labels = torch.from_numpy(np.array(training_labels))

    tensor_testing_data = torch.from_numpy(RBM.binary_encode_data(testing_data)[0])
    tensor_testing_labels = torch.from_numpy(np.array(testing_labels))



    batches_training_data = tensor_training_data.split(BATCH_SIZE)
    batches_training_label = tensor_training_labels.split(BATCH_SIZE)

    batches_training = list(zip(batches_training_data, batches_training_label))



    ########## EXTRACTING OUTLIER AND CLUSTERPOINTS ##########

    outliers = RBM.get_binary_outliers(
        dataset=testing_dataset, outlier_index=CLUSTER)
    points = RBM.get_binary_cluster_points(
        dataset=testing_dataset, cluster_index=CLUSTER-1)


    ########## ENERGY COMPARISON ##########
    print('Energy comparison...')

    outlier_energy = []

    for outlier in outliers:
        outlier = torch.from_numpy(np.reshape(outlier, (1, rbm.num_visible)))
        outlier_energy.append(rbm.free_energy(outlier).cpu().numpy().tolist())

    outlier_energy = np.array(outlier_energy)

    cluster_point_energy = []

    for point in points:
        point = torch.from_numpy(np.reshape(point, (1, rbm.num_visible)))
        cluster_point_energy.append(rbm.free_energy(point).cpu().numpy().tolist())

    cluster_point_energy = np.array(cluster_point_energy)

    o = outlier_energy.reshape((outlier_energy.shape[0]))
    c = cluster_point_energy.reshape((cluster_point_energy.shape[0]))

    #RBM.plot_energy_diff([o, c], rbm.outlier_threshold, "rbm_energies.pdf")

    #RBM.plot_hist(c, o, rbm.outlier_threshold, "rbm_hist.pdf")


    ########## OUTLIER CLASSIFICATION ##########
    print('Outlier classification...')

    predict_points = np.zeros(len(tensor_testing_data), dtype=int)

    for index, point in enumerate(tensor_testing_data.split(1),0):
        point = point.view(1, rbm.num_visible)
        predict_points[index], _ = rbm.predict_point_as_outlier(point)

    print("Predicted points test: ", predict_points)
    true_points = np.where(testing_labels < CLUSTER, 0, 1)
    accuracy, precision, recall = accuracy_score(true_points, predict_points), precision_score(true_points,predict_points), recall_score(true_points, predict_points)
    f1 = f1_score(true_points, predict_points)
    tn, fp, fn, tp = confusion_matrix(true_points, predict_points, labels=[0, 1]).ravel()

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, \nNum True Negative: {tn}, Num False Negative: {fn}, Num True Positive: {tp}, Num False Positive: {fp}')

    end = time()
    print(f'Wallclock time: {(end-start):.2f} seconds')
    fs.append(f1)
    rs.append(recall)
    ps.append(precision)

print(fs)
print(rs)
print(ps)
print(s)
'''
print(rbm.outlier_threshold)
print(np.max(c),np.min(c),np.max(o),np.min(o))
data = {"QBM":[]}
data['QBM']=[np.load('C:/Users/ge84gac/Gate_qbm/tests/cs.npy'),np.load('C:/Users/ge84gac/Gate_qbm/tests/os.npy ')]

data["RBM"] = [c, o]

thresholds = [-27.948, rbm.outlier_threshold]

keys = ["QBM", "RBM"]



thresholds = list(thresholds)
fig = plt.figure()
fig.suptitle('Point Energy', fontsize=14, fontweight='bold')

ax = fig.add_subplot()

xlabels = []
i = 0
for _ in keys:
    xlabels.append("cluster")
    xlabels.append("outlier")
    i += 2



raw_data = []
index = 0
box_index = 0
for key, (cluster, outlier) in data.items():
    if key in keys:
        mean = np.linalg.norm(np.concatenate([cluster,outlier]))
        raw_data.append(cluster*10/mean)
        raw_data.append(outlier*10/mean)
        thresholds[index] = thresholds[index]*10/mean
        xmin = (box_index)/int(len(keys)*2)
        xmax = (box_index+2)/int(len(keys)*2)
        ax.axhline(y=thresholds[index], xmin=xmin, xmax=xmax)
        box_index += 2
    index += 1

box = ax.boxplot(raw_data, showfliers=True, showmeans=True, vert=True, patch_artist=True)

boxes = []
i = 0
for _ in keys:
    boxes.append(box["boxes"][i])
    i += 2

colors = ['pink', 'lightblue']
colors = colors[:len(keys)]
colors = [ele for ele in colors for i in range(2)]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)


ax.legend(boxes, keys, loc='upper right')

ax.set_ylabel('Energy')
ymin=-0.8
ymax=0.2
plt.ylim(ymin, ymax)
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(xlabels, fontsize=8)
plt.show()
'''