#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


p='C:/Users/ge84gac/Anomoly_Detection_Annealing/src/datasets/l_o7_c5_d3_p200_v1.npy'


def draw_test_dataset(data,cluster,color=True):
    
    ''' Draws test dataset with detected outlier points.
        If there are more than two dimensions a scatterplot will be drawn.
        Additionally tells you what is currently doing since plotting sometimes
        might take a while depending on the size of your dataset.
    '''
    
    
    dims = data.shape[1]-1

    if color==True:
        
        
        clusterindex=cluster-1
        cols = [f'{i}' for i in range(1, dims+1)]
        cols.append('label')

        # .astype({'label':str}, errors='raise')
        frame = pd.DataFrame(data, columns=cols)
        clusterpoints = frame['label'] <= clusterindex
        outliers = frame['label'] > clusterindex

        frame['label'][clusterpoints] = "cluster"
        frame['label'][outliers] = "outlier"

    else:
        data = data[:, :-1]
        frame = pd.DataFrame(data)

    if dims < 2:
        print(
            f'Dataset "{path.name}" has less than two dimensions and is therefore skipped.')
        return
    else:
        print(f'Currently drawing test dataset')
        if dims == 2:
            frame.plot.scatter(0, 1, title='')
        else:
            if color==True:
                g = sns.pairplot(frame, hue="label", aspect=2)
                
                


               

                g.set(xticks=range(0, 127, 20), xmargin=-0.15)
            else:
                pd.plotting.scatter_matrix(frame, alpha=1, diagonal='kde')
            # plt.gcf().suptitle(f'{path.name}')

        #gpath = path.parent / 'graphics'
        #gpath.mkdir(mode=0o770, exist_ok=True)
        #gpath /= f'{path.with_suffix(".jpeg").name}'
        #plt.savefig(gpath)
    
    

def draw_dataset(path: Path, color=True,dataset=None ):
    ''' Will draw a dataset.
        If there are more than two dimensions a scatterplot will be drawn.
        Additionally tells you what is currently doing since plotting sometimes
        might take a while depending on the size of your dataset.
    '''
    
    data = np.load(path)
    dims = data.shape[1]-1

    if color==True:
        params = path.stem.split("_")
        clusterindex = int(params[2][1:])-1

        cols = [f'{i}' for i in range(1, dims+1)]
        cols.append('label')

        # .astype({'label':str}, errors='raise')
        frame = pd.DataFrame(data, columns=cols)
        clusterpoints = frame['label'] <= clusterindex
        outliers = frame['label'] > clusterindex

        frame['label'][clusterpoints] = "cluster"
        frame['label'][outliers] = "outlier"

    else:
        data = data[:, :-1]
        frame = pd.DataFrame(data)

    if dims < 2:
        print(
            f'Dataset "{path.name}" has less than two dimensions and is therefore skipped.')
        return
    else:
        print(f'Currently drawing {path.stem}.')
        if dims == 2:
            frame.plot.scatter(0, 1, title=f'{path.name}')
        else:
            if color==True:
                g = sns.pairplot(frame, hue="label", aspect=2)
                g.set(xticks=range(0, 127, 20), xmargin=-0.15)
            else:
                pd.plotting.scatter_matrix(frame, alpha=1, diagonal='kde')
            # plt.gcf().suptitle(f'{path.name}')

        gpath = path.parent / 'graphics'
        gpath.mkdir(mode=0o770, exist_ok=True)
        gpath /= f'{path.with_suffix(".jpeg").name}'
        plt.savefig(gpath)


def draw_dir(path: Path, flags: argparse.Namespace):
    '''Iteratively calls draw_dataset() on each file of type ".npy"
    '''
    for f in path.glob('*.npy'):
        draw_dataset(f, flags)

