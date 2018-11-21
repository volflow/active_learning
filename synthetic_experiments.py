#!/usr/bin/env python
# coding: utf-8

#%load_ext autoreload
#%autoreload 2

import os
import random

import utils.utils as utils
import DCGAN
#from SubsetMNIST import SubsetMNIST
from Networks import SimpleNet
from active_learners import *
from criterions import *

import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn import functional as F

from experiment_scripts import experiment

manual_seed = 1338
random.seed(manual_seed)
torch.manual_seed(manual_seed)
print(torch.__version__)

x_dim = 2
instances_train = 500
p = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))

x1 = np.random.randn(instances_train,x_dim) + [2,2]
y1 = np.ones(instances_train)

x2 = np.random.randn(instances_train,x_dim) + [2,-2]
y2 = np.zeros(instances_train)

x3 = np.random.randn(instances_train,x_dim) + [-2,2]
y3 = np.zeros(instances_train)

x4 = np.random.randn(instances_train,x_dim) + [-2,-2]
y4 = np.ones(instances_train)

instances_val = 50

x1_test = np.random.randn(instances_val,x_dim) + [2,2]
y1_test = np.ones(instances_val)

x2_test = np.random.randn(instances_val,x_dim) + [2,-2]
y2_test = np.zeros(instances_val
             )
x3_test = np.random.randn(instances_val,x_dim) + [-2,2]
y3_test = np.zeros(instances_val)

x4_test = np.random.randn(instances_val,x_dim) + [-2,-2]
y4_test = np.ones(instances_val)


def log_p_gm(X,means=[[2,2],[-2,2],[2,-2],[-2,-2]],stds=[1,1,1,1]):
    sum_p = torch.zeros(len(X),dtype=torch.double)

    X = torch.tensor(X,dtype=torch.double)
    for mean,std in zip(means,stds):
        p = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(mean,dtype=torch.double),
                                                                       torch.eye(2,dtype=torch.double)*std)
        sum_p += torch.exp(p.log_prob(X))
    out = torch.log(sum_p)
    out = torch.tensor(out,dtype=torch.float)
    return out

X = np.concatenate((x1,x2,x3,x4))
z = log_p_gm(X,means=[[2,2],[-2,2],[2,-2],[-2,-2]],stds=[1,1,1,1])
y = np.concatenate((y1,y2,y3,y4))

X_test = np.concatenate((x1_test,x2_test,x3_test,x4_test))
y_test = np.concatenate((y1_test,y2_test,y3_test,y4_test))

X_test = torch.tensor(X_test,dtype=torch.float)
y_test = torch.tensor(y_test,dtype=torch.long)


dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.float),
                                         torch.tensor(y,dtype=torch.long))

dataset_latent = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.float),
                                                torch.tensor(X,dtype=torch.float), #pretend X space is embedded z space
                                                torch.tensor(y,dtype=torch.long))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test,dtype=torch.float),
                                         torch.tensor(y_test,dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test_dataset)

# Plots of Gaussian Mixture

# plt.figure(figsize=(10,10))
# sns.scatterplot(x = X[:,0],y=X[:,1],hue=y,s=10,alpha=0.5)
# plt.show()

# resolution = 150
# # Sample data
# side = np.linspace(-6,6,resolution)
# X_dens,Y_dens = np.meshgrid(side,side)
# Z_dens = log_p_gm(np.dstack((X_dens,Y_dens)).reshape(-1,2)).reshape(resolution,resolution)
# # Plot the density map using nearest-neighbor interpolation
# plt.figure(figsize=(10,10))
# plt.pcolormesh(X_dens,Y_dens,Z_dens,cmap='Greys')
# sns.scatterplot(x = X[:,0],y=X[:,1],size=0.001,legend=False,hue=y)
# plt.show()

def initial_indices(clusters,size=3):
    if clusters == 0:
        return []
    out = torch.randint(0,instances_train,(size,))
    if clusters >= 2:
        out = torch.cat((out,torch.randint(instances_train,instances_train*2,(size,))))
    if clusters >= 3:
        out = torch.cat((out,torch.randint(instances_train*2,instances_train*3,(size,))))
    if clusters >= 4:
        out = torch.cat((out,torch.randint(instances_train*3,instances_train*4,(size,))))
    return out

def cluster_experiments(learner,criterion,name):
    for clusters in range(0,5):
        # clusters/4 clusters initially labeled
        print('-'*8,'instances from {}/4 clusters initially labeled'.format(clusters),'-'*8)


        torch.manual_seed(manual_seed)
        random.seed(manual_seed) # makes sure initially_labeled are euqal in each round
        initially_labeled = [initial_indices(clusters) for i in range(kwargs['rounds'])]

        out = experiment(learner,criterion=criterion,initially_labeled=initially_labeled, **kwargs)

        utils.save_obj((out),result_folder+name+'{}'.format(clusters))

if __name__ == '__main__':
    import sys
    # Hyperparameters for experiments
    kwargs = {
        'model': SimpleNet,
        'model_kwargs': {'hidden_size': 10},
        'train_set': dataset_latent,
        'test_loader': test_loader,
        'rounds': 10,
        'iters': 25,
        'inital_training_epochs': 100,
        're_training_epochs': 100,
        'hard': True,
        'temp': 0,
        'print_freq': 10,
        'plot': False,
        'plot_densfct': log_p_gm,
    }

    kwargs['temp'] = float(sys.argv[1])
    print('Temperature:',kwargs['temp'])
    result_folder = './experiment_results/synthetic_experiments2/temp{}/'.format(kwargs['temp'])
    try:
        os.makedirs(result_folder)
    except FileExistsError:
        pass

    # Random sampling
    cluster_experiments(RandomActiveLearner,None,name='rand_learner')

    # # entropy
    # cluster_experiments(SearchActiveLearner,HLoss(),name='entr_learner')
    #
    #
    # #DensityEntropy
    # cluster_experiments(SearchActiveLearner,DensityWeightedUncertainty(density=log_p_gm),name='entr_dens_learner')
    #
    #
    # #Diversity
    # cluster_experiments(DiverstiyDensityUncertaintyActiveLearner,Diversity(),name='div_learner')
    #
    #
    # #DiversityDensity
    # cluster_experiments(DiverstiyDensityUncertaintyActiveLearner,
    #                     DiversityDensity(density=log_p_gm),
    #                     name='dens_div_learner')
    #
    #
    # #DiversityDensityUncertainty
    # cluster_experiments(DiverstiyDensityUncertaintyActiveLearner,
    #                     DiversityDensityUncertainty(density=log_p_gm),
    #                     name='entr_dens_div_learner')
