import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn import functional as F


def save_model(fp, model, optimizer, losses, epoch):
    # Save model
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'losses': G_losses,
        'epoch':  epoch,
    }
    print("saving model")
    torch.save(state, fp)
    return

def load_model_dict(fp, map_location=None):
    if os.path.isfile(fp):
        print("=> loading checkpoint '{}'".format(fp))
        model_dict = torch.load(
            fp, map_location=lambda storage, loc: storage)
        return model_dict
    else:
        print("=> no checkpoint found at '{}'".format(fp))
        return None

def load_model(model, model_dict, optimizer=None):
    model.load_state_dict(model_dict['state_dict'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer'])
    epoch = model_dict['epoch']
    losses = model_dict['losses']
    print("model loaded from checkpoint")
    return losses, epoch

import pickle
def save_obj(obj,fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(fn, verbose=False):
    if verbose:
        print('loading object from {}'.format(fn))
    with open(fn, 'rb') as handle:
        return pickle.load(handle)
