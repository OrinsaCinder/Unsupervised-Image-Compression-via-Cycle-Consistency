from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
import torchvision.utils as vutils
from torchsummary import summary
import argparse
import sys
from math import log10

from Models import autoencoder
from dataloader import DataloaderCompression

Dataloader = DataloaderCompression('Data_Valid',128,1,4)