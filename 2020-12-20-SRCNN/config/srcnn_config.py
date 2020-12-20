import os
import numpy as np
from numpy.core.defchararray import find

IMAGES_DATA_DIR = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-3] + ["datasets", "ukbench", "ukbench100"])
PROJECT_DIR = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2])

IMAGES = os.path.sep.join([PROJECT_DIR, "output", "images"])
TARGETS = os.path.sep.join([PROJECT_DIR, "output", "targets"])
INPUTS_DB = os.path.sep.join([PROJECT_DIR, "output", "inputs.hdf5"])
TARGETS_DB = os.path.sep.join([PROJECT_DIR, "output", "targets.hdf5"])
MODEL_PATH = os.path.sep.join([PROJECT_DIR, "output", "srcnn.model"])
PLOT_PATH = os.path.sep.join([PROJECT_DIR, "output", "plot.png"])

BATCH_SIZE = 128
NUM_EPOCHS = 10

SCALE = 2.0
INPUT_DIM = 33

LABEL_SIZE = 21
PAD = int((INPUT_DIM-LABEL_SIZE)/2)
STRIDE = 14
