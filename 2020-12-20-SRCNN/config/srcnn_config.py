import os
import numpy as np
from numpy.core.defchararray import find

INPUT_IMAGES = "ukbench100"

PROJECT_DIR = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2])

IMAGES = os.path.sep.join([PROJECT_DIR, "output", "images"])
LABELS = os.path.sep.join([PROJECT_DIR, "output", "labels"])

INPUTS_DB = os.path.sep.join([PROJECT_DIR, "output", "inputs.hdf5"])
OUTPUTS_DB = os.path.sep.join([PROJECT_DIR, "output", "outputs.hdf5"])

MODEL_PATH = os.path.sep.join([PROJECT_DIR, "output", "srcnn.model"])
PLOT_PATH = os.path.sep.join([PROJECT_DIR, "output", "plot.png"])

IMAGES_DATA_DIR = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-3] + ["datasets", "ukbench", "ukbench100"])
