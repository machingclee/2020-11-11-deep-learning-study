from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.preprocessing import MeanSubtractionPreprocessor
from pyimagesearch.preprocessing import CropsPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils import rank5_accuracy
from tensorflow.keras.models import load_model

import numpy as np
import json

means = json.loads(open(config.DATASET_MEAN_JSON).read())
