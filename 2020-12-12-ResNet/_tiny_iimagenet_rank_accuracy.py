import os
from config import tinyimagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.preprocessing import MeanSubtractionPreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.io import HDF5DatasetGenerator
from tensorflow.keras.models import load_model
import json

FINAL_MODEL_PATH = os.path.sep.join(["output", "checkpoints", "ResNet-TinyImagenet-200-4-epoch-75.hdf5"])


means = json.loads(open(config.MEAN_JSON_PATH).read())

resize_pp = ResizePreprocessor(64, 64)
mean_subtraction_pp = MeanSubtractionPreprocessor(means["R"], means["G"], means["B"])
img_to_arr_pp = ImageToArrayPreprocessor()

test_generator = HDF5DatasetGenerator(config.TEST_HDF5_PATH, 64,
                                      preprocessors=[resize_pp, mean_subtraction_pp, img_to_arr_pp],
                                      n_classes=200)

if FINAL_MODEL_PATH is not None:
    model = load_model(FINAL_MODEL_PATH)
    preds = model.predict(test_generator.generator(),
                          steps=test_generator.numOfImages//64,
                          max_queue_size=10
                          )
    (rank1, rank5) = rank5_accuracy(preds=preds, labels=test_generator.db["labels"])
    print("[INFO] rank-1: {:.2f}%".format(rank1*100))
    print("[INFO] rank-5: {:.2f}%".format(rank5*100))

    test_generator.close()
