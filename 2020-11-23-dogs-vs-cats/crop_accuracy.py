from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.preprocessing import MeanSubtractionPreprocessor
from pyimagesearch.preprocessing import CropsPreprocessor
from pyimagesearch.utils import generate_progressbar
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils import rank5_accuracy
from tensorflow.keras.models import load_model

import numpy as np
import json

means = json.loads(open(config.DATASET_MEAN_JSON).read())

resize_pp = ResizePreprocessor(227, 227)
meanSubtraction_pp = MeanSubtractionPreprocessor(means["R"],
                                                 means["G"],
                                                 means["B"])
crop_pp = CropsPreprocessor(227, 227)
imgToArray_pp = ImageToArrayPreprocessor()

model = load_model(config.MODEL_PATH)

print("[INFO] predicting on test data (no crops)")
testGen = HDF5DatasetGenerator(
    config.TEST_HDF5,
    64,
    preprocessors=[resize_pp, meanSubtraction_pp, imgToArray_pp])

# array of batched predictions:
# it seems that model.predict helps us extend the list of predictions for each batch of the result, the shape of prediction is finally (2496, 2)
predictions = model.predict_generator(
    testGen.generator(),
    steps=testGen.numOfImages//64, max_queue_size=10)


(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()

testGen = HDF5DatasetGenerator(
    config.TEST_HDF5, 64,
    preprocessors=[meanSubtraction_pp], numOfClasses=2)

predictions = []

pbar = generate_progressbar(
    title="Evaluating: ", maxval=testGen.numOfImages//64)

for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    for image in images:
        crops = crop_pp.preprocess(image)
        # strange enough, img_to_array function in keras can also accept np.array as input
        crops = np.array([imgToArray_pp.preprocess(crop)
                          for crop in crops], dtype="float32")

        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))

    pbar.update(i)

pbar.finish()
print("[INFO] predicting on test data with crops ...")
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
