from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import ResizePreprocessor
from pyimagesearch.preprocessing import RandomSingleCropPreprocessor
from pyimagesearch.preprocessing import CropsPreprocessor
from pyimagesearch.preprocessing import MeanSubtractionPreprocessor
from pyimagesearch.callbacks import TrainingMonitorCallback
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import json
import os
import matplotlib

# os.system("export KERAS_BACKEND=plaidml.keras.backend")

matplotlib.use("Agg")
batchSize = 128
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest"
                         )

means = json.loads(open(config.DATASET_MEAN_JSON).read())

resize_pp = ResizePreprocessor(227, 227)
randomSingleCrop_pp = RandomSingleCropPreprocessor(227, 227)
meanSubtraction_pp = MeanSubtractionPreprocessor(means["R"],
                                                 means["G"],
                                                 means["B"])
imgToArray_pp = ImageToArrayPreprocessor()


trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5,
                                batchSize,
                                aug=aug,
                                preprocessors=[randomSingleCrop_pp,
                                               meanSubtraction_pp,
                                               imgToArray_pp],
                                numOfClasses=2
                                )

valGen = HDF5DatasetGenerator(config.VAL_HDF5,
                              batchSize,
                              preprocessors=[resize_pp,
                                             meanSubtraction_pp,
                                             imgToArray_pp],
                              numOfClasses=2
                              )


opt = Adam(lr=1e-3)
model = AlexNet.build(width=227,
                      height=227,
                      depth=3,
                      numOfClasses=2,
                      reg=0.002)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

path = os.path.sep.join([config.OUTPUT_PATH,
                         "{}.png".format(os.getpid())
                         ])

callbacks = [TrainingMonitorCallback(path)]

model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numOfImages // batchSize,
    validation_data=valGen.generator(),
    validation_steps=valGen.numOfImages // batchSize,
    epochs=75,
    max_queue_size=10,
    callbacks=callbacks,
    verbose=1
)

model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()
