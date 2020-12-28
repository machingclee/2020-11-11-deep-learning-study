# import the necessary packages
from tensorflow.keras.callbacks import Callback
import os


class EpochCheckpoint(Callback):
    def __init__(self, output_dir, every=5, startAt=0, model_title=""):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.output_dir = output_dir
        self.every = every
        self.intEpoch = startAt
        self.model_title = model_title

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_dir, self.model_title + "epoch-{}.hdf5".format(self.intEpoch + 1)])
            self.model.save(p, overwrite=True)

        # increment the internal epoch counter
        self.intEpoch += 1
