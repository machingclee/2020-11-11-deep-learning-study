from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitorCallback (BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitorCallback, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        self.H = {}
        # if self.jsonPath is not None:
        #     if os.path.exists(self.jsonPath):
        #         # "r" is the default mode of open
        #         self.H = json.loads(open(self.jsonPath)).read()

        #         if self.startAt > 0:
        #             for key in self.H.keys():
        #                 self.H[key] = self.H[key][:self.startAt]

    def on_epoch_end(self, epoch, logs):
        # logs has 4 keys: loss, val_loss, accuracy, val_accuracy
        # dictionary.items() return an array of (key, value) pairs
        for(key, value) in logs.items():
            # dictionary.get(key, defaultValue):
            # get the corresponding value of the key, if None, return defaultValue
            l = self.H.get(key, [])
            l.append(float(value))
            self.H[key] = l

        # output json:
        # if self.jsonPath is not None:
        #     file = open(self.jsonPath, "w")
        #     # json.dumps: serialize python dictionary into json string
        #     # json.load: deserialize json string into python object
        #     file.write(json.dumps(self.H))
        #     file.close()

        # output figure:
        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="accuracy")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(
                len(self.H["loss"])-1))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure

            nameSplit = self.figPath.split(os.path.sep)
            fileName = nameSplit[-1].split(".")
            fileName = fileName[0] + "-" + \
                str(len(self.H["loss"])-1)+"."+fileName[1]
            nameSplit[-1] = fileName
            plt.savefig(os.path.sep.join(nameSplit))
            plt.close()
