from imutils import paths
import numpy as np
import os


class ImagenetHelper:
    def __init__(self, config):
        self.config = config

        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()

    def buildClassLabels(self):
        rows = open(self.config.WORD_IDS).read().strip().split("\n")
        labelMappings = {}
        for row in rows:
            (wordID, label, hrLabel) = row.split(" ")
            labelMappings[wordID] = int(label) - 1
        return labelMappings

    def buildBlacklist(self):
        rows = open(self.config.VAL_BLACKLIST).read()
        rows = set(rows.strip().split("\n"))

        return rows

    def buildTrainingSet(self):
        imagePaths = paths.list_images(os.path.sep.join([self.config.IMAGES_DIR, "train"]))
        imgPaths = []
        labels = []

        for imagePath in imagePaths:
            wordID = imagePath.split(os.path.sep)[-2]
            # this is an integer:
            label = self.labelMappings[wordID]
            imgPaths.append(imagePath)
            labels.append(label)

        return (np.array(imgPaths), np.array(labels))

    def buildValidationSet(self):
        paths = []
        labels = []

        valFilenames = open(self.config.VAL_LIST).read()
        valFilenames = valFilenames.strip().split("\n")

        valLabels = open(self.config.VAL_LABELS).read()
        valLabels = valLabels.strip().split("\n")

        for (row, label) in zip(valFilenames, valLabels):
            (partialPath, imageNum) = row.strip().split(" ")

            if imageNum in self.valBlacklist:
                continue

            path = os.path.sep.join([self.config.IMAGES_DIR, "val", "{}.JPEG".format(partialPath)])
            paths.append(path)
            labels.append(int(label) - 1)

        return (np.array(paths), np.array(labels))
