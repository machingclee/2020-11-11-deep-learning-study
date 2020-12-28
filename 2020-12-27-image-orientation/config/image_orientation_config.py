import os


DATASET_DIR = os.path.sep.join(["C:", "Users", "user", "Repos", "Python", "DeepLearning", "deep-learning-study", "datasets", "indoorCVPR", "images"])
ROTATED_DATASET_DIR = os.path.sep.join(["C:", "Users", "user", "Repos", "Python", "DeepLearning",
                                        "deep-learning-study", "datasets", "indoorCVPR", "rotated_images"])

PRODJECT_DIR = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2])

OUTPUT_DIR = os.path.sep.join([PRODJECT_DIR, "output"])
FEATURES_HDF5 = os.path.sep.join([PRODJECT_DIR, "hdf5", "orientation_features.hdf5"])
BATCH_SIZE = 32
