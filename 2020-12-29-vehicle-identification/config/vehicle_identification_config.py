import os

DATASET_BASE_PATH = os.path.sep.join(["C:", "Users", "user", "Repos", "Python", "DeepLearning", "deep-learning-study", "datasets", "lisa"])

ANNOT_PATH = os.path.sep.join([DATASET_BASE_PATH, "allAnnotations.csv"])

PROJECT_DIR = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2])

TRAIN_RECORD = os.path.sep.join([PROJECT_DIR, "records", "training.record"])
TEST_RECORD = os.path.sep.join([PROJECT_DIR, "records", "testing.record"])
CLASSES_FILE = os.path.sep.join([PROJECT_DIR, "records", "classes.pbtxt"])

TEST_SIZE = 0.25
CLASSES = {"pedestrianCrossing": 1, "signalAhead": 2, "stop": 3}
