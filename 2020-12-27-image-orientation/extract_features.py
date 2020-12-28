# purpose: save batches of features and batches of labels

from config import image_orientation_config as config
from deeptools.utils.extract_batches_of_features import extract_batches_of_features

# structure of dataset must follow the pattern:
# dataset/class/images.jpg

extract_batches_of_features(
    dataset_dir=config.ROTATED_DATASET_DIR,
    output_path=config.FEATURES_HDF5,
    batch_size=32
)
