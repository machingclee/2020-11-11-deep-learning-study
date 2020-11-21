from pyimagesearch.utils import rank5_accuracy
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to HDF5 db")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())


def load_model_from_pickle_path(path):
    print("[INFO] loading pretrained model from {}...", path)
    return pickle.loads(open(path, "rb").read())


def read_hdf5_dbfile_from_path(path):
    print("[INFO] loading hdf5 db file from {}...", path)
    return h5py.File(path, "r")


model = load_model_from_pickle_path(args["model"])

db = read_hdf5_dbfile_from_path(args["db"])
i = int(db["labels"].shape[0] * 0.75)

preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

db.close()
