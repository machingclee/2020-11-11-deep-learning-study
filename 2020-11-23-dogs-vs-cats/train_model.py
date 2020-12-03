# usage: python train_model.py --db dataset/kaggle_dogs_vs_cats/hdf5/features.hdf5 --model dogs_vs_cats.pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pyimagesearch.utils import generate_classification_report_from_predictions

import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True)
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=1,
                help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(
    LogisticRegression(solver="lbfgs",
                       multi_class="auto",
                       max_iter=5000
                       ),
    params,
    cv=3,
    n_jobs=args["jobs"])

# I forget to add dataKey="features" in extracting features and by default it is called images in my hdf5 database:
model.fit(db["images"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params))

print("[INFO] evaluating ...")
preds = model.predict(db["images"][i:])

print(classification_report(db["labels"][i:]),
      preds, target_names=db["label_names"])

acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))

f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()
