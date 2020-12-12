import numpy as np
import json
import os

JSON_PATH = os.path.sep.join(["output", "exp-6.3-training.json"])


log = json.loads(open(JSON_PATH).read())
val_acc = log["val_accuracy"]
# val_max_index = np.argmax(val_acc)
print(val_acc[74])
