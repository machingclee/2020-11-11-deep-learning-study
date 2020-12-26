from imutils import paths
import numpy as np
import os
import shutil


# D:\datasets\ILSVRC2012
val_dir = os.path.sep.join(["D:", "datasets", "ILSVRC2012", "val"])
val_txt = os.path.sep.join(["D:", "datasets", "ILSVRC2012", "val.txt"])
val_paths = paths.list_images(val_dir)

fs = open(val_txt, "w")
count = 0
for val_path in val_paths:
    count = count + 1
    fileName = val_path.split(os.path.sep)[-1].replace(".JPEG", "")
    print("[INFO] writing {}".format("{} {}".format(fileName, count)))
    fs.write("{} {}\n".format(fileName, count))

fs.close()
