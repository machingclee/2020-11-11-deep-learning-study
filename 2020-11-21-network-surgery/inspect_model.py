from tensorflow.keras.applications import VGG16
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1,
                help="whether or not to include top of CNN")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=args["include_top"] >= 0)

print("[INFO] showing layers...")


def inspect_model(model):
    for(i, layer) in enumerate(model.layers):
        print("[INFO] {}\t{}".format(i, layer.__class__.__name__))


inspect_model(model)
