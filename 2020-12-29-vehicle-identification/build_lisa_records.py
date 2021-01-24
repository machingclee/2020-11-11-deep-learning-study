from config import vehicle_identification_config as config
from deeptools.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import cv2


def main(_):
    f = open(config.CLASSES_FILE, "w")

    for (key, value) in config.CLASSES.items():
        item = (
            "item {\n"
            "\tid: " + str(value) + "\n"
            "\tname: '"+key+"'\n"
            "}\n"
        )
        f.write(item)

    f.close()

    D = {}

    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # skip header:
    for row in rows[1:]:
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        if label not in config.CLASSES.keys():
            continue

        p = os.path.sep.join([config.DATASET_BASE_PATH, imagePath])
        b = D.get(p, [])
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    (trainKeys, testKeys) = train_test_split(list(D.keys()), test_size=config.TEST_SIZE, random_state=42)

    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys,  config.TEST_RECORD)
    ]

    for (dType, keys, outputPath) in datasets:
        writer = tf.io.TFRecordWriter(outputPath)
        total = 0

        for imagePath in keys:
            encoded = tf.io.gfile.GFile(imagePath, "rb").read()
            encoded = bytes(encoded)
            pilImage = Image.open(imagePath)

            (w, h) = pilImage.size[:2]

            filename = imagePath.split(os.path.sep)[-1]
            # e.g., png:
            encoding = filename[filename.rfind(".") + 1:]

            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            for (label, (startX, startY, endX, endY)) in D[imagePath]:
                xMin = float(startX / w)
                xMax = float(endX / w)
                yMin = float(startY / h)
                yMax = float(endY / h)

                # double check the annotation is correct by drawing annotations on images:
                image = cv2.imread(imagePath)
                startX = int(xMin * w)
                startY = int(yMin * h)
                endX = int(xMax * w)
                endY = int(yMax * h)

                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.imshow("Image", image)
                cv2.waitKey(0)

                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                total = total + 1

            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

        writer.close()
        print("[INFO] {} examples saved for '{}'".format(total, dType))


if __name__ == "__main__":
    tf.compat.v1.app.run()
