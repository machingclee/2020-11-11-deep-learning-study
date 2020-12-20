from config import srcnn_config as config
from deeptools.io import HDF5DatasetGenerator
from deeptools.nn.conv import SRCNN
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


def super_res_generator(inputData_gen, targetData_gen):
    while True:
        inputData = next(inputData_gen)[0]
        targetData = next(targetData_gen)[0]

        yield(inputData, targetData)


inputs_gen = HDF5DatasetGenerator(config.INPUTS_DB, config.BATCH_SIZE)
targets_gen = HDF5DatasetGenerator(config.TARGETS_DB, config.BATCH_SIZE)

print("[INFO] compiling model...")
opt = Adam(lr=0.001, decay=0.001/config.NUM_EPOCHS)
model = SRCNN.build(width=config.INPUT_DIM, height=config.INPUT_DIM, depth=3)
# no accuracy is needed, we don't need to record it
model.compile(loss="mse", optimizer=opt)

H = model.fit(
    super_res_generator(inputs_gen.generator(), targets_gen.generator()),
    steps_per_epoch=inputs_gen.numOfImages//config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1
)

model.save(config.MODEL_PATH, overwrite=True)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"], label="loss")
plt.title("Loss on super Resolution Training")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(config.PLOT_PATH)

inputs_gen.close()
targets_gen.close()
