import matplotlib.pyplot as plt
import numpy as np


def plot_training_graph(
        H,
        saveFilePath,
        epochs,
        title="Training Loss and Accuracy",
        xlabel="Epoch Number",
        ylabel="Loss/Accuracy"):

    plt.style.use("ggplot")
    plt.figure()

    for key in H.history.keys():
        plt.plot(np.arange(0, epochs), H.history[key], label=key)
    # plt.plot(np.arange(0, epochs),
    #          H.history["loss"],
    #          label="train_loss")
    # plt.plot(np.arange(0, epochs),
    #          H.history["val_loss"],
    #          label="val_loss")
    # plt.plot(np.arange(0, epochs),
    #          H.history["accuracy"],
    #          label="train_accuracy")
    # plt.plot(np.arange(0, epochs),
    #          H.history["val_accuracy"],
    #          label="val_accuracy")

    plt.title("Training Loss and Accuracy for model -{}".format(title))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(saveFilePath)
    plt.close()
