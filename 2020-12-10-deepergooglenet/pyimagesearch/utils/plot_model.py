from tensorflow.keras.utils import plot_model as plot


def plot_model(model, to_file, show_shapes=True, dpi=96):
    plot(model, to_file=to_file, show_shapes=show_shapes, dpi=dpi)
