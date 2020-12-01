import progressbar


def generate_progressbar(title="Processing...", maxval=1):
    widgets = [title,
               progressbar.Percentage(),
               " ",
               progressbar.Bar(),
               " ",
               progressbar.ETA()]

    pbar = progressbar.ProgressBar(maxval=maxval, widgets=widgets).start()
<<<<<<< HEAD

=======
>>>>>>> d1c66f79e2fd5d7eb55a8a98e3861c1c92c98f9b
    return pbar
