import progressbar


def generate_progressbar(title="Processing...", maxval=1):
    widgets = [title,
               progressbar.Percentage(),
               " ",
               progressbar.Bar(),
               " ",
               progressbar.ETA()]

    pbar = progressbar.ProgressBar(maxval=maxval, widgets=widgets).start()
    return pbar
