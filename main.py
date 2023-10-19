import matplotlib.pyplot as plt
from config import *

plot_losses = {"train_loss": [10, 8, 6, 2, 4, 3, 1], "val_loss": [20, 22, 16, 17, 10, 9, 8]}


def plot_results(train_res, val_res, ylabel="Loss", save=False, fig_name="loss"):
    plt.figure(figsize=(8, 6))
    plt.plot(train_res)
    plt.plot(val_res)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend(['train_' + ylabel, 'val_' + ylabel])
    plt.title("Semantic_Segmentation " + ylabel)

    if save:
        fig_path = os.path.join(ARTIFACTS_OUTPUT, fig_name + ".jpg")
        if not os.path.exists(ARTIFACTS_OUTPUT):
            os.mkdir(ARTIFACTS_OUTPUT)
        plt.savefig(fig_path, facecolor='w', transparent=False, bbox_inches='tight', dpi=100)


plot_results(plot_losses["train_loss"], plot_losses["val_loss"], "Acc", True, "Acc")
plot_results(plot_losses["train_loss"], plot_losses["val_loss"], "Acc", True, "Loss")


