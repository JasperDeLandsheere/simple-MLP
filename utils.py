import matplotlib.pyplot as plt

# Plotting activation functions and their derivatives
def plot_fn(x, fn, fn_name="", save=False):
    y = fn.forward(x)
    dy = fn.backward(x)

    plt.plot(x, y, label="f(x)")
    plt.plot(x, dy, label="d/dx f(x)")
    plt.title(fn_name)
    plt.legend()
    if save:
        plt.savefig(fn_name + ".png")
    plt.show()

# Plotting loss
def plot_loss(train_loss_arr, valid_loss_arr, name, save=False):
    plt.plot(train_loss_arr)
    plt.plot(valid_loss_arr)
    plt.title("Loss using " + name)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["train", "valid"])
    if save:
        plt.savefig(name + "loss.png")
    plt.show()

# Plotting predictions
def plot_predictions(X_test, y_test, y_preds, name, save=False):
    plt.scatter(X_test, y_test, label="true", s=0.5)
    plt.scatter(X_test, y_preds, label="pred", s=0.5)
    plt.title("Predictions using " + name)
    plt.legend()
    if save:
        plt.savefig(name + "pred.png")
    plt.show()