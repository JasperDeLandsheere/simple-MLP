import matplotlib.pyplot as plt

# Plotting activation functions and their derivatives
def plot_fn(x, fn, fn_name=""):
    y = fn.forward(x)
    dy = fn.backward(x)

    plt.plot(x, y, label="f(x)")
    plt.plot(x, dy, label="d/dx f(x)")
    plt.title(fn_name)
    plt.legend()
    plt.show()