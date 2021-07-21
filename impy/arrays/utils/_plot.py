
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_drift(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, title="drift")
    ax.plot(result.x, result.y, marker="+", color="red")
    ax.grid()
    # delete the default axes and let x=0 and y=0 be new ones.
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # let the interval of x-axis and that of y-axis be equal.
    ax.set_aspect("equal")
    # set the x/y-tick intervals to 1.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    return None

def plot_gaussfit_result(raw, fit):
    x0 = raw.shape[1]//2
    y0 = raw.shape[0]//2
    plt.figure(figsize=(6,4))
    plt.subplot(2, 1, 1)
    plt.title("x-direction")
    plt.plot(raw[y0].value, color="gray", alpha=0.5, label="raw image")
    plt.plot(fit[y0], color="red", label="fit")
    plt.subplot(2, 1, 2)
    plt.title("y-direction")
    plt.plot(raw[:,x0].value, color="gray", alpha=0.5, label="raw image")
    plt.plot(fit[:,x0], color="red", label="fit")
    plt.tight_layout()
    plt.show()
    return None