# library imports
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def baseline_plot():
    x = [1,2,3,4,5,6,7,8,9]
    y = [0.496, 0.552, 0.693, 0.8, 0.833, 0.87, 0.904, 0.909, 0.91]
    y_err = [0.052, 0.032, 0.027, 0.021, 0.018, 0.015, 0.014, 0.013, 0.013]
    plt.errorbar(x,
                 y,
                 y_err,
                 fmt="-o",
                 capsize=2,
                 markersize=6,
                 color="black",
                 label="Model's prediction")
    # baseline
    plt.plot(x,
             [(i+1)/24 for i in range(len(x))],
             "--o",
             markersize=6,
             color="red",
             label="Random choose")
    plt.xlabel("K")
    plt.ylabel("Model's accuracy")
    plt.grid(color="black",
             alpha=0.1)
    plt.ylim((0, 1))
    plt.xlim((0.9, len(x)+0.1))
    plt.xticks(x, x, rotation=0)
    plt.yticks([i/10 for i in range(11)], [i/10 for i in range(11)], rotation=0)
    plt.tight_layout()
    plt.legend()
    plt.savefig("baseline.png")
    plt.close()


if __name__ == '__main__':
    baseline_plot()
