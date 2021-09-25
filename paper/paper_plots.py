import numpy as np
import matplotlib.pyplot as plt


def plot_pipeline_worth_it():
    x = list(range(3))
    y = [2 / 90 * 100, 8 / 90 * 100, 80 / 90 * 100]
    line = plt.bar(x,
                   y,
                   color="black",
                   align="center",
                   ecolor='black',
                   capsize=10)
    for i in range(len(y)):
        plt.annotate("{:.2f}".format(y[i]), xy=(x[i], y[i]), ha='center', va='bottom')

    plt.xticks(x, ["Filter", "Embedding", "Filter + Embedding"])
    plt.xlabel("Pipeline", fontsize=14)
    plt.ylabel("Percent of data sets", fontsize=14)
    plt.ylim((0, 100))
    plt.xlim((-0.5, 2.5))
    plt.savefig("plot_pipeline_worth_it.png")
    plt.close()


if __name__ == '__main__':
    plot_pipeline_worth_it()
