# library imports
import numpy as np
import matplotlib.pyplot as plt


class Plots:
    """
    Manage the plots for the manuscript
    """

    def __init__(self):
        pass

    @staticmethod
    def plot(x: list,
             y_data: list,
             y_feature: list,
             y_lyaponuv: list,
             y_data_yerr: list,
             y_feature_yerr: list,
             y_lyaponuv_yerr: list,
             name: str,
             width: float = 0.3):
        x_vals = np.array([i for i in range(len(x))])
        plt.grid(axis="y", alpha=0.2, color="b")
        plt.bar(x_vals - 1 * width, y_data, width=width, yerr=y_data_yerr, color="#4285F4", label='Record-stability')
        plt.bar(x_vals, y_feature, width=width, yerr=y_feature_yerr, color="#DB4437", label='Feature-stability')
        plt.bar(x_vals + 1 * width, y_lyaponuv, width=width, yerr=y_lyaponuv_yerr, color="#0F9D58", label='Lyapunov-stability')
        plt.xlabel('')
        plt.ylabel('Normalized stability score')
        plt.xticks(x_vals, x)
        plt.yticks([0.1 * i for i in range(11)])
        plt.ylim((0, 1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend()
        plt.savefig("{}.png".format(name))
        plt.close()


if __name__ == '__main__':
    x = ["Chi square",
                 "Symmetrical uncertainty",
                 "Information gain",
                 "Pearson correlation",
                 "Spearman correlation",
                 "Remove low variance",
                 "Missing value ratio",
                 "Fishers score"]

    Plots.plot(x=["Chi square", "Pearson correlation"],
               y_data=[0.81, 0.63],
               y_feature=[0.54, 0.84],
               y_lyaponuv=[0.65, 0.77],
               y_data_yerr=[0.03, 0.09],
               y_feature_yerr=[0.07, 0.03],
               y_lyaponuv_yerr=[0.08, 0.06],
               name="target_based_fs")


    Plots.plot(x=x,
               y_data=[],
               y_feature=[],
               y_lyaponuv=[],
               y_data_yerr=[],
               y_feature_yerr=[],
               y_lyaponuv_yerr=[],
               name="shift_concept_drift")
    Plots.plot(x=x,
               y_data=[],
               y_feature=[],
               y_lyaponuv=[],
               y_data_yerr=[],
               y_feature_yerr=[],
               y_lyaponuv_yerr=[],
               name="moving_concept_drift")
    Plots.plot(x=x,
               y_data=[],
               y_feature=[],
               y_lyaponuv=[],
               y_data_yerr=[],
               y_feature_yerr=[],
               y_lyaponuv_yerr=[],
               name="non_uniform_nulls")
