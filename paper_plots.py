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
        plt.xticks(x_vals, x, rotation=0)
        plt.yticks([0.1 * i for i in range(11)])
        plt.ylim((0, 1))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("{}.png".format(name))
        plt.close()


if __name__ == '__main__':
    x = ["CS", "SU", "IG", "PC", "RLV", "FS"]

    Plots.plot(x=["Chi square", "Pearson correlation"],
               y_data=[0.81, 0.63],
               y_feature=[0.54, 0.84],
               y_lyaponuv=[0.65, 0.77],
               y_data_yerr=[0.03, 0.09],
               y_feature_yerr=[0.07, 0.03],
               y_lyaponuv_yerr=[0.08, 0.06],
               name="target_based_fs")

    Plots.plot(x=x,
               y_data=[0.67, 0.55, 0.73, 0.59, 0.8, 0.83],
               y_feature=[0.85, 0.81, 0.88, 0.83, 0.98, 0.94],
               y_lyaponuv=[0.43, 0.31, 0.48, 0.46, 0.78, 0.53],
               y_data_yerr=[0.06, 0.09, 0.07, 0.1, 0.03, 0.1],
               y_feature_yerr=[0.02, 0.03, 0.01, 0.03, 0.02, 0.02],
               y_lyaponuv_yerr=[0.08, 0.11, 0.05, 0.08, 0.04, 0.13],
               name="shift_concept_drift")


    Plots.plot(x=x,
               y_data=[0.77, 0.68, 0.83, 0.81, 0.91, 0.87],
               y_feature=[0.88, 0.84, 0.91, 0.89, 0.98, 0.93],
               y_lyaponuv=[0.56, 0.48, 0.59, 0.62, 0.86, 0.65],
               y_data_yerr=[0.04, 0.11, 0.06, 0.12, 0.04, 0.08],
               y_feature_yerr=[0.03, 0.02, 0.02, 0.02, 0.01, 0.01],
               y_lyaponuv_yerr=[0.06, 0.12, 0.04, 0.1, 0.07, 0.1],
               name="moving_concept_drift")

    Plots.plot(x=x,
               y_data=[0.23, 0.43, 0.38, 0.76, 0.48, 0.37],
               y_feature=[0.85, 0.87, 0.92, 0.87, 0.92, 0.86],
               y_lyaponuv=[0.53, 0.68, 0.48, 0.82, 0.79, 0.72],
               y_data_yerr=[0.13, 0.08, 0.1, 0.06, 0.09, 0.1],
               y_feature_yerr=[0.05, 0.03, 0.04, 0.03, 0.03, 0.02],
               y_lyaponuv_yerr=[0.09, 0.1, 0.07, 0.05, 0.06, 0.08],
               name="non_uniform_nulls")
