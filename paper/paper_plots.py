import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_no_free_lunch():
    df = pd.read_csv(r"C:\Users\lazeb\Desktop\-feature_selection_explenability_stability\meta_table_data\meta_dataset.csv")
    relevant_df = df[[col for col in list(df) if "expandability" not in col and "acc" in col]]
    pipeline_names = list(relevant_df)
    pipeline_index_best = []
    fs_filter_index_best = []
    fs_embedded_index_best = []
    for row_index, row in relevant_df.iterrows():
        best_index = np.nanargmax(row)
        pipeline_index_best.append(best_index)
        fs_filter_index_best.append(best_index % 8)
        fs_embedded_index_best.append(best_index % 3)

    # hist count
    values, counts = np.unique(pipeline_index_best, return_counts=True)
    plt.bar(range(len(values)),
            [val/len(pipeline_index_best) for val in counts],
            width=0.8,
            color="black")
    plt.xlabel("FS Pipeline")
    plt.xlabel("Probability to be the optimal pipeline")
    plt.grid(color="black",
             axis="y",
             alpha=0.1)
    # plt.xticks(range(len(pipeline_names)), pipeline_names)
    plt.savefig("full_pipeline.png")
    plt.close()

    # hist count
    values, counts = np.unique(fs_filter_index_best, return_counts=True)
    plt.bar(range(len(values)),
            [val/len(fs_filter_index_best) for val in counts],
            width=0.8,
            color="black")
    plt.xlabel("Filter FS algorithm")
    plt.xlabel("Probability to be the optimal Filter FS algorithm")
    plt.grid(color="black",
             axis="y",
             alpha=0.1)
    # plt.xticks(range(len(pipeline_names)), pipeline_names)
    plt.savefig("filter_fs.png")
    plt.close()

    # hist count
    values, counts = np.unique(fs_embedded_index_best, return_counts=True)
    plt.bar(range(len(values)),
            [val/len(fs_embedded_index_best) for val in counts],
            width=0.8,
            color="black")
    plt.xlabel("Embedded FS algorithm")
    plt.xlabel("Probability to be the optimal Embedded FS algorithm")
    plt.grid(color="black",
             axis="y",
             alpha=0.1)
    # plt.xticks(range(len(pipeline_names)), pipeline_names)
    plt.savefig("embedded_fs.png")
    plt.close()

if __name__ == '__main__':
    plot_no_free_lunch()
