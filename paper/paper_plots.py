import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_no_free_lunch():
    df = pd.read_csv(r"C:\Users\lazeb\Desktop\-feature_selection_explenability_stability\meta_table_data\meta_dataset.csv")
    relevant_df = df[[col for col in list(df) if "expandability" not in col and "acc" in col]]
    pipeline_names = list(relevant_df)

    pipeline_index_best = {item.replace("accuracy-", "").replace("-", ",").replace("_", " "): 0 for item in pipeline_names}
    fs_names = ["chi_square",
                 "symmetrical_uncertainty",
                 "information_gain",
                 "pearson_correlation",
                 "spearman_correlation",
                 "remove_low_variance",
                 "missing_value_ratio",
                 "fishers_score"]
    fs_filter_index_best = {item.replace("_", " "): 0 for item in fs_names}
    embbeded_names = ["decision_tree",
                    "lasso",
                    "linear_svc"]
    fs_embedded_index_best = {item.replace("_", " "): 0 for item in embbeded_names}
    for row_index, row in relevant_df.iterrows():
        new_row = [val + random.random()/1000 for val in row]
        best_index = np.nanargmax(new_row)
        pipeline_index_best[pipeline_names[best_index].replace("accuracy-", "").replace("-", ",").replace("_", " ")] += 1
        fs_filter_index_best[fs_names[best_index % 8].replace("_", " ")] += 1
        fs_embedded_index_best[embbeded_names[best_index % 3].replace("_", " ")] += 1

    # hist count
    keys = list(pipeline_index_best.keys())
    values = [pipeline_index_best[key] for key in keys]
    count = sum(values)
    plt.bar(range(len(pipeline_index_best)),
            [val/count for val in values],
            width=0.8,
            color="black")
    plt.xlabel("FS Pipeline")
    plt.ylabel("Probability to be the optimal pipeline")
    plt.grid(color="black",
             axis="y",
             alpha=0.1)
    keys = [key.replace("chi square", "CS")
                .replace("symmetrical uncertainty", "SU")
                .replace("pearson correlation", "PC")
                .replace("remove low variance", "RLV")
                .replace("missing value ratio", "MVR")
                .replace("spearman correlation", "SC")
                .replace("fishers score", "FS")
                .replace("information gain", "IG") for key in keys]
    keys = [key.replace("decision tree", "DT").replace("lasso", "L").replace("linear svc", "SVC") for key in keys]
    plt.xticks(range(len(keys)), keys, rotation=90)
    plt.tight_layout()
    plt.savefig("full_pipeline.png")
    plt.close()

    # hist count
    keys = list(fs_filter_index_best.keys())
    values = [fs_filter_index_best[key] for key in keys]
    count = sum(values)
    plt.bar(range(len(fs_filter_index_best)),
            [val/count for val in values],
            width=0.8,
            color="black")
    plt.xlabel("Filter FS algorithm")
    plt.ylabel("Probability to be the optimal Filter FS algorithm")
    plt.grid(color="black",
             axis="y",
             alpha=0.1)

    keys = [key.replace("chi square", "CS")
                .replace("symmetrical uncertainty", "SU")
                .replace("pearson correlation", "PC")
                .replace("remove low variance", "RLV")
                .replace("missing value ratio", "MVR")
                .replace("spearman correlation", "SC")
                .replace("fishers score", "FS")
                .replace("information gain", "IG") for key in keys]
    plt.xticks(range(len(keys)), keys, rotation=90)
    plt.tight_layout()
    plt.savefig("filter_fs.png")
    plt.close()

    # hist count
    keys = list(fs_embedded_index_best.keys())
    values = [fs_embedded_index_best[key] for key in keys]
    count = sum(values)
    plt.bar(range(len(values)),
            [val/count for val in values],
            width=0.8,
            color="black")
    plt.xlabel("Embedded FS algorithm")
    plt.ylabel("Probability to be the optimal Embedded FS algorithm")
    plt.grid(color="black",
             axis="y",
             alpha=0.1)
    keys = [key.replace("decision tree", "DT").replace("lasso", "L").replace("linear svc", "SVC") for key in keys]
    plt.xticks(range(len(keys)), keys, rotation=90)
    plt.tight_layout()
    plt.savefig("embedded_fs.png")
    plt.close()

if __name__ == '__main__':
    plot_no_free_lunch()
