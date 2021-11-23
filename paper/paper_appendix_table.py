import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv("../meta_table_data/meta_dataset.csv")
    df = df[["ds_name", "x-row_count", "x-col_count"]]
    df["index"] = [index + 1 for index, row in df.iterrows()]
    df["domain"] = ["" for index, row in df.iterrows()]

    df.to_csv("paper.csv",
              header=False,
              sep="&")