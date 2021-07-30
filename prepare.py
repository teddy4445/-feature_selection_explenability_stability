import os


def prepare():
    for folder_name in ["data", "results"]:
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), folder_name))
        except:
            pass
    for results_folder_name in ["expandability", "stability"]:
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), "results", results_folder_name))
        except:
            pass


if __name__ == '__main__':
    prepare()
