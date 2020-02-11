import os


def create_dir(path_to_dir):
    """
    check whether the directory already exists. if not, create it
    """

    try:
        os.makedirs(path_to_dir)
        print("created {}".format(path_to_dir))
    except Exception as err:
        print(err)
