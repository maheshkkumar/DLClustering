import os


def check_directory(path):
    """Module to perform sanity check on existence of a folder.
    """
    if not os.path.exists(path):
        os.makedirs(path)
