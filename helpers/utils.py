import os

def check_directory(path):
    """
    A method to create the directory, if it does not exists.
    :param path: String
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)