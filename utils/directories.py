import errno
import os
import re
from functools import cmp_to_key


def sort_string_func(x, y):
    if len(x) > len(y):
        return 1
    if len(y) > len(x):
        return -1
    if str(x).lower() > str(y).lower():
        return 1
    return -1


def get_checkpoint_file_list(checkpoint_dir, name):
    regex = name + r"_[0-9]"
    files = [f for f in os.listdir(checkpoint_dir) if re.match(regex, f)]
    files = sorted(files, key=cmp_to_key(sort_string_func))
    return files


def safe_mkdir_recursive(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise


def safe_mknode_recursive(destiny_dir, node_name, overwrite):
    safe_mkdir_recursive(destiny_dir)
    if overwrite and os.path.exists(destiny_dir + node_name):
        os.remove(destiny_dir + node_name)
    if not os.path.exists(destiny_dir + node_name):
        os.mknod(destiny_dir + node_name)
        return False
    return True


def add_text_to_txt_file(text, destiny, file_name, overwrite=False):
    """
    Adds a txt file at the training directory with the location of the scaler functions used to transform the data that
    created the model for the first time

    :param text: Text to write in the text file
    :param destiny: Directory of the txt file
    :param file_name: Name of the text file
    :param overwrite: whether to overwrite the existing file
    """

    existed = safe_mknode_recursive(destiny, file_name, overwrite)
    if overwrite or (not overwrite and not existed):
        file = open(destiny + file_name, 'w')
        file.write(text)
        file.close()
