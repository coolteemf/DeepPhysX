import os
import inspect
import shutil


def createDir(dirname, check_existing):
    # TODO Rename variable and update doc
    """
    Create a directory of the given name. If it already exist and specified, add a unique identifier at the end.

    :param str dirname: Name of the directory to create
    :param str check_existing: If True check for existence of a similar directory

    :return: Name of the created directory as string
    """
    if os.path.isdir(dirname):
        print("Directory conflict: you are going to overwrite {}.".format(dirname))
        parent = os.path.abspath(os.path.join(dirname, os.pardir))
        deepest_repertory = check_existing.split('/')[-1]
        copies_list = [folder for folder in os.listdir(parent) if
                       os.path.isdir(os.path.join(parent, folder)) and
                       folder.__contains__(deepest_repertory)]
        new_name = os.path.basename(os.path.normpath(dirname)) + '_{}/'.format(len(copies_list))
        dirname = os.path.join(parent, new_name)
        print("Create a new directory {} for this session.".format(dirname))
    os.makedirs(dirname)
    return dirname


def copyDir(src_dir, dest_parent_dir, dest_dir):
    """
    Copy source directory to destination directory at the end of destination parent directory

    :param str src_dir: Source directory to copy
    :param str dest_parent_dir: Parent of the destination directory to copy
    :param str dest_dir: Destination directory to copy to

    :return: destination directory that source has been copied to
    """
    dest_dir = os.path.join(dest_parent_dir, dest_dir)
    if os.path.isdir(dest_dir):
        print("Directory conflict: you are going to overwrite by copying in {}.".format(dest_dir))
        copies_list = [folder for folder in os.listdir(dest_parent_dir) if
                       os.path.isdir(os.path.join(dest_parent_dir, folder)) and
                       folder.__contains__(dest_dir)]
        new_name = dest_dir + '({})/'.format(len(copies_list))
        dest_dir = os.path.join(dest_parent_dir, new_name)
        print("Copying {} into the new directory {} for this session.".format(src_dir, dest_dir))
    else:
        new_name = dest_dir + '/'
        dest_dir = os.path.join(dest_parent_dir, new_name)
    shutil.copytree(src_dir, dest_dir)
    return dest_dir


def getFirstCaller():
    """
    WTF ?
    :return:
    """
    frm = inspect.stack()[-1]
    mod = inspect.getmodule(frm[0])
    caller_path = os.path.dirname(os.path.abspath(mod.__file__))
    return caller_path
