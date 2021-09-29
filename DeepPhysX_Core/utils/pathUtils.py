from os.path import join as osPathJoin
from os.path import isdir, abspath, normpath, dirname, basename
from os import listdir, pardir, makedirs
from inspect import getmodule, stack
from shutil import copytree


def createDir(dirname, check_existing):
    # TODO Rename variable and update doc
    """
    Create a directory of the given name. If it already exist and specified, add a unique identifier at the end.

    :param str dirname: Name of the directory to create
    :param str check_existing: If True check for existence of a similar directory

    :return: Name of the created directory as string
    """
    if isdir(dirname):
        print("Directory conflict: you are going to overwrite {}.".format(dirname))
        parent = abspath(osPathJoin(dirname, pardir))
        deepest_repertory = check_existing.split('/')[-1]
        copies_list = [folder for folder in listdir(parent) if
                       isdir(osPathJoin(parent, folder)) and
                       folder.__contains__(deepest_repertory)]
        new_name = basename(normpath(dirname)) + '_{}/'.format(len(copies_list))
        dirname = osPathJoin(parent, new_name)
        print("Create a new directory {} for this session.".format(dirname))
    makedirs(dirname)
    return dirname


def copyDir(src_dir, dest_parent_dir, dest_dir):
    """
    Copy source directory to destination directory at the end of destination parent directory

    :param str src_dir: Source directory to copy
    :param str dest_parent_dir: Parent of the destination directory to copy
    :param str dest_dir: Destination directory to copy to

    :return: destination directory that source has been copied to
    """
    dest_dir = osPathJoin(dest_parent_dir, dest_dir)
    if isdir(dest_dir):
        print("Directory conflict: you are going to overwrite by copying in {}.".format(dest_dir))
        copies_list = [folder for folder in listdir(dest_parent_dir) if
                       isdir(osPathJoin(dest_parent_dir, folder)) and
                       folder.__contains__(dest_dir)]
        new_name = dest_dir + '({})/'.format(len(copies_list))
        dest_dir = osPathJoin(dest_parent_dir, new_name)
        print("Copying {} into the new directory {} for this session.".format(src_dir, dest_dir))
    else:
        new_name = dest_dir + '/'
        dest_dir = osPathJoin(dest_parent_dir, new_name)
    copytree(src_dir, dest_dir)
    return dest_dir


def getFirstCaller():
    """

    :return: The repertory in which the main script is
    """
    # Get the stack of called scripts
    scripts_list = stack()[-1]
    # Get the first one (the one launched by the user)
    module = getmodule(scripts_list[0])
    # Return the path of this script
    return dirname(abspath(module.__file__))
