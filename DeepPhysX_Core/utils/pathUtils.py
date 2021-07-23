import os
import inspect
import shutil


def createDir(dirname, check_existing):
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
    # Get the stack of called scripts
    scripts_list = inspect.stack()[-1]
    # Get the first one (the one launched by the user)
    module = inspect.getmodule(scripts_list[0])
    # Return the path of this script
    return os.path.dirname(os.path.abspath(module.__file__))
