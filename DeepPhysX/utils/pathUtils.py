import os
import inspect


def createDir(dirname, key):
    if os.path.isdir(dirname):
        print("Directory conflict: you are going to overwrite {}.".format(dirname))
        parent = os.path.join(dirname, os.pardir)
        copies_list = [folder for folder in os.listdir(parent) if
                       os.path.isdir(os.path.join(parent, folder)) and
                       folder.__contains__(key)]
        dirname = dirname + '({})'.format(len(copies_list))
        print("Create a new directory {} for this session.".format(dirname))
    os.makedirs(dirname)
    return dirname


def getFirstCaller():
    frm = inspect.stack()[-1]
    mod = inspect.getmodule(frm[0])
    caller_path = os.path.dirname(os.path.abspath(mod.__file__))
    return caller_path
