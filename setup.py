# # Automatically download and setup the missing libraries
# from setuptools import find_packages, setup
#
# setup(
#     name='DeepPhysX_Core',
#     packages=find_packages(include=['DeepPhysX_Core', 'DeepPhysX_PyTorch', 'DeepPhysX_Sofa']),
#     version='20.12.00',
#     description='Core project of the DeepPhysX_Core environment',
#     author='Mimesis',
#     license='',
#     #install_requires=["setuptools", "numpy", "wheel", "twine"]
# )
#
#
# # Automatically add DeepPhysX_Core modules to the python user site
# import os
# deep_physix_directory = f"{os.path.dirname(os.path.realpath(__file__))}"
# for path in os.listdir(deep_physix_directory):
#     if os.path.isdir(path) and 'DeepPhysX_Core' in path:
#         print(f"Linking {deep_physix_directory}/{str(path)} to site package")
#         os.system(f'ln -sFfv {deep_physix_directory}/{str(path)} $(python3 -m site --user-site)')

from pathlib import Path
from os import listdir, system
from os.path import join
from distutils.sysconfig import get_python_lib
current_absolute_path = Path(__file__).parent.absolute()

for dir in listdir(current_absolute_path):
    if "DeepPhysX_" in dir:
        system(f'ln -sFfv {join(current_absolute_path,dir)} {get_python_lib()}')
system(f'ln -sFfv {current_absolute_path} {get_python_lib()}')