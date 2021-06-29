# Automatically download and setup the missing libraries
from setuptools import find_packages, setup

setup(
    name='DeepPhysX_Core',
    packages=find_packages(include=['DeepPhysX_Core', 'DeepPhysX_PyTorch', 'DeepPhysX_Sofa']),
    version='20.12.00',
    description='Core project of the DeepPhysX_Core environment',
    author='Mimesis',
    license='',
    install_requires=["setuptools", "numpy", "wheel", "twine"]
)


# Automatically add DeepPhysX_Core modules to the python user site
import os
deep_physix_directory = f"{os.path.dirname(os.path.realpath(__file__))}"
for path in os.listdir(deep_physix_directory):
    if os.path.isdir(path) and 'DeepPhysX_Core' in path:
        os.system(f'ln -sFfv {deep_physix_directory}/{str(path)} $(python3 -m site --user-site)')
