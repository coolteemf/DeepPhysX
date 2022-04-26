from setuptools import setup, find_packages
from config import PACKAGES


# Init DeepPhysX packages and dependencies to install
roots = ['Core']
available = {'ai': ['PyTorch'],
             'simu': ['Sofa']}
dependencies = {'Core': ['numpy', 'vedo', 'tensorboard', 'tensorboardX', 'pyDataverse'],
                'Sofa': [],
                'PyTorch': ['torch', 'psutil']}
packages = []
requires = []

# Include user config
user_ai_packages = []
user_simu_packages = []
for user_packages, key in zip([user_ai_packages, user_simu_packages], ['ai', 'simu']):
    for root in available[key]:
        if PACKAGES[root.lower()]:
            user_packages.append(root)

# Define the main packages to install
roots += user_ai_packages
roots += user_simu_packages

# Configure packages and subpackages list and dependencies list
prefix = 'DeepPhysX_'
packages = [f'{prefix + root}' for root in roots]
requires = []
for root in roots:
    sub_packages = find_packages(where=prefix + root)
    packages += [f'{prefix + root}.{sub_package}' for sub_package in sub_packages]
    requires += dependencies[root]

# Installation
setup(name='DeepPhysX',
      version='1.0',
      description='A Python framework interfacing AI with numerical simulation.',
      author='Mimesis',
      author_email='robin.enjalbert@inria.fr',
      packages=packages,
      install_requires=requires,
      )
