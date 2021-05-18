from setuptools import find_packages, setup

setup(
    name='DeepPhysX',
    packages=find_packages(include=['DeepPhysX']),
    version='20.12.00',
    description='Core project of the DeepPhysX environment',
    author='Mimesis',
    license='',
    install_requires=["setuptools", "numpy", "wheel", "twine"]
)
