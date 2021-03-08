from setuptools import find_packages, setup

setup(
    name='DeepPhysX',
    packages=find_packages(include=['DeepPhysX']),
    version='20.12.00',
    description='Core project of the DeepPhysX environment',
    author='Mimesis',
    license='',
    install_requires=["setuptools >= 50.0.0", "numpy >= 1.20.1", "wheel", "twine"]
)
