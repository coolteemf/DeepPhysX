## DeepPhysX

Core project of the DeepPhysX environment.

### Install Python library 
In your own Python environment, make sure you have pip installed:
* `pip install wheel`
* `pip install setuptools`
* `pip install twine`

To build the DeepPhysX library, run the following command line in the root folder of this project:

`python3 setup.py bdist_wheel`

This created a `.whl` file into a new `dist` folder. You can finally install the library by using:

`python3 -m pip install /path/to/your/wheelfile.whl`

Once you have installed your Python library, you can import it using:

`import DeepPhysX`
