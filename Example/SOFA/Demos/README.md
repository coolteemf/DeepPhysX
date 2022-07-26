## Demos

This repository contains higher level applications. 
The understanding of these sessions will be easier after studying the content of "Tutorial" and "Features" repositories.

> :warning: **PyTorch usage**: This working session uses the Fully Connected implementation from DeepPhysX_Torch 
package, please make sure it is installed if you want to run training scripts.

### Armadillo, Beam, Liver

The objective of these three sessions is to enable a neural network to predict deformations from forces applied on the
object.
Then, the network will interface the following data:
* `input`: applied forces on surface;
* `output`: deformation to apply on volume.

Each demo will contain the following scripts:
* `download.py`: Allow to automatically download the missing demo data. Launching this script is not mandatory.
* `runSofa.py`: Launch the `Environment/<Demo>Sofa` in a SOFA GUI.
* `dataset.py`: Launch the data generation from `Environemnt/<Demo>Training` Environment.
* `training.py`: Launch the training session either from existing Dataset or from `Environemnt/<Demo>Training`.
* `validation.py`: Compare the predictions of the network with the ground truth in `Environment/<Demo>Validation`.
* `prediction.py`: Launch the prediction session with `Environment/<Demo>Prediction`.
