import tensorboardX as tb
import numpy as np
import torch


def generateDefaultScene():
    return {
        'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
        'light': {'cls': 'AmbientLight', 'color': '#ffffff', 'intensity': 0.75}
    }


class StatsManager:
    def __init__(self, log_dir, sliding_window_size=50):
        self.log_dir = log_dir
        self.writer = tb.SummaryWriter(log_dir)
        self.sliding_window_size = sliding_window_size
        self.values = np.full((4, sliding_window_size), np.inf)
        self.current_value = np.zeros(4, dtype=np.int)
        self.train_loss = np.array([])
        self.tag_dict = {}

    def add_trainBatchLoss(self, value, count):
        mean, var = self.computeMeanAndVar(value, 0, count + 1)
        self.writer.add_scalar("Train/Batch/Loss", value, count)
        self.writer.add_scalar("Train/Batch/Mean", mean, count)
        self.writer.add_scalar("Train/Batch/Variance", var, count)

    def add_trainEpochLoss(self, value, count):
        mean, var = self.computeMeanAndVar(value, 1, count + 1)
        self.writer.add_scalar("Train/Epoch/Loss", value, count)
        self.writer.add_scalar("Train/Epoch/Mean", mean, count)
        self.writer.add_scalar("Train/Epoch/Variance", var, count)

    def add_trainTestBatchLoss(self, trainValue, testValue, count):
        if trainValue is not None:
            self.writer.add_scalars("Combined/Batch/Loss", {'Train': trainValue}, count)
        if testValue is not None:
            self.writer.add_scalars("Combined/Batch/Loss", {'Test': testValue}, count)

    def add_valuesMultiPlot(self, graphName, tags, values, counts):
        for t, v, c in zip(tags, values, counts):
            self.writer.add_scalars(graphName, {t: v}, c)

    def add_testLoss(self, value, count):
        mean, var = self.computeMeanAndVar(value, 2, count + 1)
        self.writer.add_scalar("Test/Valid/Loss", value, count)
        self.writer.add_scalar("Test/Valid/Mean", mean, count)
        self.writer.add_scalar("Test/Valid/Variance", var, count)

    def add_testLossOOB(self, value, count):
        mean, var = self.computeMeanAndVar(value, 3, count + 1)
        self.writer.add_scalar("Test/Out-of-boundaries/Loss", value, count)
        self.writer.add_scalar("Test/Out-of-boundaries/Mean", mean, count)
        self.writer.add_scalar("Test/Out-of-boundaries/Variance", var, count)

    def add_customScalar(self, tag, value, count):
        self.writer.add_scalar(tag, value, count)

    def add_customScalarFull(self, tag, value, count):
        try:
            self.tag_dict[tag]
        except KeyError:
            self.tag_dict[tag] = len(self.tag_dict) + 4  # Size of self.mean at the initialization
        mean, var = self.computeMeanAndVar(value, self.tag_dict[tag], count + 1)
        self.writer.add_scalar(tag + "/Value", value, count)
        self.writer.add_scalar(tag + "/Mean", mean, count)
        self.writer.add_scalar(tag + "/Variance", var, count)

    def computeMeanAndVar(self, value, index, count):
        if index > self.values.shape[0] - 1:
            self.values = np.concatenate((self.values, np.full((1, self.sliding_window_size), np.inf)))
            self.values[index] = value
            self.current_value = np.append(1)  # Set to 1 since we had the value above and the "pointer" update is done after the insertion
            return value, 0.0
        else:
            self.values[index][self.current_value[index]] = value
            self.current_value[index] = (self.current_value[index]+1) % self.sliding_window_size
            mean = np.mean(self.values[index])
            var = np.mean(self.values[index] * self.values[index]) - mean**2
            return mean, var

    def add_3DPointCloud(self, tag, vertices, colors=None, BN3=False, config_dict=None):
        if config_dict is None:
            config_dict = {**generateDefaultScene(), **self.generateDefaultMaterial()}
        # Information should be written using (Batch, number of vertex, 3) as shape. Hence, if not we emulate it
        if not BN3:
            v = vertices[None, :, :]
            if colors is not None:
                c = colors[None, :, :]
        else:
            v = vertices
            c = colors

        self.writer.add_mesh(tag=tag, vertices=v, colors=c, config_dict=config_dict)

    def add_3DMesh(self, tag, vertices, colors=None, faces=None, BN3=False, config_dict=None):
        if config_dict is None:
            config_dict = {**generateDefaultScene(), **self.generateDefaultMaterial()}
        # Information should be written using (Batch, number of vertex, 3) as shape. Hence, if not we emulate it
        if not BN3:
            v = vertices[None, :, :]
            if colors is not None:
                c = colors[None, :, :]
            if faces is not None:
                f = faces[None, :, :]
        else:
            v = vertices
            c = colors
            f = faces
        self.writer.add_mesh(tag=tag, vertices=v, colors=c, faces=f, config_dict=config_dict)

    def generateDefaultMaterial(self):
        return {'material': {'cls': 'MeshStandardMaterial', 'roughness': 1, 'metalness': 0, 'color': '#8888ff'}}


    def close(self):
        self.writer.close()
        del self.train_loss

    def description(self, minimal=False):
        description_string = "\n\nData Manager : \n"
        description_string += "Training data location : {}\n".format(self.log_dir)
        return description_string
