import tensorboardX as tb
import numpy as np
import torch


def generateDefaultScene():
    return {
        'camera': {'cls': 'PerspectiveCamera', 'fov': 75},
        'light': {'cls': 'AmbientLight', 'color': '#ffffff', 'intensity': 0.75}
    }


def generateDefaultMaterial():
    return {'material': {'cls': 'MeshStandardMaterial', 'roughness': 1, 'metalness': 0, 'color': '#8888ff'}}


class DataManager:
    def __init__(self, log_dir, keep_losses=False):
        self.log_dir = log_dir
        self.writer = tb.SummaryWriter(log_dir)
        self.mean = np.full(4, np.inf)  # Contain in the first dimension the mean, and second the variance of the mean
        self.train_loss = np.array([])
        self.keep_losses = keep_losses
        self.tag_dict = {}

    def add_trainBatchLoss(self, value, count):
        var = self.updateMeanGetVar(value, 0, count + 1)
        self.writer.add_scalar("Train/Batch/Loss", value, count)
        self.writer.add_scalar("Train/Batch/Mean", self.mean[0], count)
        if var is not None:
            self.writer.add_scalar("Train/Batch/Variance", var, count)
        if self.keep_losses is True:
            self.train_loss = np.append(self.train_loss, value)

    def add_trainEpochLoss(self, value, count):
        var = self.updateMeanGetVar(value, 1, count + 1)
        self.writer.add_scalar("Train/Epoch/Loss", value, count)
        self.writer.add_scalar("Train/Epoch/Mean", self.mean[1], count)
        if var is not None:
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
        var = self.updateMeanGetVar(value, 2, count + 1)
        self.writer.add_scalar("Test/Valid/Loss", value, count)
        self.writer.add_scalar("Test/Valid/Mean", self.mean[2], count)
        if var is not None:
            self.writer.add_scalar("Test/Valid/Variance", var, count)

    def add_testLossOOB(self, value, count):
        var = self.updateMeanGetVar(value, 3, count + 1)
        self.writer.add_scalar("Test/Out-of-boundaries/Loss", value, count)
        self.writer.add_scalar("Test/Out-of-boundaries/Mean", self.mean[3], count)
        if var is not None:
            self.writer.add_scalar("Test/Out-of-boundaries/Variance", var, count)

    def add_customScalar(self, tag, value, count):
        self.writer.add_scalar(tag, value, count)

    def add_customScalarFull(self, tag, value, count):
        try:
            self.tag_dict[tag]
        except KeyError:
            self.tag_dict[tag] = len(self.tag_dict) + 4  # Size of self.mean at the initialization
        var = self.updateMeanGetVar(value, self.tag_dict[tag], count + 1)
        self.writer.add_scalar(tag + "/Value", value, count)
        self.writer.add_scalar(tag + "/Mean", self.mean[self.tag_dict[tag]], count)
        if var is not None:
            self.writer.add_scalar(tag + "/Variance", var, count)

    def updateMeanGetVar(self, value, index, count):
        if index > self.mean.shape[0] - 1:
            self.mean = np.concatenate((self.mean, np.full(index - (self.mean.shape[0] - 1), np.inf)))
        if self.mean[index] == np.inf:
            self.mean[index] = value
            return None
        else:
            # Mean value over the last 50 elements
            n = count if count < 50 else 50
            variance = self.mean[index]
            self.mean[index] = self.mean[index] + (value - self.mean[index]) / n
            return variance - self.mean[index]

    def add_3DPointCloud(self, tag, vertices, colors=None, BN3=False, config_dict=None):
        if config_dict is None:
            config_dict = {**generateDefaultScene(), **generateDefaultMaterial()}
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
            config_dict = {**generateDefaultScene(), **generateDefaultMaterial()}
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

    def add_network_weight_grad(self, network, count, save_weights=False, save_gradients=True):
        for tag, value in network.named_parameters():
            tag = tag.replace('.', '/')
            if save_weights:
                self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), count)
            if save_gradients:
                self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), count)

    def saveLossToCSV(self):
        if self.keep_losses is False:
            return
        save_path = self.log_dir + "/Losses.csv"
        file = open(save_path, "w+")
        file.write(str(self.train_loss))
        file.close()
        print("Batch losses saved at : " + save_path)

    def close(self):
        self.writer.close()
        del self.train_loss

    def description(self, minimal=False):
        description_string = "\n\nData Manager : \n"
        description_string += "Store losses as csv : {} \n".format(self.keep_losses)
        if self.keep_losses:
            description_string += "CSV file path : {}\n".format(self.log_dir + "/Losses.csv")
        description_string += "Training data location : {}\n".format(self.log_dir)
        return description_string
