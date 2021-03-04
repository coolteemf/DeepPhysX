"""
Test script for the DeepPhysX tools
Train a network to return the first decimal of a random number in [0, 1[
"""

from DeepPhysX.Manager.DatasetManager import DatasetManager
import numpy as np
import random
import torch


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def createData(size):
    data = np.empty((size, 10), dtype=np.float32)
    labels = np.empty(size, dtype=np.longlong)
    for i in range(size):
        val = round(random.random(), 2)
        while val >= 1.0:
            val = round(random.random(), 2)
        data[i] = np.array([val for _ in range(10)])
        labels[i] = int(10.0 * val)
    return torch.from_numpy(data), torch.from_numpy(labels)


def train(epochs, x, y, model, criterion, optimizer):
    for t in range(epochs):
        y_pred = model(x)
        # Loss
        loss = criterion(y_pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 100 == 99:
            print(t, loss.item())
    return model


def test(x, y, model):
    success = 0
    with torch.no_grad():
        for i in range(len(x)):
            output = model(x[i])
            output = output.numpy()
            pred = np.argmax(output)
            if pred == y[i].item():
                success += 1
        print(success * 100.0 / len(x))


def main():
    x, y = createData(200)
    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model = train(10000, x, y, model, criterion, optimizer)

    x_test, y_test = createData(400)
    test(x_test, y_test, model)
    return


if __name__ == '__main__':
    main()
