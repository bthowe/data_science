import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


def data_create():
    centers = np.array([[0, 0]] * 100 + [[1, 1]] * 100 + [[0, 1]] * 100 + [[1, 0]] * 100)
    data = torch.Tensor(np.random.normal(0, 0.2, (400, 2)) + centers)

    lab = torch.Tensor(np.array([0] * 200 + [1] * 200)).int()

    return data, lab


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5, True)
        self.fc2 = nn.Linear(5, 1, True)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

def train(model, loss_func, optimizer, batch_size, epochs, inputs, targets):
    '''This function trains the model.'''
    for idx in range(0, epochs):
        j = np.random.choice(len(inputs.numpy()), batch_size, replace=False)  # batch indeces
        X = Variable(torch.Tensor(inputs.numpy()[j, :]))  # batch of features of size "batch_size"
        y = Variable(torch.Tensor(targets.numpy()[j].reshape(batch_size, 1)))  # corresponding batch of outcomes

        optimizer.zero_grad()  # zero the gradient buffers

        output = model(X)  # predictions from the model

        loss = loss_func(output, y)  # calculate loss from the predictions
        loss.backward()  # computes gradients
        optimizer.step()  # updates weights based on gradients

        if idx % 50 == 0:
            print("Epoch {: >8} Loss: {}".format(idx, loss.data.numpy()))  # every 50 batches prints the loss
    return model

def plot(X, y, model):
    '''Plots the predicted probabilities based on the weights and biases from the hidden and output layers'''
    weights = model.fc1.weight.detach().numpy()  # weights from the hidden layer
    num_base_log_regs = weights.shape[0]  # number of nodes in the hidden layer

    biases = model.fc1.bias.detach().numpy().reshape((1, num_base_log_regs))  # biases in the hidden layer

    mesh = np.column_stack(a.reshape(-1) for a in np.meshgrid(np.r_[-1:2:100j], np.r_[-1:2:100j]))

    ones = np.ones((len(mesh), 1))
    ymesh1 = np.dot(mesh, weights.T) + np.dot(ones, biases)  # hidden layer predicted probabilities
    ymesh2 = model(torch.Tensor(mesh)).detach().numpy()  # output layer predicted probabilities

    for i in range(0, num_base_log_regs):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        mesh = ax.imshow(ymesh1[:, i].reshape(100, 100), cmap=plt.cm.RdYlBu, origin='lower', extent=(-1, 2, -1, 2), vmin=0, vmax=1)
        ax.scatter(X[:, 0], X[:, 1], c=y.detach().numpy(), cmap=plt.cm.RdYlBu, edgecolor='w', lw=1)
        ax.axis((-1, 2, -1, 2))
        plt.title('Predicted Probabilities, Node {}'.format(i + 1))
        plt.colorbar(mesh, ax=ax)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    mesh2 = ax2.imshow(ymesh2.reshape(100, 100), cmap=plt.cm.RdYlBu, origin='lower', extent=(-1, 2, -1, 2), vmin=0, vmax=1)
    ax2.scatter(X[:, 0], X[:, 1], c=y.detach().numpy(), cmap=plt.cm.RdYlBu, edgecolor='w', lw=1)
    ax2.axis((-1, 2, -1, 2))
    plt.colorbar(mesh2, ax=ax2)
    plt.title('Output Predicted Probabilities')
    plt.show()

if __name__ == '__main__':
    np.random.seed(2)

    batch_size = 100
    epochs = 50000

    X_train, y_train = data_create()

    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    model = train(net, criterion, optimizer, batch_size, epochs, X_train, y_train)

    plot(X_train, y_train, model)
