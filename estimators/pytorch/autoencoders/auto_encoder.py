import os
import sys
import torch
import torchvision
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def data_create():
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
    train_set = torch.Tensor(mnist.train.images)
    test_set = torch.Tensor(mnist.test.images)
    return train_set, test_set


def main(train_set, test_set):
    n_epochs = 100
    batch_size = 150
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    model = AutoEncoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        for data in train_loader:
            data = Variable(data)
            output = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_output = model(test_set)
        train_output = model(train_set)
        print('Epoch {:2d} batch accuracy {:3.5f}, train accuracy {:3.5f}, test accuracy {:3.5f}'.format(epoch + 1, loss.data, criterion(train_output, train_set), criterion(test_output, test_set)))

    torch.save(model, 'autoencoder.pt')


def viz(data_test, index):
    model = torch.load('autoencoder.pt')

    plt.figure()
    plt.title('Actual')
    plt.imshow(data_test[index].numpy().reshape((28, 28)), cmap=plt.cm.gray_r)

    image_pred = model(data_test[index]).detach().numpy()
    plt.figure()
    plt.title('Generated')
    plt.imshow(image_pred.reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.show()


if __name__ == '__main__':
    data_train, data_test = data_create()
    main(data_train, data_test)

    viz(data_test, 30)
