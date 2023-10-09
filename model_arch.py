import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, n_filters, n_kernels):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_filters, n_filters, kernel_size=n_kernels[0], padding=1)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=n_kernels[1], padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(n_filters, n_filters, kernel_size=n_kernels[2], padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters)

        self.shortcut = nn.Conv1d(n_filters, n_filters, kernel_size=1, padding=0)
        self.bn_shortcut = nn.BatchNorm1d(n_filters)
        self.out_block = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)
        x = x + shortcut
        out_block = self.out_block(x)
        return out_block

class Resnet1D(nn.Module):
    def __init__(self, params):
        super(Resnet1D, self).__init__()

        self.n_cnn_filters = params["n_cnn_filters"]
        self.n_cnn_kernels = params["n_cnn_kernels"]
        self.n_fc_units = params["n_fc_units"]
        self.n_classes = params["n_classes"]

        self.block1 = ResnetBlock(self.n_cnn_filters[0], self.n_cnn_kernels)
        self.block2 = ResnetBlock(self.n_cnn_filters[1], self.n_cnn_kernels)
        self.block3 = ResnetBlock(self.n_cnn_filters[2], self.n_cnn_kernels)
        self.block4 = ResnetBlock(self.n_cnn_filters[2], self.n_cnn_kernels)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.n_fc_units[0], self.n_fc_units[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.n_fc_units[0], self.n_fc_units[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.n_fc_units[1], self.n_classes)

    def forward(self, inputs):
        signal_input = inputs["features_input"]

        out_block1 = self.block1(signal_input)
        out_block2 = self.block2(out_block1)
        out_block3 = self.block3(out_block2)
        out_block4 = self.block4(out_block3)

        x = self.flatten(out_block4)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        output = self.fc3(x)
        return output
