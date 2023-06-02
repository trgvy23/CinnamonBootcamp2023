import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, is_downsample=False):
        """
        Creates a basic block module for a convolutional neural network.
        
        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            is_downsample (bool): Indicates whether downsampling is applied.
        """
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        
        # First convolutional layer
        if is_downsample:
            self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        # Downsample layer
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        elif input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
            self.is_downsample = True

    def forward(self, x):
        """
        Performs the forward pass of the basic block module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        
        if self.is_downsample:
            x = self.downsample(x)
        
        return F.relu(x.add(y), True)

def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    """
    Create a sequence of BasicBlock layers.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        repeat_times (int): Number of times to repeat the BasicBlock.
        is_downsample (bool): Whether to apply downsampling.

    Returns:
        nn.Sequential: Sequence of BasicBlock layers.
    """
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        """
        Convolutional neural network model.

        Args:
            num_classes (int): Number of output classes.
            reid (bool): Whether to use the model for person re-identification.
        """
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.layer1 = make_layers(64, 64, 2, False)
        self.layer2 = make_layers(64, 128, 2, True)
        self.layer3 = make_layers(128, 256, 2, True)
        self.layer4 = make_layers(256, 512, 2, True)
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.reid:
            # Person re-identification
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        
        # Classification
        x = self.classifier(x)
        return x
