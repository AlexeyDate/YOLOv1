import numpy as np
from torch import nn
from tools.loadWeights import load_conv, load_conv_batch_norm


class Extraction(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AvgPool2d(kernel_size=13, stride=13),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier(x)

    def load_weights(self, weightfile):
        """
        loading weights to Extraction.
        model target weight size should be 23455400.
        weight file after converting to numpy should have size 23455400, therefore, it should be noted that the
        first 4 indexes are considered as the heading.

        param: binary file of weights
        """

        with open(weightfile, 'rb') as fp:
            header = np.fromfile(fp, count=4, dtype=np.int32)
            buf = np.fromfile(fp, dtype=np.float32)
            start = 0

            # load weights to convolution layers
            for num_layer, layer in enumerate(self.conv_block):
                if start >= buf.size:
                    break
                if isinstance(layer, nn.modules.conv.Conv2d):
                    conv_layer = self.conv_block[num_layer]
                    if num_layer + 1 != len(self.conv_block):
                        if isinstance(self.conv_block[num_layer + 1], nn.modules.BatchNorm2d):
                            batch_norm_layer = self.conv_block[num_layer + 1]
                            start = load_conv_batch_norm(buf, start, conv_layer, batch_norm_layer)
                    else:
                        start = load_conv(buf, start, conv_layer)

            # load weights to output layer
            conv_layer = self.classifier[0]
            start = load_conv(buf, start, conv_layer)

            if start == buf.size:
                print("Extraction weight file upload successfully")


class YOLOv1(nn.Module):
    """
    Yolov1 class implements the original architecture of the Yolov1 model.
    this class contains model of Extraction class - it is CNN backbone of yolo model
    """

    def __init__(self, S=7, B=2, C=3, device='cpu', extraction_weights_path=None):
        """
        param: S - grid size (default = 7)
        param: B - number of boxes (default = 2)
        param: C - number of classes (default = 3)
        param: device - device of model (default = cpu)
        param: extraction_weights_file - weight file of Extraction backbon model (default = None)
        """

        super(YOLOv1, self).__init__()

        self.S = S
        self.B = B
        self.C = C

        self.extraction = Extraction().to(device)

        if extraction_weights_path is not None:
            self.extraction.load_weights(extraction_weights_path)

        self.extraction.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.detection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.S * self.S * 1024, out_features=4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=4096, out_features=self.S * self.S * (5 * self.B + self.C))
        )

    def forward(self, x):
        x = self.extraction(x)
        x = self.detection(x)
        batch_size = x.shape[0]
        return x.reshape(batch_size, self.S, self.S, 5 * self.B + self.C)
