import torch.nn as nn
import torch
from torchsummary import summary
# import torchsummaryX
from lib.medzoo.BaseModelClass import BaseModel


class TinyNet(BaseModel):
    """
    Very small alexnet
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8):
        super(TinyNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1,
                                     bias=False)
        self.lrelu_conv_c2 = self.lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1,
                                     bias=False)
        self.lrelu_conv_c3 = self.lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1,
                                     bias=False)
        self.lrelu_conv_c4 = self.lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)

        self.globa_Avg = nn.AdaptiveAvgPool3d((4,4,4))
        self.fc1 = nn.Linear(self.base_n_filter * 8 * 4 * 4 * 4,self.base_n_filter * 4)
        self.fc2 = nn.Linear(self.base_n_filter * 4, n_classes)
        self.dropout = nn.Dropout(p=0.6)

    # def lrelu_conv(self, feat_in, feat_out):
    #     return nn.Sequential(
    #         nn.InstanceNorm3d(feat_in),
    #         nn.LeakyReLU(),
    #         nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1(x)
        # out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)

        out = self.conv3d_c2(out)
        # out = self.dropout3d(out)
        out = self.lrelu_conv_c2(out)

        out = self.conv3d_c3(out)
        # out = self.dropout3d(out)
        out = self.lrelu(out)

        out = self.conv3d_c4(out)
        # out = self.dropout3d(out)
        out = self.lrelu(out)

        out = self.globa_Avg(out)
        out = torch.flatten(out, 1)
        out = self.lrelu(self.fc1(self.dropout(out)))
        out = self.fc2(self.dropout(out))

        return out

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("TinyNet test is complete")
