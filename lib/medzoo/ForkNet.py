import torch.nn as nn
import torch
from torchsummary import summary
# import torchsummaryX
from lib.medzoo.BaseModelClass import BaseModel


class BaseNet(BaseModel):
    """
    AlexNet no FL
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8, show_active = False):
        super(BaseNet, self).__init__()
        self.show_active = show_active
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        self.globa_Avg = nn.AdaptiveAvgPool3d((2,4,4))
        self.fc1 = nn.Linear(self.base_n_filter * 8 * 2 * 4 * 4,self.base_n_filter * 4)
        self.fc2 = nn.Linear(self.base_n_filter * 4, n_classes)
        self.dropout = nn.Dropout(p=0.6)

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)

        return out

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 3, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("BaseNet test is complete")

class ForkNet(BaseModel):
    """
    Combine 3 AlexNet no ForkNet
    """

    def __init__(self, in_channels, n_classes, base_n_filter=8, show_active = False):
        super(ForkNet, self).__init__()
        self.show_active = show_active
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.alexnet = BaseNet(in_channels = self.in_channels//3, n_classes=self.n_classes, base_n_filter=8)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 4 *3, self.base_n_filter * 4 *3)
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4 *3, self.base_n_filter * 8, kernel_size=3, stride=2, padding=1,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        self.globa_Avg = nn.AdaptiveAvgPool3d((2,4,4))
        self.fc1 = nn.Linear(self.base_n_filter * 8 * 2 * 4 * 4,self.base_n_filter * 4)
        self.fc2 = nn.Linear(self.base_n_filter * 4, n_classes)
        self.dropout = nn.Dropout(p=0.6)


    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        #  Level 1 context pathway
        CBCT = x[:,0,:].unsqueeze(1)
        CT = x[:,1,:].unsqueeze(1)
        Dose = x[:,2,:].unsqueeze(1)

        CBCT = self.alexnet(CBCT)
        CT = self.alexnet(CT)
        Dose = self.alexnet(Dose)

        out = torch.cat([CBCT, CT, Dose], dim=1)
        del CBCT
        del CT
        del Dose

        out = self.conv_norm_lrelu_l4(out)

        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)

        y_out = self.globa_Avg(out)
        y_out = torch.flatten(y_out, 1)
        y_out = self.lrelu(self.fc1(self.dropout(y_out)))
        y_out = self.fc2(self.dropout(y_out))

        return y_out

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 3, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("ForkNet test is complete")
