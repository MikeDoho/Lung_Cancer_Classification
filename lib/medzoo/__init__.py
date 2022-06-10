import torch.optim as optim

from .Densenet3D import DualPathDenseNet, DualSingleDenseNet, SinglePathDenseNet
from .ResNet3DMedNet import generate_resnet3d
from .AlexNet import AlexNet
from .ResNet import ResNet, BasicBlock, Bottleneck
from .MobileNet import MobileNetV2
from .ShuffleNet import ShuffleNetV2
from .SqueezeNet import SqueezeNet
from .ForkNet import ForkNet
from .TinyNet import TinyNet

model_list = ['TinyNet','ForkNet','AlexNet','AlexNet2', 'VNet_3', 'VNetLight_3','V_YNET', 'V_YNET2',  'V_YNET2_2', 'U_YNET3D', 'UNET3D', 'U_YNET3D_2', 'U_YNET3D_3', 'UNET3D', 'DENSENET1', "UNET2D",
              "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
              "HIGHRESNET",'ResNet18','ResNet50','ResNet18_2','ResNet50_2','MobileNetV2','MobileNetV2_2','ShuffleNetV2','SqueezeNet']


def create_model(args):
    model_name = args.model
    assert model_name in model_list
    in_channels = args.inChannels
    num_classes = args.classes
    if (args.test is not None) and (args.test == True):
        print("Building Model . . . . . . . ." + model_name)
        if model_name == 'V_YNET2_2':
            model = VNetLight_2(in_channels=in_channels, elu=False, classes=num_classes, show_active = args.show_active)
        elif model_name == 'U_YNET3D_2':
            model = UNet3D_2(in_channels=in_channels, n_classes=num_classes, base_n_filter=8, show_active = args.show_active)
        
        print("Model created!")
        return model
    else:
        lr = args.lr
        optimizer_name = args.opt
        weight_decay = 1e-3
        print("Building Model . . . . . . . ." + model_name)

        if  model_name == 'TinyNet':
            model = TinyNet(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
        elif  model_name == 'ForkNet':
            model = ForkNet(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
        elif  model_name == 'AlexNet':
            model = AlexNet(in_channels=in_channels, n_classes=num_classes, base_n_filter=8)
        elif model_name == 'AlexNet2':
            model = AlexNet(in_channels=in_channels*2, n_classes=num_classes, base_n_filter=8)
        elif model_name == 'ShuffleNetV2':
            model = ShuffleNetV2(in_ch=in_channels, num_classes=num_classes, width_mult=0.25)
        elif model_name == 'SqueezeNet':
            model = SqueezeNet(version=1.1, in_channel=in_channels, num_classes=num_classes)
        elif model_name == 'MobileNetV2':
            model = MobileNetV2(num_classes=num_classes, width_mult=0.5, input_channel=in_channels, last_channel=256)
        elif model_name == 'MobileNetV2_2':
            model = MobileNetV2(num_classes=num_classes, width_mult=0.5, input_channel=in_channels*2, last_channel=256)
        elif model_name == 'ResNet18':
            model = ResNet(BasicBlock, [1, 1, 1, 1], n_input_channels=in_channels)#ResNet10
        elif model_name == 'ResNet50':
            model = ResNet(Bottleneck, [3, 4, 6, 3], n_input_channels=in_channels)
        elif model_name == 'ResNet18_2':
            model = ResNet(BasicBlock, [1, 1, 1, 1], n_input_channels=in_channels*2)
        elif model_name == 'ResNet50_2':
            model = ResNet(Bottleneck, [3, 4, 6, 3], n_input_channels=in_channels*2)
        elif model_name == 'DENSENET1':
            model = SinglePathDenseNet(in_channels=in_channels, classes=num_classes)
        elif model_name == 'DENSENET2':
            model = DualPathDenseNet(in_channels=in_channels, classes=num_classes)
        elif model_name == 'DENSENET3':
            model = DualSingleDenseNet(in_channels=in_channels, drop_rate=0.1, classes=num_classes)
        
        print(model_name, 'Number of params: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamax':
            optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)



        print("Model and optimizer created!")
        return model, optimizer
