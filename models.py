import torch
import torch.nn as nn
import torchvision.models as models

#This class is for setup the training, and could be used for dry run
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.fc_input_size = self._get_fc_input_size()        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_fc_input_size(self):
        x = torch.randn(1, 3, 256, 256)  # (batch_size, channels, height, width)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.view(x.size(0), -1).size(1)  # Adjusted based on the size after convolution and pooling layers

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, num_classes, num_layers_to_retrain=10):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)        
        start_layer = len(self.densenet.features) - num_layers_to_retrain
        
        # Freeze layers up to start_layer
        for i, param in enumerate(self.densenet.features.parameters()):
            if i < start_layer:
                param.requires_grad = False
        
        # Modify the last layer to match the number of classes
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

class VGG19(nn.Module):
    def __init__(self, num_classes, num_layers_to_retrain=10):
        super(VGG19, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        
        # Freeze layers up to start_layer
        start_layer = len(self.vgg.features) - num_layers_to_retrain
        for i, param in enumerate(self.vgg.features.parameters()):
            if i < start_layer:
                param.requires_grad = False
        
        # Modify the last layer to match the number of classes
        in_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.vgg(x)
         
class ModelFactory:
    @staticmethod
    def create_model(model_type, num_classes, num_layers_to_retrain=10):
        if model_type == 'MyModel':
            return MyModel(num_classes)
        elif model_type == 'DenseNet':
            return DenseNet(num_classes, num_layers_to_retrain)
        elif model_type == 'VGG19':
            return VGG19(num_classes, num_layers_to_retrain)          
        else:
            raise ValueError('Invalid model type')
