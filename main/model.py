import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalBriefNet(nn.Module):

    def __init__(self, n_classes=2):
        super(LocalBriefNet, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv5x5 = nn.Conv3d(1, 32, 5)
        self.conv3x3 = nn.Conv3d(32, 32, 3)
        self.last_conv = nn.Conv3d(32, 2, 3)
        self.fc = nn.Linear(2 * 24 * 30 * 24, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv5x5(x))
        x = self.pool(x)
        x = F.relu(self.conv3x3(x))
        x = F.relu(self.conv3x3(x))
        x = F.relu(self.conv3x3(x))
        x = self.pool(x)

        x = F.relu(self.last_conv(x))
        x = x.view(-1, 2 * 24 * 30 * 24)
        x = self.fc(x)
        return x


class joint_model(nn.Module):      
    def __init__(self, tab_in_shape, enc_shape = 8, n_classes = 3, classifier='vgg'):
        super(joint_model, self).__init__()

        self.autoencoder = Autoencoder(in_shape = tab_in_shape, out_cls = n_classes, enc_shape = 8)
        if classifier == 'vgg':
            self.classifier = VGG(n_classes= n_classes)

        self.layer1 = nn.Sequential(
            nn.Linear(64 + 32 * 3 * 4 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2))

        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2))
        self.linear = nn.Linear(128, n_classes)

    def forward(self, img, tab):

        feat_emb1 = self.autoencoder.feature_extractor(tab)
        feat_emb2 = self.classifier.feature_extractor(img)

        feat = torch.cat([feat_emb1, feat_emb2], dim = 1)
        x = self.layer1(feat)
        x = self.layer2(x)
        x = self.linear(x)
        return x


class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, out_cls, enc_shape = 8):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, enc_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        self.linear = nn.Linear(64, out_cls)
        
    def feature_extractor(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.linear(x)
        return x

class VGG(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 32*2, 3, padding=1)
        self.conv2 = nn.Conv3d(32*2, 64*2, 3, padding=1)
        self.conv3 = nn.Conv3d(64*2, 128*2, 3, padding=1)
        self.conv4 = nn.Conv3d(128*2, 128*2, 3, padding=1)
        self.conv5 = nn.Conv3d(128*2, 256*2, 3, padding=1)
        self.conv6 = nn.Conv3d(256*2, 256*2, 3, padding=1)
        self.conv8 = nn.Conv3d(256*2, 256*2, 3, padding=1)
        self.conv9 = nn.Conv3d(256*2, 256*2, 3, padding=1)
        self.conv7 = nn.Conv3d(256*2, 32, 1)
        self.fc1 = nn.Linear(32 * 3 * 4 * 3, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def feature_extractor(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.pool(x)

        x = F.relu(self.conv7(x))
        x = x.view(-1, 32 * 3 * 4 * 3)
        
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNModel(nn.Module):
    def __init__(self,n_classes):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.conv_layer3 = self._conv_layer_set(64, 128)
        self.conv_layer4 = self._conv_layer_set(128, 128)
        self.fc1 = nn.Linear(7*9*7*128, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)       
        self.flatten = nn.Flatten() 
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    

    def feature_extractor(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.flatten(out)
        return out

    def forward(self,x):
        out = self.feature_extractor(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out
        