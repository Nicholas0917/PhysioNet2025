import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F
from net1d import Net1D, Swish
from ECGFeatureExtractor import ECGFeatureExtractor, GELU

class Encoder(nn.Module):
    def __init__(self, device, config=None, initialize_weights=True):
        super().__init__()
        self.config = config
        self.base_model_name = self.config.model_name

        if self.base_model_name == 'ecgfounder':
            self.base_model = Net1D(
                in_channels=12,
                base_filters=64,
                ratio=1,
                filter_list=[64,160,160,400,400,1024,1024],
                m_blocks_list=[2,2,2,3,3,4,4],
                kernel_size=16,
                stride=2,
                groups_width=16,
                verbose=False,
                use_bn=True,
                use_do=True,
                n_classes=1,
                return_features=True,
                dropout_rate=self.config.net1d_dropout_rate
            )
            self.base_model_output_dim = 1024 # Output feature dimension for Net1D
        elif self.base_model_name == 'ECGFeatureExtractor': # New branch for ECGFeatureExtractor
            self.base_model = ECGFeatureExtractor(
                in_channels=12, 
                base_filters=96, # ConvNeXt base filters
                expansion_ratio=4, # Inverted bottleneck expansion ratio
                filter_list=[128,256,512,1024],    # ConvNeXt channels
                m_blocks_list=[3,3,9,3],   # ConvNeXt depths
                kernel_size=16, 
                stride=2, 
                groups_width=1, # This parameter is not directly used for groups in BasicBlock due to depthwise conv
                verbose=False, # Set to False for deployment
                drop_path_rate=self.config.net1d_dropout_rate, # Drop path rate for ConvNeXt
                n_classes=1, # For feature extraction, output 1 class for compatibility, but we only use features
                return_features=True # Ensure features are returned
            )
            self.base_model_output_dim = 1024 # Output feature dimension for ECGFeatureExtractor (last filter in filter_list)
        elif self.base_model_name == 'ResNet18':
            # ResNet as feature extractor, removing its classification head
            self.base_model = ResNetFeatureExtractor(BasicBlock, [2, 2, 2, 2], in_channel=12, config=self.config)
            self.base_model_output_dim = 512 * BasicBlock.expansion # Output feature dimension for ResNet18
        elif self.base_model_name == 'ResNet34':
            self.base_model = ResNetFeatureExtractor(BasicBlock, [3, 4, 6, 3], in_channel=12, config=self.config)
            self.base_model_output_dim = 512 * BasicBlock.expansion
        elif self.base_model_name == 'ResNet50':
            self.base_model = ResNetFeatureExtractor(Bottleneck, [3, 4, 6, 3], in_channel=12, config=self.config)
            self.base_model_output_dim = 512 * Bottleneck.expansion
        else:
            raise ValueError(f"Unsupported base_model_name: {self.base_model_name}. Please check config.model_name.")


        if initialize_weights:
            print(f"Initializing {self.base_model_name} with Kaiming normal.")
            self._initialize_weights(self.base_model)
        
        meta_dim = self.config.get_meta_feature_dim()
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.BatchNorm1d(128, eps=1e-3),
            Swish(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, eps=1e-3),
            Swish()
        )
        if initialize_weights:
            self._initialize_weights(self.meta_net)
        self.to(device)

    def _initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, signal, meta_feature=None):
        if torch.isnan(signal).any():
            print("NaN detected in input signal to Encoder!")
            import sys
            sys.exit(1)

        if self.base_model_name in ['ResNet18', 'ResNet34', 'ResNet50']:
            signal_features = self.base_model(signal)
        else:
            _, signal_features = self.base_model(signal)
        
        if torch.isnan(signal_features).any():
            print("NaN detected in base_model output (signal_features)!")
            import sys
            sys.exit(1)

        if meta_feature is not None:
            if torch.isnan(meta_feature).any():
                print("NaN detected in input meta_feature to Encoder!")
                import sys
                sys.exit(1)
            if meta_feature.dim() == 1:
                meta_feature = meta_feature.unsqueeze(0)
            meta_feature = self.meta_net(meta_feature)
            if torch.isnan(meta_feature).any():
                print("NaN detected in meta_net output!")
                import sys
                sys.exit(1)
            features = torch.cat([signal_features, meta_feature], dim=1)
        else:
            features = signal_features
        return features

class Classifier(nn.Module):
    def __init__(self, input_dim, dropout_rate, initialize_weights=True):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512, eps=1e-4),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)
        )
        if initialize_weights:
            self._initialize_weights(self.classifier)

    def _initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        return self.classifier(features)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, dropout_rate, num_domains, initialize_weights=True):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512, eps=1e-4),
            Swish(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_domains)
        )
        if initialize_weights:
            self._initialize_weights(self.domain_classifier)

    def _initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        return self.domain_classifier(features)

class HybridModel(nn.Module):
    def __init__(self, device, config=None, pth_path=None): # Added pth_path parameter
        super().__init__()
        self.config = config
        self.pth_path = pth_path # Store pth_path if needed for internal loading, though typically handled externally
        
        initialize_submodules = (pth_path is None) # Only initialize weights if no pth_path is provided

        self.encoder = Encoder(device, config, initialize_weights=initialize_submodules)
        encoder_output_dim = self.encoder.base_model_output_dim + 256

        self.classifier = Classifier(encoder_output_dim, self.config.dropout_rate, initialize_weights=initialize_submodules)
        self.domain_classifier = DomainClassifier(encoder_output_dim, self.config.dropout_rate, num_domains=self.config.num_domains, initialize_weights=initialize_submodules) # Use config.num_domains
        
        self.to(device)
        
        if self.pth_path is not None:
            print(f"Loading model weights from {self.pth_path}")
            loaded_data = torch.load(self.pth_path, map_location=device)
            if "state_dict" in loaded_data:
                self.load_state_dict(loaded_data["state_dict"])
            else:
                self.load_state_dict(loaded_data)
            print("Model weights loaded successfully.")

    def forward(self, signal, meta_features=None):
        features = self.encoder(signal, meta_features)
        if torch.isnan(features).any():
            print("NaN detected in encoder output (features)!")
            import sys
            sys.exit(1)
        
        task_output = self.classifier(features)
        if torch.isnan(task_output).any():
            print("NaN detected in classifier output (task_output)!")
            import sys
            sys.exit(1)

        domain_output = self.domain_classifier(features)
        if torch.isnan(domain_output).any():
            print("NaN detected in domain classifier output (domain_output)!")
            import sys
            sys.exit(1)
            
        return task_output, domain_output, features

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, config=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, config=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.se = SELayer(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, block, layers, in_channel=12, config=None):
        super(ResNetFeatureExtractor, self).__init__()
        self.config = config
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Removed fc1 and fc layers as this is a feature extractor
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, config=self.config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, config=self.config))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x.shape = [batch_size, 12, 4096]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # Flatten the features
        return x
    
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatureExtractor(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # This part might need adjustment if pre-trained weights are for full ResNet
        # and not just feature extractor. For now, keep as is.
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18'])
        # Filter out classifier weights if loading full ResNet pretrained model
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatureExtractor(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet34'])
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatureExtractor(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50'])
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatureExtractor(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet101'])
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatureExtractor(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet152'])
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    return model
