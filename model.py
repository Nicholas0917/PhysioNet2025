import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F
from models.net1d import Net1D, Swish
from models.ECGFeatureExtractor import ECGFeatureExtractor, GELU
from models.SEResNet import ResNetFeatureExtractor, BasicBlock, Bottleneck

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
                dropout_rate=self.config.model['net1d_dropout_rate']
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
                drop_path_rate=self.config.model['net1d_dropout_rate'], # Drop path rate for ConvNeXt
                n_classes=1, # For feature extraction, output 1 class for compatibility, but we only use features
                return_features=True # Ensure features are returned
            )
            self.base_model_output_dim = 1024 # Output feature dimension for ECGFeatureExtractor (last filter in filter_list)
        elif self.base_model_name == 'ResNet18':
            # ResNet as feature extractor, removing its classification head
            self.base_model = ResNetFeatureExtractor(BasicBlock, [2, 2, 2, 2], in_channel=12, dropout_rate=self.config.model['dropout_rate'])
            self.base_model_output_dim = 512 * BasicBlock.expansion # Output feature dimension for ResNet18
        elif self.base_model_name == 'ResNet34':
            self.base_model = ResNetFeatureExtractor(BasicBlock, [3, 4, 6, 3], in_channel=12, dropout_rate=self.config.model['dropout_rate'])
            self.base_model_output_dim = 512 * BasicBlock.expansion
        elif self.base_model_name == 'ResNet50':
            self.base_model = ResNetFeatureExtractor(Bottleneck, [3, 4, 6, 3], in_channel=12, dropout_rate=self.config.model['dropout_rate'])
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
            nn.Dropout(self.config.model['dropout_rate']),
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
        
        initialize_submodules = (pth_path is None and not self.config.model['use_pretrained']) # Only initialize weights if no pth_path is provided and not using pretrained model

        self.encoder = Encoder(device, config, initialize_weights=initialize_submodules)
        encoder_output_dim = self.encoder.base_model_output_dim + 256

        self.classifier = Classifier(encoder_output_dim, self.config.model['dropout_rate'], initialize_weights=initialize_submodules)
        self.domain_classifier = DomainClassifier(encoder_output_dim, self.config.model['dropout_rate'], num_domains=self.config.dann['num_domains'], initialize_weights=initialize_submodules) # Use config.dann['num_domains']
        
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
