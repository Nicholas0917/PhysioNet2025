import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F
from models.basic_blocks import BasicBlock, Bottleneck, ResNetFeatureExtractor
from models.net1d import Net1D, Swish
from models.ECGFeatureExtractor import ECGFeatureExtractor, GELU

class HybridModel(nn.Module):
    def __init__(self, device, pth_path, config=None):
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
            base_model_output_dim = 1024 # Output feature dimension for Net1D
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
                drop_path_rate=0.0, # Example drop path rate, set to 0 to disable
                n_classes=1, # For feature extraction, output 1 class for compatibility, but we only use features
                return_features=True # Ensure features are returned
            )
            base_model_output_dim = 1024 # Output feature dimension for ECGFeatureExtractor (last filter in filter_list)
        elif self.base_model_name == 'ResNet18':
            # ResNet as feature extractor, removing its classification head
            self.base_model = ResNetFeatureExtractor(BasicBlock, [2, 2, 2, 2], in_channel=12, config=self.config)
            base_model_output_dim = 512 * BasicBlock.expansion # Output feature dimension for ResNet18
        elif self.base_model_name == 'ResNet34':
            self.base_model = ResNetFeatureExtractor(BasicBlock, [3, 4, 6, 3], in_channel=12, config=self.config)
            base_model_output_dim = 512 * BasicBlock.expansion
        elif self.base_model_name == 'ResNet50':
            self.base_model = ResNetFeatureExtractor(Bottleneck, [3, 4, 6, 3], in_channel=12, config=self.config)
            base_model_output_dim = 512 * Bottleneck.expansion
        else:
            raise ValueError(f"Unsupported base_model_name: {self.base_model_name}. Please check config.model_name.")

        if pth_path is not None:
            try:
                checkpoint = torch.load(pth_path, map_location=device)
                # Handle both full checkpoint and state_dict cases
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                                if not k.startswith('dense.') and not k.startswith('fc.') and not k.startswith('fc1.')} # Exclude classifier layers
                else:
                    state_dict = {k: v for k, v in checkpoint.items() 
                                if not k.startswith('dense.') and not k.startswith('fc.') and not k.startswith('fc1.')}
                self.base_model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded pretrained weights from {pth_path} for {self.base_model_name}.")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights from {pth_path} for {self.base_model_name}. Error: {e}. Initializing with Kaiming normal.")
                self._initialize_weights(self.base_model)
        else:
            print(f"pth_path is None. Initializing {self.base_model_name} with Kaiming normal.")
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
        
        self.classifier = nn.Sequential(
            nn.Linear(base_model_output_dim + 256, 512), # Use dynamic output dim
            nn.BatchNorm1d(512, eps=1e-4),
            Swish(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 1)
        )
        
        self._initialize_weights(self.meta_net)
        self._initialize_weights(self.classifier)
        
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

    def forward(self, x, meta_features=None):
        
        # Handle different return patterns for different model types
        if self.base_model_name in ['ResNet18', 'ResNet34', 'ResNet50']:
            # ResNet models return only features
            signal_features = self.base_model(x)
        else:
            # Net1D and ECGFeatureExtractor return (logits, features)
            _, signal_features = self.base_model(x)
        
        if meta_features is not None:
            if meta_features.dim() == 1:
                meta_features = meta_features.unsqueeze(0)
            meta_features = self.meta_net(meta_features)
            
            features = torch.cat([signal_features, meta_features], dim=1)
        else:
            features = signal_features
            
        output = self.classifier(features)
            
        return output