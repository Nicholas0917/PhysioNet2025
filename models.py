import torch
import torch.nn as nn
from net1d import Net1D, Swish

class HybridModel(nn.Module):
    def __init__(self, device, pth_path, config=None):
        super().__init__()
        self.config = config
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
        
        checkpoint = torch.load(pth_path, map_location=device)
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                     if not k.startswith('dense.')}
        self.base_model.load_state_dict(state_dict, strict=False)
        
        meta_dim = self.config.get_meta_feature_dim()
        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.BatchNorm1d(128, eps=1e-4),
            Swish(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, eps=1e-4),
            Swish()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 256, 512),
            nn.BatchNorm1d(512, eps=1e-4),
            Swish(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 1)
        )
        
        for m in list(self.meta_net.modules()) + list(self.classifier.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.to(device)

    def forward(self, x, meta_features=None):
        _, signal_features = self.base_model(x)
        
        if torch.isnan(signal_features).any() or torch.isinf(signal_features).any():
            print("WARNING: BaseModel output contains NaN/Inf values")
        
        if meta_features is not None:
            if meta_features.dim() == 1:
                meta_features = meta_features.unsqueeze(0)
            meta_features = self.meta_net(meta_features)
            
            if torch.isnan(meta_features).any() or torch.isinf(meta_features).any():
                print("WARNING: MetaNet output contains NaN/Inf values")
            
            features = torch.cat([signal_features, meta_features], dim=1)
        else:
            features = signal_features
            
        output = self.classifier(features)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("WARNING: Classifier output contains NaN/Inf values")
            
        return output
