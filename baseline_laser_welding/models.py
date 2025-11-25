import torch.nn as nn
from torch.hub import load_state_dict_from_url
import timm
from timm.layers import Mlp
from typing import Tuple
import torch
import torch.nn.functional as F
import math

__all__ = ['resnet3d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    if isinstance(stride, int):
        stride = (1, stride, stride)
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, groups=groups, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    if isinstance(stride, int):
        stride = (1, stride, stride)
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, modality='RGB'):
        super(ResNet3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.modality = modality
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self._make_stem_layer()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, 4)

        for m in self.modules():  # self.modules() --> Depth-First-Search the Net
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        if self.modality == 'RGB':
            inchannels = 3
        elif self.modality == 'Flow':
            inchannels = 2
        elif self.modality == 'Grayscale':
            inchannels = 1
        else:
            raise ValueError('Unknown modality: {}'.format(self.modality))
        self.conv1 = nn.Conv3d(inchannels, self.inplanes, kernel_size=(5, 7, 7),
                               stride=2, padding=(2, 3, 3), bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=2,
                                    padding=(0, 1, 1))  # kernel_size=(2, 3, 3)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d,
                             inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d models.
            module_name_2d (str): The name of corresponding conv module in the
                2d models.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + '.weight'

        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _inflate_bn_params(self, bn3d, state_dict_2d, module_name_2d,
                           inflated_param_names):
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d models.
            module_name_2d (str): The name of corresponding bn module in the
                2d models.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            # some buffers like num_batches_tracked may not exist in old
            # checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    def inflate_weights(self, state_dict_r2d):
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d models,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        """

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.BatchNorm3d):
                if name + '.weight' not in state_dict_r2d:
                    print(f'Module not exist in the state_dict_r2d: {name}')
                else:
                    shape_2d = state_dict_r2d[name + '.weight'].shape
                    shape_3d = module.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        print(f'Weight shape mismatch for: {name}'
                              f'3d weight shape: {shape_3d}; '
                              f'2d weight shape: {shape_2d}. ')
                    else:
                        if isinstance(module, nn.Conv3d):
                            self._inflate_conv_params(module, state_dict_r2d, name, inflated_param_names)
                        else:
                            self._inflate_bn_params(module, state_dict_r2d, name, inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            print(f'These parameters in the 2d checkpoint are not loaded: {remaining_names}')


def resnet3d(arch, progress=True, modality='Grayscale', pretrained2d=True, **kwargs):
    """
    Args:
        arch (str): The architecture of resnet
        modality (str): The modality of input, 'RGB' or 'Flow'
        progress (bool): If True, displays a progress bar of the download to stderr
        pretrained2d (bool): If True, utilize the pretrained parameters in 2d models
    """

    arch_settings = {
        'resnet_tiny': (BasicBlock3d, (1, 1, 1, 1)),
        'resnet18': (BasicBlock3d, (2, 2, 2, 2)),
        'resnet34': (BasicBlock3d, (3, 4, 6, 3)),
        'resnet50': (Bottleneck3d, (3, 4, 6, 3)),
        'resnet101': (Bottleneck3d, (3, 4, 23, 3)),
        'resnet152': (Bottleneck3d, (3, 8, 36, 3))
    }

    model = ResNet3d(*arch_settings[arch], modality=modality, **kwargs)
    if pretrained2d:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.inflate_weights(state_dict)
    return model


class Conv2DClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model('timm/mobilenetv3_small_100.lamb_in1k', pretrained=True, num_classes=0, 
                                          in_chans=in_channels)
        self.mlp = Mlp(1024, 256, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        return x
    

class NNClassifier(nn.Module):
    def __init__(self, config: Tuple[int, int] = (8, 64, 32, 1)) -> None:
        super().__init__()
        module_list = []
        for i in range(len(config) - 2):
            module_list.append(nn.Linear(config[i], config[i + 1]))
            module_list.append(nn.ReLU())
            # module_list.append(nn.Dropout())
        module_list.append(nn.Linear(config[-2], config[-1]))  # last layer
        self.layers = nn.Sequential(*module_list)

    def forward(self, x):
        x = self.layers(x)
        return x


class TabularMLP(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.num_labels = num_labels
        
        # Основной блок обработки признаков
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # Многоголовая система для разных меток
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ) for _ in range(num_labels)
        ])
        
        # Промежуточные классификаторы для каждой метки
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim // 4, hidden_dim // 8) for _ in range(num_labels)
        ])
        
        # Размер комбинированных признаков: 
        # (hidden_dim // 8 * num_labels) + (hidden_dim // 2)
        combined_features_dim = (hidden_dim // 8 * num_labels) + (hidden_dim // 2)
        
        # Финальный классификатор
        self.final_classifier = nn.Sequential(
            nn.Linear(combined_features_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
    def forward(self, x):
        # Основной проход через feature net
        features = self.feature_net(x)
        
        # Применяем специализированные головы и классификаторы
        head_outputs = []
        for i, head in enumerate(self.attention_heads):
            head_out = head(features)
            head_out = self.classifiers[i](head_out)
            head_outputs.append(head_out)
        
        # Объединяем выходы всех голов
        specialized_output = torch.cat(head_outputs, dim=1)
        
        # Комбинируем специализированные выходы с основными признаками
        combined_features = torch.cat([specialized_output, features], dim=1)
        
        # Финальная классификация
        final_output = self.final_classifier(combined_features)
        
        return final_output
    

class CNNTimeSeriesClassifier(nn.Module):
    """Чисто CNN модель для классификации временных рядов"""
    
    def __init__(self, num_sensors, seq_length, num_classes, 
                 conv_filters=[64, 128, 256], kernel_sizes=[7, 5, 3],
                 dropout_rate=0.3, use_batch_norm=True):
        super(CNNTimeSeriesClassifier, self).__init__()
        
        self.num_sensors = num_sensors
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        # CNN layers
        self.conv_blocks = nn.ModuleList()
        
        # Первый блок
        self.conv_blocks.append(
            self._make_conv_block(num_sensors, conv_filters[0], kernel_sizes[0], 
                                dropout_rate, use_batch_norm)
        )
        
        # Последующие блоки
        for i in range(1, len(conv_filters)):
            self.conv_blocks.append(
                self._make_conv_block(conv_filters[i-1], conv_filters[i], kernel_sizes[i],
                                    dropout_rate, use_batch_norm)
            )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate the size after convolutions for the dense layer
        # Простой способ: прогоним фиктивный тензор чтобы узнать размер
        with torch.no_grad():
            dummy_input = torch.randn(1, num_sensors, seq_length)
            dummy_output = self._forward_conv(dummy_input)
            conv_output_size = dummy_output.view(1, -1).size(1)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(conv_filters[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size, dropout, use_bn):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     padding=kernel_size//2, bias=not use_bn),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        
        if use_bn:
            layers.insert(1, nn.BatchNorm1d(out_channels))
        
        layers.append(nn.MaxPool1d(2))  # Downsample by 2
        
        return nn.Sequential(*layers)
    
    def _forward_conv(self, x):
        # Forward pass through convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, num_sensors)
        # Convert to: (batch_size, num_sensors, seq_length) for Conv1d
        x = x.transpose(1, 2)
        
        # Feature extraction with CNN
        features = self._forward_conv(x)
        
        # Global pooling
        pooled = self.global_pool(features)  # (batch_size, channels, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch_size, channels)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class HybridCNNTransformerMultilabelV2(nn.Module):
    """Упрощенная версия с общей CNN"""
    
    def __init__(self, num_sensors, seq_len, num_classes,
                 cnn_hidden_dims=[64, 128, 256],
                 d_model=256, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1):
        super(HybridCNNTransformerMultilabelV2, self).__init__()
        
        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Общая CNN для всех сенсоров
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(num_sensors, cnn_hidden_dims[0], 7, padding=3),
            nn.BatchNorm1d(cnn_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_hidden_dims[0], cnn_hidden_dims[1], 5, padding=2),
            nn.BatchNorm1d(cnn_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(cnn_hidden_dims[1], cnn_hidden_dims[2], 3, padding=1),
            nn.BatchNorm1d(cnn_hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),

            # nn.Conv1d(cnn_hidden_dims[2], cnn_hidden_dims[3], 3, padding=1),
            # nn.BatchNorm1d(cnn_hidden_dims[3]),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool1d(100)
        )
        
        # Projection to transformer dimension
        self.input_projection = nn.Linear(cnn_hidden_dims[-1], d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_sensors)
        batch_size = x.size(0)
        
        # 1. Конвертируем для CNN: (batch_size, num_sensors, seq_len)
        x = x.transpose(1, 2)
        
        # 2. Извлечение признаков с помощью CNN
        cnn_features = self.cnn_extractor(x)  # (batch_size, cnn_hidden_dims[-1], 100)
        cnn_features = cnn_features.transpose(1, 2)  # (batch_size, 100, cnn_hidden_dims[-1])
        
        # 3. Проекция в пространство transformer
        projected_features = self.input_projection(cnn_features)
        
        # 4. Позиционное кодирование
        projected_features = self.pos_encoder(projected_features.transpose(0, 1)).transpose(0, 1)
        
        # 5. Transformer encoder
        transformer_output = self.transformer_encoder(projected_features)
        
        # 6. Attention pooling
        attention_weights = F.softmax(
            self.attention_pool(transformer_output).squeeze(-1), dim=-1
        )
        global_representation = torch.sum(
            transformer_output * attention_weights.unsqueeze(-1), dim=1
        )
        
        # 7. Multilabel классификация
        logits = self.classifier(global_representation)
        
        return logits
    

class MultiLayerBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes) # *2 to account for bidirectional LSTM

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        # h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device) # *2 to account for bidirectional LSTM
        # c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(x.device) # *2 to account for bidirectional LSTM
        # Forward propagate bidirectional LSTM
        out, _ = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    

class PositionalEncodingV2(nn.Module):
    """Позиционное кодирование для Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingV2, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    
    def __init__(self, input_dim, seq_len, num_classes,
                 d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Проекция входных данных в пространство трансформера
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncodingV2(d_model, max_len=seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Важно: (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention pooling для агрегации временной оси
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        # 1. Проекция в пространство трансформера
        x_proj = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 2. Добавление позиционного кодирования
        x_pos = self.pos_encoder(x_proj)
        
        # 3. Применение dropout
        x_pos = self.dropout(x_pos)
        
        # 4. Transformer encoder
        transformer_output = self.transformer_encoder(x_pos)  # (batch_size, seq_len, d_model)
        
        # 5. Attention pooling для получения глобального представления
        attention_weights = torch.softmax(
            self.attention_pool(transformer_output).squeeze(-1), dim=-1
        )  # (batch_size, seq_len)
        
        # Взвешенная сумма по временной оси
        global_repr = torch.sum(
            transformer_output * attention_weights.unsqueeze(-1), dim=1
        )  # (batch_size, d_model)
        
        # 6. Классификация
        logits = self.classifier(global_repr)  # (batch_size, num_classes)
        
        return logits
    
