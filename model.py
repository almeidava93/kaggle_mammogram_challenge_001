import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from config import MammogramClassifierConfig

class RMSNorm(nn.Module):
    """
    RMSNorm normalizes activations based on the root mean square of the activations themselves, rather than using mini-batch or layer statistics. This approach ensures that the activations are consistently scaled regardless of the mini-batch size or the number of features. Additionally, RMSNorm introduces learnable scale parameters, offering similar adaptability to BN.

    Reference:
    - https://arxiv.org/abs/1910.07467
    - https://2020machinelearning.medium.com/deep-dive-into-deep-learning-layers-rmsnorm-and-batch-normalization-b2423552be9f
    """
    def __init__(self, dim: int, eps: float = 1e-6, frac: float = 1.0, dims_to_apply_to: list[int] = [-1]):
        super().__init__()
        self.eps = eps
        self.frac = frac
        self.weight = nn.Parameter(torch.ones(dim))  # Initialized to ones
        self.dims_to_apply_to = dims_to_apply_to

    def forward(self, x):
        if self.frac < 1.0:
            partial_size = int(x.size(-1) * self.frac)
            rms = torch.rsqrt(torch.mean(x[..., :partial_size] ** 2, dim=self.dims_to_apply_to, keepdim=True) + self.eps)
        else:
            rms = torch.rsqrt(torch.mean(x ** 2, dim=self.dims_to_apply_to, keepdim=True) + self.eps)

        normed = x * rms
        output = normed * (1 + self.weight.to(x.dtype).unsqueeze(0))
        return output
    

# setup model
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, activation, stride=1, n_conv=2, use_rms_norm=False, rms_norm_dim=None, cnn_rms_norms_dims_to_apply=None):
        super(ResNetBlock, self).__init__()
        self.dropout = None
        self.activation = getattr(nn, activation)()
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )
            if use_rms_norm:
                self.shortcut.append(RMSNorm(rms_norm_dim, dims_to_apply_to=cnn_rms_norms_dims_to_apply))
            else:
                self.shortcut.append(nn.BatchNorm2d(out_channels))

        self.pipeline = nn.Sequential()
        for idx in range(0, n_conv):
            # If last conv layer
            if idx == n_conv - 1:
                self.pipeline.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

                if use_rms_norm:
                    self.pipeline.append(RMSNorm(rms_norm_dim, dims_to_apply_to=cnn_rms_norms_dims_to_apply))
                else:
                    self.pipeline.append(nn.BatchNorm2d(out_channels))
                
                if dropout > 0:
                    self.pipeline.append(nn.Dropout(dropout))
                
                break

            # if the first conv layer
            if idx == 0:
                self.pipeline.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
                if use_rms_norm:
                    self.pipeline.append(RMSNorm(rms_norm_dim, dims_to_apply_to=cnn_rms_norms_dims_to_apply))
                else:
                    self.pipeline.append(nn.BatchNorm2d(out_channels))

                self.pipeline.append(self.activation)
                continue
            
            # middle conv layers
            self.pipeline.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            if use_rms_norm:
                self.pipeline.append(RMSNorm(rms_norm_dim, dims_to_apply_to=cnn_rms_norms_dims_to_apply))
            else:
                self.pipeline.append(nn.BatchNorm2d(out_channels))
            self.pipeline.append(self.activation)

        
    def forward(self, x):
        out = self.pipeline(x)
        out += self.shortcut(x)
        out = self.activation(out)

        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int, dropout: float, activation: str = 'ReLU'):
        super().__init__()
        self.activation = getattr(nn, activation)()
        self.ffn = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_features),
            self.activation,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.ffn(x)
        return x
    

class MammogramScreeningClassifier(nn.Module):
    def __init__(self, 
                 config: MammogramClassifierConfig,
                 ):
        super().__init__()

        self.config = config
        self.feature_dim = config.feature_dim
        self.max_images_per_study = config.max_images_per_study
        self.img_metadata_cat_cols = config.img_metadata_cat_cols

        if config.use_vit_b_16:
            print("Using ViT-B/16 as pretrained backbone")
            base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

            # freeze weights
            if config.freeze_pretrained_weights:
                for param in base_model.parameters():
                    print("Freezing ViT-B/16 weights")
                    param.requires_grad = False

            # replace head
            print("Replacing ViT-B/16 head")
            base_model.heads.head = nn.Linear(768, 256)

            self.pipeline = base_model
            self.projector = nn.Sequential()

        else:
            # Configure ResNets by default
            resnet_blocks = []
            num_init_features = config.num_img_init_features
            rms_norm_dim = config.img_size//2

            # Add first block
            resnet_blocks.append(
                ResNetBlock(config.num_img_channels, num_init_features, stride=2, rms_norm_dim=rms_norm_dim, dropout=config.cnn_dropout, activation=config.cnn_activation)
            )

            for _ in range(config.cnn_resnet_n_blocks):
                resnet_blocks.append(
                    ResNetBlock(num_init_features,   num_init_features*2,  stride=2, rms_norm_dim=rms_norm_dim//2, dropout=config.cnn_dropout, activation=config.cnn_activation),
                )
                num_init_features *= 2
                rms_norm_dim = rms_norm_dim//2

            self.pipeline = nn.Sequential(
                *resnet_blocks,
                nn.AdaptiveAvgPool2d((1, 1)),  # ensure fixed output size
            )

            # Flatten CNN output to vector
            self.projector = nn.Linear(num_init_features, config.feature_dim)

        # Embeddings for images metadata
        self.meta_embeddings = nn.ModuleDict()
        for cat in config.img_metadata_cat_cols:
            if cat == 'PatientAge':
                config.classes_per_cat[cat] = 1
                self.meta_embeddings[cat] = nn.Linear(1, config.feature_dim)
                continue

            if config.add_linear_proj_to_embeddings:
                self.meta_embeddings[cat] = nn.Sequential(
                    nn.Embedding(config.classes_per_cat[cat], config.feature_dim),
                    nn.Linear(config.feature_dim, config.feature_dim),
                )
                continue

            self.meta_embeddings[cat] = nn.Embedding(config.classes_per_cat[cat], config.feature_dim)

        # Linear projection for concatenated embeddings if enabled
        if config.concatenate_embeddings:
            self.concatenated_embeddings_projector = nn.Linear(len(config.img_metadata_cat_cols) * config.feature_dim, config.feature_dim)

        # Post-CNN block
        self.ffn = None
        self.transformer = None
        self.pre_encoder_ffn = None
        self.pre_ffn_rms_norm = None

        # Feedforward block
        if config.use_ffn:
            assert config.ffn_hidden_dim is not None and config.ffn_activation is not None
            self.ffn = FeedForwardBlock(config.feature_dim * self.max_images_per_study, hidden_size=config.ffn_hidden_dim, out_features=config.feature_dim, dropout=config.dropout, activation=config.ffn_activation)

            if config.add_pre_ffn_rms_norm:
                assert config.pre_ffn_rms_norm_dim is not None and isinstance(config.pre_ffn_rms_norm_dim, int)
                self.pre_ffn_rms_norm = RMSNorm(config.pre_ffn_rms_norm_dim, dims_to_apply_to=config.cnn_rms_norms_dims_to_apply)

        # If not, use transformer encoder
        else:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=config.feature_dim, nhead=config.num_attn_heads, dropout=config.dropout, activation='gelu', batch_first=False, norm_first=config.transformer_norm_first)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers, enable_nested_tensor=False)

            if config.add_pre_encoder_ffn:
                assert config.ffn_hidden_dim is not None and config.ffn_activation is not None
                self.pre_encoder_ffn = FeedForwardBlock(config.feature_dim, hidden_size=config.ffn_hidden_dim, out_features=config.feature_dim, dropout=config.dropout, activation=config.ffn_activation)

        # Feedforward block after attention
        self.post_attn_ffn = None
        if config.use_post_attn_ffn:
            assert config.ffn_hidden_dim is not None and config.ffn_activation is not None
            self.post_attn_ffn = FeedForwardBlock(config.feature_dim, hidden_size=config.ffn_hidden_dim, out_features=config.feature_dim, dropout=config.dropout, activation=config.ffn_activation)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feature_dim, 1),
        )

    def forward(self, image_sets, image_mask, imgs_metadata):
        """
        image_sets: [B, S, C, H, W] -> batch of studies with S images
        image_mask: [B, S] with 1 for valid images, 0 for padding
        """
        B, S, C, H, W = image_sets.shape
        x = image_sets.view(B * S, C, H, W)  # [B*S, C, H, W]

        features = self.pipeline(x).view(B * S, -1)  # [B*S, nf*16]
        features = self.projector(features)          # [B*S, feature_dim]
        features = features.view(B, S, -1)           # [B, S, feature_dim]

        # Apply meta embeddings
        if self.config.concatenate_embeddings:
            embeddings_features = []
            for cat_idx, cat in enumerate(self.img_metadata_cat_cols):
                if cat == 'PatientAge':
                    pat_ages = imgs_metadata[:, :, cat_idx].unsqueeze(-1)
                    embeddings_features.append(self.meta_embeddings[cat](pat_ages))
                    continue
                embeddings_features.append(self.meta_embeddings[cat](imgs_metadata[:, :, cat_idx].to(torch.long)))

            # Project concatenated embeddings and add to image features
            features += self.concatenated_embeddings_projector(torch.cat(embeddings_features, dim=-1))

        else:
            for cat_idx, cat in enumerate(self.img_metadata_cat_cols):
                if cat == 'PatientAge':
                    pat_ages = imgs_metadata[:, :, cat_idx].unsqueeze(-1)
                    features += self.meta_embeddings[cat](pat_ages)
                    continue
                features += self.meta_embeddings[cat](imgs_metadata[:, :, cat_idx].to(torch.long))

        if self.ffn is not None:
            if self.pre_ffn_rms_norm is not None:
                features = self.pre_ffn_rms_norm(features)
            features = features.reshape(B, S * self.feature_dim)
            features = self.ffn(features)

        else:
            if self.pre_encoder_ffn is not None:
                features = self.pre_encoder_ffn(features)

            # Transformer expects [S, B, feature_dim]
            features = features.transpose(0, 1)  # [S, B, feature_dim]
            attn_mask = ~image_mask.bool()       # [B, S] -> True = PAD
            output = self.transformer(features, src_key_padding_mask=attn_mask)  # [S, B, feature_dim]
            output = output.transpose(0, 1)  # [B, S, feature_dim]

            # Mean pooling over valid tokens
            masked_output = output * image_mask.unsqueeze(-1)
            features = masked_output.sum(dim=1) / image_mask.sum(dim=1, keepdim=True)

        output = self.classifier(features).squeeze(-1) # [B]

        return output