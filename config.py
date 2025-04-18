from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field
import torch

class MammogramClassifierConfig(BaseModel):
    exp: str = Field(..., description="Experiment name")
    batch_size: int = Field(20, description="Batch size")
    dataset_size: int = Field(-1, description="Dataset size")
    feature_dim: int = Field(256, description="Feature dimension")
    img_size: int = Field(256, description="Image size")
    device: str = Field('cuda' if torch.cuda.is_available() else 'cpu', description="Device to use for training")
    dropout: float = Field(0.1, description="Dropout rate")
    learning_rate: float = Field(0.0001, description="Learning rate")
    learning_rate_scheduler: str = Field('ExponentialLR', description="Learning rate scheduler")
    lr_exponentiallr_gamma: float = Field(0.95, description="Gamma for ExponentialLR")
    lr_cycliclr_max_lr: Optional[float] = Field(0.001, description="Max learning rate for CyclicLR")
    num_epochs: int = Field(10, description="Number of epochs")
    steps_per_epoch: Optional[int] = Field(None, description="Number of steps per epoch")
    workers: int = Field(0, description="Number of workers for data loading")
    pin_memory: bool = Field(False, description="Pin memory for data loading")
    weight_decay: float = Field(0.0, description="Weight decay for optimizer")
    eps: float = Field(1e-8, description="Epsilon to avoid division by zero in optimizer and other computations")
    max_images_per_study: Optional[int] = Field(None, description="Max number of images per study")
    cache_data: bool = Field(False, description="Cache data in memory")
    use_cache: bool = Field(False, description="Use cached data")

    # Feedforward parameters
    use_ffn: bool = Field(False, description="Use feedforward network")
    ffn_hidden_dim: Optional[int] = Field(None, description="Hidden dimension for feedforward network")
    ffn_activation: str = Field('ReLU', description="Activation function for feedforward network")
    add_pre_ffn_rms_norm: bool = Field(False, description="Add RMS norm before feedforward network")
    pre_ffn_rms_norm_dim: Optional[int] = Field(None, description="Dimension for RMS norm before feedforward network")

    # Metadata embeddings
    add_linear_proj_to_embeddings: bool = Field(False, description="Add linear projection to embeddings")
    concatenate_embeddings: bool = Field(False, description="Concatenate embeddings and project them instead of just adding them up")

    # Loss function parameters
    pos_weight_scaler: float = Field(1.0, description="Scaler for positive weight in loss function")
    
    # Sampling parameters
    sampler_pos_prob: Optional[float] = Field(None, description="Probability of sampling positive samples")
    use_weighted_random_sampler: bool = Field(True, description="Use weighted random sampler for imbalanced datasets in the training set")

    # Image transformations
    transform: Optional[Any] = Field(None, description="Image transformations applied to all images and data splits")
    remove_dark_pixels: bool = Field(True, description="Remove dark pixels from images")
    add_padding_pixels: bool = Field(False, description="Center and add padding pixels to images until img_size")
    add_random_transforms: bool = Field(False, description="Add random transforms to augment data during training")
    random_transforms_prob: Optional[float] = Field(None, description="Probability of applying random transforms")
    random_transforms_max: Optional[int] = Field(None, description="Max number of random transforms to apply per image")
    randomize_image_order: bool = Field(False, description="Randomize image order in the batch during training")
    no_preprocessing: bool = Field(False, description="Do not apply any preprocessing to images. Only resize is applied.")
    invert_background: bool = Field(True, description="Invert images with background different from value 0 to attempt to uniformize them.")

    # Transformer encoder parameters
    num_attn_heads: Optional[int] = Field(8, description="Number of attention heads")
    num_encoder_layers: Optional[int] = Field(2, description="Number of encoder layers")
    add_pre_encoder_ffn: bool = Field(False, description="Add feedforward network before transformer encoder")
    use_post_attn_ffn: bool = Field(False, description="Use feedforward network after transformer encoder")
    transformer_norm_first: bool = Field(False, description="Apply normalization before attention in transformer encoder")

    # CNN encoder parameters
    num_img_channels: int = Field(1, description="Number of image channels for CNN")
    num_img_init_features: int = Field(64, description="Number of initial features for CNN")
    cnn_dropout: float = Field(0.1, description="Dropout rate for CNN")
    cnn_activation: str = Field('ReLU', description="Activation function for CNN")
    cnn_resnet_n_conv: int = Field(2, description="Number of convolutional layers in a single ResNet block")
    cnn_resnet_n_blocks: int = Field(5, description="Number of ResNet blocks")
    cnn_use_rms_norm: bool = Field(False, description="Use RMS norm in CNN instead of BatchNorm")
    cnn_rms_norms_dims_to_apply: list[int] = Field([-1], description="Dimensions to apply RMS norm in CNN")

    # Start from checkpoint
    start_from_checkpoint: Optional[str] = Field(None, description="Path to checkpoint to start from")

    # Use pretrained models
    use_vit_b_16: bool = Field(False, description="Use ViT-B/16 pretrained model")
    freeze_pretrained_weights: bool = Field(False, description="Freeze pretrained weights")
    load_pretrained_weights: bool = Field(False, description="Load pretrained weights")

    # Relevant paths
    images_metadata_path: Path = Field(Path('img_studies_metadata.csv'), description="Path to images metadata CSV file")
    train_split_path: Optional[Path] = Field(Path('train_split.csv'), description="Path to training split CSV file")
    val_split_path: Optional[Path] = Field(Path('val_split.csv'), description="Path to validation split CSV file")
    test_split_path: Optional[Path] = Field(Path('test_split.csv'), description="Path to test split CSV file")

    # Image metadata
    img_metadata_cat_cols: list[str] = ['ViewPosition', 'PatientSex', 'ImageLaterality', 'BreastImplantPresent', 'PatientOrientation_0', 'PatientOrientation_1', 'FilterMaterial', 'img_width', 'img_height', 'RescaleType', 'PixelIntensityRelationshipSign', 'PixelIntensityRelationship', 'QualityControlImage', 'FieldOfViewRotation', 'FocalSpots', 'Grid', 'Manufacturer', 'ExposureControlMode', 'DetectorConditionsNominalFlag', 'AnodeTargetMaterial']
    classes_per_cat: Optional[dict[str, int]] = Field(None, description="Number of classes per categorical feature")
    img_metadata_num_cols: list[str] = ['KVP', 'BodyPartThickness', 'CompressionForce', 'RelativeXRayExposure', 'Exposure', 'PositionerPrimaryAngle', 'DistanceSourceToPatient', 'DistanceSourceToDetector', 'DetectorTemperature']

    # Previous training data for the current experiment
    training_curves: Optional[dict[str, list[float]]] = Field(None, description="Training curves for the current experiment")
    best_results: Optional[dict[str, float]] = Field(None, description="Best results for the current experiment")
    start_epoch: Optional[int] = Field(0, description="Epoch to start from")
    start_step: Optional[int] = Field(0, description="Step to start from")
    start_loss: Optional[float] = Field(0.0, description="Loss to start from")
    last_auc: Optional[Any] = Field(None, description="Last BinaryAUROC object")

    # Configure checkpoint saving behavior
    save_every_step: bool = Field(False, description="Save model checkpoint after every step")
    save_after_n_steps: Optional[int] = Field(None, description="Save model checkpoint after n steps")
    save_every_epoch: bool = Field(True, description="Save model checkpoint after every epoch")
    save_after_n_epochs: Optional[int] = Field(None, description="Save model checkpoint after n epochs")