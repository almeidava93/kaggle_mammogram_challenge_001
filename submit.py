from pathlib import Path
import pandas as pd
from tqdm import tqdm 
import torch
import pandas as pd
from pathlib import Path
import toml
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import MammogramDataset
import argparse
from model import MammogramScreeningClassifier


parser = argparse.ArgumentParser(description="Train a mammogram classification model")
parser.add_argument("--exp", type=str, help="Experiment name", required=True)
parser.add_argument("--path", type=bool, help="Model path", required=False, default=None)
parser.add_argument("--batch-size", type=int, help="Batch size", required=False, default=None)
args = parser.parse_args()

CURRENT_EXP = args.exp

# Load experiment config
config_path = Path('ex_config.toml')

with open(config_path, 'r') as f:
    experiments_config = toml.load(f)
config = experiments_config[CURRENT_EXP]

# Load data splits
test_split_path = Path('test_split.csv')
train_split_path = Path('train_split.csv') 
images_metadata_path = Path('img_studies_metadata.csv')

train_df = pd.read_csv(train_split_path)
test_df = pd.read_csv(test_split_path)
images_metadata_df = pd.read_csv(images_metadata_path, index_col=0)

# Prepare metadata df
img_metadata_cat_cols = ['ViewPosition', 'PatientSex', 'ImageLaterality', 'BreastImplantPresent', 'PatientOrientation_0', 'PatientOrientation_1']

## PatientAge
images_metadata_df['PatientAge'] = images_metadata_df['PatientAge'].str.extract(r'(\d+)').astype(float)
mean_age = images_metadata_df[images_metadata_df['AccessionNumber'].isin(train_df['AccessionNumber'])]['PatientAge'].mean()
images_metadata_df['PatientAge'] = images_metadata_df['PatientAge'].fillna(mean_age)/120

## PatientOrientation
images_metadata_df['PatientOrientation_0'] = images_metadata_df['PatientOrientation'].apply(lambda x: eval(x)[0])
images_metadata_df['PatientOrientation_1'] = images_metadata_df['PatientOrientation'].apply(lambda x: eval(x)[1])

## Transform to categorical
n_categories = {}
for col in img_metadata_cat_cols:
    n_categories[col] = images_metadata_df[col].nunique()
    images_metadata_df[col] = images_metadata_df[col].astype('category').cat.codes

# Get max images per study
max_images_per_study = images_metadata_df.groupby(by=['AccessionNumber'])['PatientID'].count().max().item()

# Load hyperparameters and config data
batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
dataset_size = config['dataset_size']
dropout = config['dropout']
feature_dim = config['feature_dim']
img_size = config['img_size']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = config['learning_rate']
learning_rate_scheduler = config.get('learning_rate_scheduler', 'ExponentialLR')
lr_exponentiallr_gamma = config.get('lr_exponentiallr_gamma', 0.95)
lr_cycliclr_max_lr = config.get('lr_cycliclr_max_lr', learning_rate*100)

num_epochs = config['num_epochs']
weight_decay = config['weight_decay']

use_ffn = config.get('use_ffn', None)
ffn_hidden_dim = config.get('ffn_hidden_dim', None)
ffn_activation = config.get('ffn_activation', None)

add_pre_ffn_rms_norm = config.get('add_pre_ffn_rms_norm', False)
pre_ffn_rms_norm_dim = config.get('pre_ffn_rms_norm_dim', None)

add_linear_proj_to_embeddings = config.get('add_linear_proj_to_embeddings', False)

# Tune the weight of positive examples
pos_weight_scaler = config.get('pos_weight_scaler', 1.0)

# Image transformations
remove_dark_pixels = config.get('remove_dark_pixels', False)
add_padding_pixels = config.get('add_padding_pixels', False)

# Random transforms to augment data during training
add_random_transforms = config.get('add_random_transforms', None)
random_transforms_prob = config.get('random_transforms_prob', None)
random_transforms_max = config.get('random_transforms_max', None)
randomize_image_order = config.get('randomize_image_order', None)

# Customizations to transformer encoder
num_heads = config.get('num_attn_heads', None)
num_layers = config.get('num_encoder_layers', None)
add_pre_encoder_ffn = config.get('add_pre_encoder_ffn', None)
use_post_attn_ffn = config.get('use_post_attn_ffn', None)
transformer_norm_first = config.get('transformer_norm_first', False)

# Customizations to CNN encoder
nc = config['num_img_channels'] # number of channels
nf = config['num_img_init_features'] # number of features to begin with
cnn_dropout = config.get('cnn_dropout', 0) # to add or not dropout after each block
cnn_activation = config.get('cnn_activation', 'ReLU')
cnn_resnet_n_conv = config.get('cnn_resnet_n_conv', 2)
cnn_use_rms_norm = config.get('cnn_use_rms_norm', False)
cnn_rms_norms_dims_to_apply = config.get('cnn_rms_norms_dims_to_apply', [-1])

# Start from checkpoint
start_from_checkpoint = config.get('start_from_checkpoint', None)

# Use pretrained models
use_vit_b_16 = config.get('use_vit_b_16', False)
freeze_pretrained_weights = config.get('freeze_pretrained_weights', False)

from datasets import MammogramDataset
from train import collate_fn

# Prepare img transform	
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
])

test_dataset = MammogramDataset(
    'test_split.csv', 
    dataset_size=-1,
    img_size=img_size,
    transform=transform, 
    max_images_per_study=max_images_per_study,
    use_weighted_random_sampler=False, 
    use_vit_b_16=use_vit_b_16,
    images_metadata_df=images_metadata_df,
    img_metadata_cat_cols=img_metadata_cat_cols,
)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Load model
model = MammogramScreeningClassifier(
        img_size=img_size,
        img_metadata_cat_cols=img_metadata_cat_cols,
        nc=nc, 
        nf=nf, 
        dropout=dropout, 
        feature_dim=feature_dim, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        max_images_per_study=max_images_per_study, 
        use_ffn=use_ffn, 
        ffn_hidden_size=ffn_hidden_dim, 
        ffn_activation=ffn_activation, 
        add_pre_encoder_ffn=add_pre_encoder_ffn, 
        add_pre_ffn_rms_norm=add_pre_ffn_rms_norm, 
        pre_ffn_rms_norm_dim=pre_ffn_rms_norm_dim, 
        add_linear_proj_to_embeddings=add_linear_proj_to_embeddings, cnn_rms_norms_dims_to_apply=cnn_rms_norms_dims_to_apply, 
        use_vit_b_16=use_vit_b_16,
        freeze_pretrained_weights=freeze_pretrained_weights,
        n_categories=n_categories,  
        use_post_attn_ffn=use_post_attn_ffn,
        cnn_activation=cnn_activation,
        cnn_dropout=cnn_dropout,
        transformer_norm_first=transformer_norm_first
    ).to(device)

if args.path is not None:
    print(f"Loading checkpoint from {args.path}")
    model_weights = torch.load(args.path, map_location=device)['model_state_dict']

else:
    print(f"Loading checkpoint from model/{CURRENT_EXP}/best_model.pth")
    model_weights = torch.load(Path(f'model/{CURRENT_EXP}/best_model.pth'), map_location=device)

# Load model weights
model.load_state_dict(model_weights)
model.eval()

from tqdm import tqdm 

submission_data = []
with torch.no_grad():
    for inputs, studies_ids, masks, imgs_metadata in tqdm(test_dataloader, total=len(test_dataset)//batch_size):
        inputs = inputs.to(device)
        studies_ids = studies_ids.to(device)
        masks = masks.to(device)
        imgs_metadata = imgs_metadata.to(device)
        outputs = model(inputs, masks, imgs_metadata)
        probs = torch.sigmoid(outputs.view(-1))
        submission_data += [{'AccessionNumber': idx, 'target': p} for idx, p in zip(studies_ids.tolist(), probs.tolist())]

submission_df = pd.DataFrame.from_records(submission_data)
submission_df['AccessionNumber'] = submission_df['AccessionNumber'].astype(str).str.zfill(6)

# Save submission
# Check if submissions for the current experiment exist
if Path(f'submissions/submission_{CURRENT_EXP}_001.csv').exists():
    n_submissions = len(list(Path('submissions').glob(f'submission_{CURRENT_EXP}_*')))
    n_submissions = str(n_submissions + 1).zfill(3)
    submission_df.to_csv(Path(f'submissions/submission_{CURRENT_EXP}_{n_submissions}.csv'), index=False)
else:
    submission_df.to_csv(Path(f'submissions/submission_{CURRENT_EXP}_001.csv'), index=False)