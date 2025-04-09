import copy
import logging
import time
from typing import Optional
import torch
from torch import nn
import pandas as pd
from pathlib import Path
import toml
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from datasets import MammogramDataset
import matplotlib.pyplot as plt
import argparse

from logs import get_logger

logger = get_logger(__name__, log_level=logging.DEBUG, log_to_file=True)


def collate_fn(batch):
    """
    batch: list of tuples (imgs, target, mask)
      imgs: [max_images, C, H, W]
      target: scalar
      mask: [max_images]
    """
    images = torch.stack([item[0] for item in batch], dim=0)      # [B, max_images, C, H, W]
    targets = torch.stack([item[1] for item in batch], dim=0)     # [B]
    masks = torch.stack([item[2] for item in batch], dim=0)       # [B, max_images]
    imgs_metadata = torch.stack([item[3] for item in batch], dim=0)

    return images, targets, masks, imgs_metadata


def train_classification_model(curr_exp, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=5, starting_epoch=0, starting_step=0, starting_loss=0, last_auc: Optional[BinaryAUROC]=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # keep the best weights stored separately
    best_epoch = 0
    best_val_auc = 0
    best_val_loss = float('inf')

    # Each epoch has a training, validation, and test phase
    phases = ['train', 'val']

    # Initialize metrics
    binary_auc = BinaryAUROC().to(device)

    # Create directory to store model checkpoints
    model_checkpoint_path = Path('model', curr_exp)
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Keep track of how loss and accuracy evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase+'_loss'] = []
        training_curves[phase+'_auc'] = []

    try:
        for epoch in range(starting_epoch, num_epochs):
            logger.debug(f'Epoch {epoch+1}/{num_epochs}')
            logger.debug('-' * 10)

            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                since_phase = time.time()

                if epoch == starting_epoch and phase == 'train':
                    step = starting_step
                    running_loss = starting_loss
                    if last_auc is not None:
                        binary_auc = last_auc
                else:
                    step = 0
                    running_loss = 0.0

                for inputs, labels, masks, imgs_metadata in dataloaders[phase]:
                    try:
                        step += 1
                        if step > dataset_sizes[phase]//batch_size:
                            break

                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        masks = masks.to(device)
                        imgs_metadata = imgs_metadata.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs, masks, imgs_metadata)
                            loss = criterion(outputs.view(-1), labels.to(torch.float))

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        probs = torch.sigmoid(outputs.view(-1))
                        binary_auc.update(probs, labels)

                        running_loss += loss.item() * inputs.size(0)
                        logger.debug(f"Step {step} - Loss: {loss.item():.4f}")
                                        
                        if phase == 'train':
                            scheduler.step()
                            time_elapsed = time.time() - since
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': running_loss,
                                'last_step': step,
                                'last_auc': binary_auc,
                                }, model_checkpoint_path / Path(f'checkpoint_last.pth'))
                            
                            if step % 10 == 0:
                                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': running_loss,
                                    'last_step': step,
                                    'last_auc': binary_auc,
                                    }, model_checkpoint_path / Path(f'checkpoint_last_backup.pth'))
                            
                            experiments_config[curr_exp]['best_results'] = {
                                'best_epoch': best_epoch,
                                'best_val_loss': best_val_loss,
                                'best_val_auc': best_val_auc,
                                'time_elapsed_min': time_elapsed // 60,
                                'time_elapsed_sec': time_elapsed % 60,
                                'time_elapsed_per_epoch_min': (time_elapsed / (epoch + 1)) // 60,
                                'time_elapsed_per_epoch_sec': (time_elapsed / (epoch + 1)) % 60,
                            }

                            experiments_config[curr_exp]['training_curves'] = training_curves

                            # Save the experiment data to disk
                            with open(config_path, 'w') as f:
                                toml.dump(experiments_config, f)

                            # Save the best model weights to disk
                            torch.save(best_model_wts, model_checkpoint_path / "best_model.pth")
                    except Exception as e:
                        logger.error(f'''Error in {phase} phase at step {step}''', exc_info=True)
                    finally:
                        continue

                epoch_loss = running_loss / dataset_sizes[phase]
                training_curves[phase+'_loss'].append(epoch_loss)
                phase_time_elapsed = time.time() - since_phase

                logger.debug(f'{phase:5} Loss: {epoch_loss:.4f} Time elapsed: {phase_time_elapsed // 60:.0f}m {phase_time_elapsed % 60:.0f}s')

                epoch_val_auc = binary_auc.compute().item()
                training_curves[phase+'_auc'].append(epoch_val_auc)
                logger.debug(f'{phase:5} AUC: {epoch_val_auc:.4f}')

                binary_auc.reset()

                if phase == 'val' and epoch_val_auc > best_val_auc:
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_val_loss = epoch_loss
                    best_val_auc = epoch_val_auc

                # Save model and traning checkpoints
                if phase == 'train':
                    time_elapsed = time.time() - since
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        }, model_checkpoint_path / Path(f'checkpoint_{str(epoch+1).zfill(2)}.pth'))
                    
                    experiments_config[curr_exp]['best_results'] = {
                        'best_epoch': best_epoch,
                        'best_val_loss': best_val_loss,
                        'best_val_auc': best_val_auc,
                        'time_elapsed_min': time_elapsed // 60,
                        'time_elapsed_sec': time_elapsed % 60,
                        'time_elapsed_per_epoch_min': (time_elapsed / (epoch + 1)) // 60,
                        'time_elapsed_per_epoch_sec': (time_elapsed / (epoch + 1)) % 60,
                    }

                    experiments_config[curr_exp]['training_curves'] = training_curves

                    # Save the experiment data to disk
                    with open(config_path, 'w') as f:
                        toml.dump(experiments_config, f)

                    # Save the best model weights to disk
                    torch.save(best_model_wts, model_checkpoint_path / "best_model.pth")

    except KeyboardInterrupt:
        logger.debug("Training interrupted. Saving best model and results so far...")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

    finally:
        time_elapsed = time.time() - since
        logger.debug(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        logger.debug(f'Best val AUC: {best_val_auc:4f} at epoch {best_epoch+1}')

        experiments_config[curr_exp]['best_results'] = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc,
            'time_elapsed_min': time_elapsed // 60,
            'time_elapsed_sec': time_elapsed % 60,
            'time_elapsed_per_epoch_min': (time_elapsed / (epoch + 1)) // 60,
            'time_elapsed_per_epoch_sec': (time_elapsed / (epoch + 1)) % 60,
        }

        experiments_config[curr_exp]['training_curves'] = training_curves

        # Save the experiment data to disk
        with open(config_path, 'w') as f:
            toml.dump(experiments_config, f)

        # Save the best model weights to disk
        torch.save(best_model_wts, model_checkpoint_path / "best_model.pth")

        # Load the best model weights back into the model
        model.load_state_dict(best_model_wts)

        return model, training_curves


def plot_training_curves(training_curves, show=False):
    """
    Plots loss, accuracy, F1, and AUC from the training_curves dictionary.
    """
    metrics = ['loss', 'auc']
    
    for metric in metrics:
        train_values = training_curves.get(f'train_{metric}', None)
        val_values = training_curves.get(f'val_{metric}', None)
        max_len = min(len(train_values), len(val_values))

        plt.figure()
        if train_values is not None and train_values != []:
            plt.plot(range(1, max_len + 1), train_values[:max_len], label='Train')
        
        if train_values == [] and val_values == []:
            continue  # Skip metrics not tracked

        plt.plot(range(1, max_len + 1), val_values[:max_len], label='Validation')
        plt.title(f'{metric.upper()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot to a file
        filename = Path('model', CURRENT_EXP) / f'training_curves_{metric}.png'
        plt.savefig(filename)
        
        if show:
            plt.show()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a mammogram classification model")
    parser.add_argument("--exp", type=str, help="Name of the experiment", required=True)
    args = parser.parse_args()

    # Load experiment config
    CURRENT_EXP = args.exp
    config_path = Path('ex_config.toml')

    with open(config_path, 'r') as f:
        experiments_config = toml.load(f)

    config = experiments_config[CURRENT_EXP]

    # Load hyperparameters and config data
    batch_size = config['batch_size']
    dataset_size = config['dataset_size']
    dropout = config['dropout']
    feature_dim = config['feature_dim']
    img_size = config['img_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f'Using device: {device}')

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


    # Prepare image metadata
    img_metadata_cat_cols = ['ViewPosition', 'PatientSex', 'ImageLaterality', 'BreastImplantPresent', 'PatientOrientation_0', 'PatientOrientation_1']

    images_metadata_df = pd.read_csv(Path('img_studies_metadata.csv'), index_col=0)
    train_split_df = pd.read_csv(Path('train_split.csv'), index_col=0)

    ## PatientAge
    images_metadata_df['PatientAge'] = images_metadata_df['PatientAge'].str.extract(r'(\d+)').astype(float)
    mean_age = images_metadata_df[images_metadata_df['AccessionNumber'].isin(train_split_df['AccessionNumber'])]['PatientAge'].mean()
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


    # Prepare img transform	
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])

    # Prepare datasets
    train_dataset = MammogramDataset(
            'train_split.csv', 
            img_size=img_size,
            transform=transform, 
            max_images_per_study=max_images_per_study,
            use_weighted_random_sampler=True, 
            add_random_transforms=add_random_transforms,
            randomize_image_order=randomize_image_order,
            images_metadata_df=images_metadata_df,
            pos_weight_scaler=pos_weight_scaler,
            remove_dark_pixels=remove_dark_pixels,
            add_padding_pixels=add_padding_pixels,
            random_transforms_p=random_transforms_prob,
            random_transforms_max=random_transforms_max,
            use_vit_b_16=use_vit_b_16,
            img_metadata_cat_cols=img_metadata_cat_cols,
            dataset_size=dataset_size
        )
    
    val_dataset = MammogramDataset(
            'val_split.csv',
            img_size=img_size,
            transform=transform, 
            max_images_per_study=max_images_per_study,
            use_weighted_random_sampler=True, 
            add_random_transforms=add_random_transforms,
            randomize_image_order=randomize_image_order,
            images_metadata_df=images_metadata_df,
            pos_weight_scaler=pos_weight_scaler,
            remove_dark_pixels=remove_dark_pixels,
            add_padding_pixels=add_padding_pixels,
            random_transforms_p=random_transforms_prob,
            random_transforms_max=random_transforms_max,
            use_vit_b_16=use_vit_b_16,
            img_metadata_cat_cols=img_metadata_cat_cols,
            dataset_size=dataset_size
        )

    # Prepare dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        sampler=train_dataset.sampler
        )

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    dataloaders = {'train': train_dataloader,
                'val': val_dataloader
                }

    dataset_sizes = {'train': len(train_dataset),
                    'val': len(val_dataset)
                    }
    

    # Prepare model
    from model import MammogramScreeningClassifier

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
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_dataset.pos_weight) # For binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch = 0
    start_step = 0
    start_loss = 0
    last_auc = None

    if start_from_checkpoint is not None:
        try:
            logger.debug(f"Loading checkpoint from {start_from_checkpoint}.")
            state_dict = torch.load(start_from_checkpoint, weights_only=False)
            
            # Load model weights
            model.load_state_dict(state_dict['model_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])

            # Set starting epoch
            start_epoch = state_dict['epoch']

            # Set starting step
            start_step = state_dict['last_step']

            # Set starting loss
            start_loss = state_dict['loss']

            # Set last auc
            last_auc = state_dict['last_auc']
            
            logger.debug(f"Resuming training from epoch {start_epoch+1}.")
        except Exception as e:
            logger.debug(f"Error loading checkpoint: {e}")
            logger.debug(f"Started training for a new model.")
            start_epoch = 0
            start_step = 0
            start_loss = 0
            last_auc = None
        finally:
            pass

    # Define learning rate scheduler
    schedulers = {
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_exponentiallr_gamma),
        'CyclicLR': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=lr_cycliclr_max_lr, step_size_up=len(train_dataset)//batch_size, mode='exp_range'),
    }

    scheduler = schedulers[learning_rate_scheduler]

    # Train the model. We also will store the results of training to visualize
    model, training_curves = train_classification_model(CURRENT_EXP, model, dataloaders, dataset_sizes, 
                                        criterion, optimizer, scheduler, num_epochs=num_epochs, starting_epoch=start_epoch, starting_step=start_step, starting_loss=start_loss, last_auc=last_auc)
    plot_training_curves(training_curves)