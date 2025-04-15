import copy
import logging
import time
import torch
from torch import nn
from pathlib import Path
import toml
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from config import MammogramClassifierConfig
from datasets import MammogramDataset
import matplotlib.pyplot as plt
import argparse
from model import MammogramScreeningClassifier
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


def train_classification_model(curr_exp, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                               config: MammogramClassifierConfig):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # keep the best weights stored separately
    best_epoch = 0
    best_val_auc = 0
    best_val_loss = float('inf')

    if config.best_results is not None:
        best_epoch = config.best_results['best_epoch']
        best_val_auc = config.best_results['best_val_auc']
        best_val_loss = config.best_results['best_val_loss']
        since = time.time() - config.best_results['time_elapsed_min'] * 60 - config.best_results['time_elapsed_sec']

    # Each epoch has a training, validation, and test phase
    phases = ['train', 'val']

    # Initialize metrics
    binary_auc = BinaryAUROC().to(config.device)

    # Create directory to store model checkpoints
    model_checkpoint_path = Path('model', curr_exp)
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Keep track of how loss and accuracy evolves during training
    training_curves = {}
    for phase in phases:
        training_curves[phase+'_loss'] = []
        training_curves[phase+'_auc'] = []

    # Load previous training curves if they exist
    if config.training_curves is not None:
        training_curves = config.training_curves

    try:
        for epoch in range(config.start_epoch, config.num_epochs):
            logger.debug(f'Epoch {epoch+1}/{config.num_epochs}')
            logger.debug('-' * 10)

            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                since_phase = time.time()

                if epoch == config.start_epoch and phase == 'train':
                    step = config.start_step
                    running_loss = config.start_loss
                    if config.last_auc is not None:
                        binary_auc = config.last_auc
                else:
                    step = 0
                    running_loss = 0.0

                if step > dataset_sizes[phase]//config.batch_size:
                    logger.debug(f"End of {phase} phase")
                    continue

                for inputs, labels, masks, imgs_metadata in dataloaders[phase]:
                    if step > dataset_sizes[phase]//config.batch_size:
                        logger.debug(f"End of {phase} phase")
                        break

                    try:
                        step += 1
                        inputs = inputs.to(config.device)
                        labels = labels.to(config.device)
                        masks = masks.to(config.device)
                        imgs_metadata = imgs_metadata.to(config.device)

                        # Zero the parameter gradients
                        # See https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
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

                            training_state = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': running_loss,
                                'last_step': step,
                                'last_auc': binary_auc,
                                }
                            
                            # Save checkpoints if enabled
                            if config.save_every_step:
                                torch.save(training_state, model_checkpoint_path / Path(f'checkpoint_last.pth'))
                            
                            if config.save_after_n_steps is not None and step % config.save_after_n_steps == 0:
                                if step % 10 == 0:
                                    torch.save(training_state, model_checkpoint_path / Path(f'checkpoint_last_backup.pth'))

                        
                    except Exception as e:
                        logger.error(f'''Error in {phase} phase at step {step}''', exc_info=True)
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
                    if config.save_every_epoch or (
                        config.save_after_n_epochs is not None and epoch % config.save_after_n_epochs == 0
                    ):
                        time_elapsed = time.time() - since
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'last_step': step,
                            'last_auc': binary_auc,
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
    parser.add_argument("--workers", type=int, help="Number of workers", required=False, default=0)
    parser.add_argument("--pin-memory", type=bool, help="Activate memory pinning", required=False, default=False)
    args = parser.parse_args()

    # Load experiment config
    CURRENT_EXP = args.exp
    config_path = Path('ex_config.toml')

    with open(config_path, 'r') as f:
        experiments_config = toml.load(f)

    config = MammogramClassifierConfig(exp=CURRENT_EXP, **experiments_config[CURRENT_EXP])
    config.workers = args.workers
    config.pin_memory = args.pin_memory
    logger.debug(f'Using device: {config.device}')

    # Prepare datasets
    train_dataset = MammogramDataset(split='train', config=config)
    val_dataset = MammogramDataset(split='val', config=config)

    # Prepare dataloaders
    train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            collate_fn=collate_fn, 
            sampler=train_dataset.sampler,
            num_workers=config.workers,
            pin_memory=config.pin_memory,
        )

    val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=config.workers,
            pin_memory=config.pin_memory,
        )

    dataloaders = {'train': train_dataloader,
                'val': val_dataloader
                }

    dataset_sizes = {'train': len(train_dataset),
                    'val': len(val_dataset)
                    }
    
    # Prepare model
    model = MammogramScreeningClassifier(config).to(config.device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_dataset.pos_weight) # For binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.eps)

    if config.start_from_checkpoint is not None:
        try:
            logger.debug(f"Loading checkpoint from {config.start_from_checkpoint}.")
            state_dict = torch.load(config.start_from_checkpoint, weights_only=False)
            
            # Load model weights
            model.load_state_dict(state_dict['model_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])

            # Set starting epoch
            config.start_epoch = state_dict['epoch']

            # Set starting step
            config.start_step = state_dict['last_step']

            # Set starting loss
            config.start_loss = state_dict['loss']

            # Set last auc
            config.last_auc = state_dict['last_auc']
            
            logger.debug(f"Resuming training at epoch {config.start_epoch+1}.")
        except Exception as e:
            logger.debug(f"Error loading checkpoint: {e}")
            logger.debug(f"Started training for a new model.")
            config.best_results = None
            config.training_curves = None
        finally:
            pass

    # Define learning rate scheduler
    schedulers = {
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_exponentiallr_gamma),
        'CyclicLR': torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.learning_rate, max_lr=config.lr_cycliclr_max_lr, step_size_up=len(train_dataset)//config.batch_size, mode='exp_range'),
    }

    scheduler = schedulers[config.learning_rate_scheduler]

    # Train the model. We also will store the results of training to visualize
    model, training_curves = train_classification_model(
            CURRENT_EXP, 
            model, 
            dataloaders, 
            dataset_sizes,
            criterion, 
            optimizer, 
            scheduler, 
            config,
        )
    plot_training_curves(training_curves)