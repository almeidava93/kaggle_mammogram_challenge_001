import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm 
import torch
import pandas as pd
from pathlib import Path
import toml
from torchvision import transforms
from torch.utils.data import DataLoader
from config import MammogramClassifierConfig
from datasets import MammogramDataset
import argparse
from model import MammogramScreeningClassifier
from datasets import MammogramDataset
from train import collate_fn
from logs import get_logger

logger = get_logger(__name__, log_level=logging.DEBUG, log_to_file=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a mammogram classification model")
    parser.add_argument("--exp", type=str, help="Experiment name", required=True)
    parser.add_argument("--path", type=str, help="Model path", required=False, default=None)
    parser.add_argument("--batch-size", type=int, help="Batch size", required=False, default=None)
    parser.add_argument("--workers", type=int, help="Number of workers", required=False, default=0)
    parser.add_argument("--pin-memory", type=bool, help="Activate memory pinning", required=False, default=False)
    parser.add_argument("--device", type=str, help="Device", required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load experiment config
    CURRENT_EXP = args.exp
    config_path = Path('ex_config.toml')

    with open(config_path, 'r') as f:
        experiments_config = toml.load(f)

    config = MammogramClassifierConfig(exp=CURRENT_EXP, **experiments_config[CURRENT_EXP])
    config.workers = args.workers
    config.pin_memory = args.pin_memory

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Load dataset
    test_dataset = MammogramDataset(split='test', config=config)

    # Prepare dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        )

    # Load model
    model = MammogramScreeningClassifier(config).to(config.device)

    if args.path is not None:
        print(f"Loading checkpoint from {args.path}")
        model_weights = torch.load(args.path, map_location=config.device, weights_only=False)['model_state_dict']

    else:
        print(f"Loading checkpoint from model/{CURRENT_EXP}/best_model.pth")
        model_weights = torch.load(Path(f'model/{CURRENT_EXP}/best_model.pth'), map_location=config.device)

    # Load model weights
    model.load_state_dict(model_weights)
    model.eval()

    # Make predictions
    submission_data = []
    with torch.no_grad():
        for inputs, studies_ids, masks, imgs_metadata in tqdm(test_dataloader, total=len(test_dataset)//config.batch_size):
            inputs = inputs.to(config.device)
            studies_ids = studies_ids.to(config.device)
            masks = masks.to(config.device)
            imgs_metadata = imgs_metadata.to(config.device)
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