import kagglehub
from pathlib import Path
import pandas as pd

datasets = [
    "felipekitamura/spr-mmg-1",
    "felipekitamura/spr-mmg-2",
    "felipekitamura/spr-mmg-02",
    "felipekitamura/spr-mmg-3",
    "felipekitamura/spr-mmg-4",
    "felipekitamura/spr-mmg-5",
    "felipekitamura/spr-mmg-6",
    "felipekitamura/spr-mmg-7",
    "felipekitamura/spr-mmg-8",
    "felipekitamura/spr-mmg-9",
]

dataset_paths = []

# Download latest version
for dataset in datasets:
    path = kagglehub.dataset_download(dataset)
    dataset_paths.append(path)
    print(f"Path to {dataset} files:", path)

path_list = []
for path in dataset_paths:
    path_list += list(Path(path).glob("*"))

path_dict = {k.parts[-1]:k for k in path_list}

# Add file path to train data csv and save to disk
train_df = pd.read_csv('train.csv')
train_df['AccessionNumber'] = train_df['AccessionNumber'].astype(str).str.zfill(6)
train_df['path'] = train_df['AccessionNumber'].map(path_dict)
train_df.to_csv('train.csv', index=False)