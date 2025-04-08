from pathlib import Path
import time
from typing import Optional
import pandas as pd
import pydicom
from tqdm import tqdm 

test_split_path = Path('test_split.csv')
train_split_path = Path('train.csv') # Include train and val split

train_df = pd.read_csv(train_split_path)
test_df = pd.read_csv(test_split_path)

dicom_file_paths = []
img_studies_metadata = []
img_with_errors = []

for assession_data in tqdm(train_df.to_dict('records') + test_df.to_dict('records'), desc="Collecting each image study path"):
    curr_study_metadata = {}
    curr_dicom_file_paths = list(Path(assession_data['path']).glob('*.dcm'))
    dicom_file_paths += curr_dicom_file_paths

def get_study_metadata(dicom_img_path: str, ttl_hash=None) -> tuple[dict, Optional[dict|None]]:
    del ttl_hash
    try:
        study_metadata = {}
        error = None
        dicom_img = pydicom.dcmread(dicom_img_path,force=True)
        study_metadata['ViewPosition'] = dicom_img.ViewPosition
        study_metadata['PatientOrientation'] = dicom_img.PatientOrientation
        study_metadata['PatientSex'] = dicom_img.PatientSex
        study_metadata['SeriesDescription'] = dicom_img.SeriesDescription
        study_metadata['PatientID'] = dicom_img.PatientID
        study_metadata['AccessionNumber'] = dicom_img.AccessionNumber
        study_metadata['ImageLaterality'] = dicom_img.ImageLaterality
        study_metadata['BreastImplantPresent'] = dicom_img.BreastImplantPresent
        study_metadata['PatientBirthDate'] = dicom_img.PatientBirthDate
        study_metadata['PatientAge'] = dicom_img.PatientAge
        study_metadata['img_height'] = dicom_img.pixel_array.shape[0]
        study_metadata['img_width'] = dicom_img.pixel_array.shape[1]

        study_metadata['path'] = filepath
        
        img_studies_metadata.append(study_metadata)
    except Exception as e:
        error = {'path': filepath, 'error': str(e)}
    finally:
        return study_metadata, error


# Get image study metadata
for filepath in tqdm(dicom_file_paths, desc='Collecting each image study metadata'):
    study_metadata, error = get_study_metadata(filepath)
    if error is not None:
        img_with_errors.append(error)
    else:
        img_studies_metadata.append(study_metadata)

# Save to disk
img_studies_metadata_df = pd.DataFrame.from_records(img_studies_metadata)
img_studies_metadata_df.to_csv(Path(f'img_studies_metadata.csv'))
img_with_errors_df = pd.DataFrame.from_records(img_with_errors)
img_with_errors_df.to_csv(Path(f'img_with_errors.csv'))
print('Image study metadata saved to img_studies_metadata.csv')
print('Image with errors saved to img_with_errors.csv')