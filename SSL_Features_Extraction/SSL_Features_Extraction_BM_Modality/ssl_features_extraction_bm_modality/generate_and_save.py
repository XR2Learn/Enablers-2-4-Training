import os
from tqdm import tqdm
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule


def generate_and_save(
        encoder: Union[torch.nn.Module, LightningModule],
        data_path: str,
        outputs_folder: str,
        input_type: str,
        output_type: str,
        transforms: Dict[str, Any]
):
    """
    generate_and_save : given the encoder, extract the features and save to .npy files

    Args:
        encoder: the pytorch encoder model to extract features from
        data_path: can take two forms:
            - csv containing the paths to the files for which features have to be extracted and saved
            - path to a folder with .npy files
        outputs_folder: outputs path with all components outputs
        input_type: type of inputs (should match folder in outputs_folder)
        output_type: type of outputs (generated features will be saved into outputs_folder/output_type)
        transforms: transforms applied to data
    Returns:
        none
    """
    if data_path.endswith(".csv") and not os.path.isdir(data_path):
        files = pd.read_csv(os.path.join(outputs_folder, data_path))['files']
        filename = os.path.basename(data_path)
        cur_transforms = transforms[filename.split(".")[0]]
    elif os.path.isdir(data_path):
        files = os.listdir(os.path.join(data_path))
        cur_transforms = transforms["train"]
    else:
        raise ValueError("Incorrect data_path format")

    for npy_path in tqdm(files):
        x = np.load(
            os.path.join(outputs_folder, input_type, npy_path).replace('\\', '/')
        )
        if len(x.shape) <= 1:
            x = np.expand_dims(x, axis=-1)
        x_tensor = cur_transforms(x)
        features = encoder(x_tensor).squeeze()
        os.makedirs(os.path.join(outputs_folder, output_type), exist_ok=True)
        np.save(os.path.join(outputs_folder, output_type, npy_path.split(os.path.sep)[-1]), features.detach().numpy())
