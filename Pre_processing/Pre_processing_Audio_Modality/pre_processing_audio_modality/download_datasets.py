import requests
import io
import os
import zipfile
from tqdm import tqdm


def download_RAVDESS(full_dataset_path):
    zip_file_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

    r = requests.get(zip_file_url, stream=True)
    progress_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='B', unit_scale=True,
                        desc='Download progress: RAVDESS dataset')
    dat = b''.join(x for x in r.iter_content(chunk_size=16384) if progress_bar.update(len(x)) or True)

    z = zipfile.ZipFile(io.BytesIO(dat))
    z.extractall(full_dataset_path)
