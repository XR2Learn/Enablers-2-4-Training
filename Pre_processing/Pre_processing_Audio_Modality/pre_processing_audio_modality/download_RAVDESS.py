import requests, zipfile, io
zip_file_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
r = requests.get(zip_file_url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("Pre_processing\Pre_processing_Audio_Modality\datasets\RAVDESS")