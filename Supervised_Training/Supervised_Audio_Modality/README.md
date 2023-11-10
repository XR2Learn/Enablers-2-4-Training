# Supervised Emotion Detection Training - Audio Modality

Component to train an ED model using audio data input type. 

`Input`: CSV file with labels + path to features file (.npy files) + timestamp. 
- CSV Input file should be in the folder `/datasets/<input_CSV_file.CSV>`

`Output`: File with trained weights (Not implemented).
- Output file with trained weights is saved in the folder `/outputs/<file_weights>`

## Development Environment Setup
Create and activate virtual environment with `venv`. If you wish to use GPU in your environment, install requirements from `requirements-gpu.txt` (line 3):
``` 
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements{-gpu}.txt
```

## Development Notes
- Dataset: RAVDESS (audio dataset annotated with seven emotion classes).
- Encoder to generate features: wav2vec2
- Features stored as: .npy files (from tensor to ndarray to file)
- Dataset .CSV file with labels, path to features and timestamps. 

## Tests
To launch tests:
```
sh run_tests.sh
```