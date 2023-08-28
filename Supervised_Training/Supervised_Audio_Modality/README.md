# Supervised Emotion Detection Training - Audio Modality

Component to train an ED model using audio data input type. 

`Input`: CSV file with labels + path to features file (.npy files) + timestamp. 
- CSV Input file should be in the folder `/datasets/<input_CSV_file.CSV>`

`Output`: File with trained weights (Not implemented).
- Output file with trained weights is saved in the folder `/outputs/<file_weights>`

## Development Environment Setup


## Development Notes
- Dataset: RAVDESS (audio dataset annotated with seven emotion classes).
- Encoder to generate features: wav2vec2
- Features stored as: .npy files (from tensor to ndarray to file)
- Dataset .CSV file with labels, path to features and timestamps. 