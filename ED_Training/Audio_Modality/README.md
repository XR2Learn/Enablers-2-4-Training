# Emotion Detection Training - Audio Modality

Project to train an ED model using audio data input type. 

## Development Notes

- Dataset: RAVDESS (audio dataset annotated with seven emotion classes).
- Encoder to generate features: wav2vec2
- Features stored as: .npy files (from tensor to ndarray to file)
- Dataset .CSV file with labels, path to features and timestamps. 