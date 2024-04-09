# Handcrafted Features Extraction Body Tracking Modality

Handcrafted Features Extraction component for the Body Tracking Modality from Magic XRoom data.

# Including component in Docker-compose.yml file as a service

```yaml
handcrafted-features-generation-body-tracking:
    image: some.registry.com/xr2learn-enablers/handcrafted-features-generation-body-tracking:latest
    build:
      context: 'Handcrafted_Features_Extraction/Handcrafted_Features_Extraction_Body_Tracking_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./Handcrafted_Features_Extraction/Handcrafted_Features_Extraction_Body_Tracking_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python handcrafted_features_extraction_body_tracking_modality/generate_features.py

```
