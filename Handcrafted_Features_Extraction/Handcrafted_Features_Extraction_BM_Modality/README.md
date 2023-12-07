# Handcrafted Features Extraction BM Modality
Handcrafted features for BM Modality

# Including component in Docker-compose.yml file as a service 

```yaml
handcrafted-features-generation-bm:
    image: some.registry.com/xr2learn-enablers/handcrafted-features-generation-bm:latest
    build:
      context: '<<component>>/Handcrafted_Features_Extraction_BM_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./<<component>>/Handcrafted_Features_Extraction_BM_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python handcrafted_features_extraction_bm_modality/generate_features.py

```

