# SSL Features Extraction BM Modality

SSL Features Extraction for BM Modality

# Including component in Docker-compose.yml file as a service

```yaml
ssl-features-generation-bm:
    image: some.registry.com/xr2learn-enablers/ssl-features-generation-bm:latest
    build:
      context: 'SSL_Features_Extraction/SSL_Features_Extraction_BM_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./SSL_Features_Extraction/SSL_Features_Extraction_BM_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python ssl_features_extraction_bm_modality/generate_features.py

```
