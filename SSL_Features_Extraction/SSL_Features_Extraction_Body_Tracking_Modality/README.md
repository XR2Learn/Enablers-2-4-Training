# SSL Features Extraction Body Tracking Modality

SSL features extraction component for the body tracking modality.

# Including component in Docker-compose.yml file as a service

```yaml
ssl-features-generation-body-tracking:
    image: some.registry.com/xr2learn-enablers/ssl-features-generation-body-tracking:latest
    build:
      context: 'SSL_Features_Extraction/SSL_Features_Extraction_Body_Tracking_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./SSL_Features_Extraction/SSL_Features_Extraction_Body_Tracking_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python ssl_features_extraction_body_tracking_modality/generate_features.py

```
