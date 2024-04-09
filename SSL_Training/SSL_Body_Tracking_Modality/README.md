# SSL Body Tracking Modality

SSL Tranining component for Body Tracking Modality.

# Including component in Docker-compose.yml file as a service

```yaml
ssl-body-tracking:
    image: some.registry.com/xr2learn-enablers/ssl-body-tracking:latest
    build:
      context: 'SSL_Training/SSL_Body_Tracking_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./SSL_Training/SSL_Body_Tracking_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python ssl_body_tracking_modality/pre_train.py

```
