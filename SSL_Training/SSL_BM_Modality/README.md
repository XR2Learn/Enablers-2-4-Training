# SSL BM Modality

SSL pre-training for BM Modality

# Including component in Docker-compose.yml file as a service

```yaml
ssl-bm:
    image: some.registry.com/xr2learn-enablers/ssl-bm:latest
    build:
      context: 'SSL_Training/SSL_BM_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./SSL_Training/SSL_BM_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python ssl_bm_modality/pre_train.py

```
