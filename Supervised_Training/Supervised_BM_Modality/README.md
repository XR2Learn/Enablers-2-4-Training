# Supervised BM Modality

Supervised training for BM modality

# Including component in Docker-compose.yml file as a service

```yaml
ed-training-bm:
    image: some.registry.com/xr2learn-enablers/ed-training-bm:latest
    build:
      context: 'Supervised_Training/Supervised_BM_Modality'
      dockerfile: 'Dockerfile'
    volumes:
      - "./Supervised_Training/Supervised_BM_Modality:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "${CONFIG_FILE_PATH:-./configuration.json}:/app/configuration.json"
    working_dir: /app
    environment:
      - EXPERIMENT_ID=${EXPERIMENT_ID:-development-model}
    command: python supervised_bm_modality/train.py

```
