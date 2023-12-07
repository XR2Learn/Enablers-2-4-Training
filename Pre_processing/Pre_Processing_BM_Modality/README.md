# Pre Processing BM Modality
Pre-processing for BioMeasurements Modality

# Including component in Docker-compose.yml file as a service 

```yaml
ssl-bm:
    image: some.registry.com/xr2learn-enablers/ssl-bm:latest
    build:
      context: '<<component>>/<<project_folder>>'
      dockerfile: 'Dockerfile'
    volumes:
      - "./<<component>>/<<project_folder>>:/app"
      - "./datasets:/app/datasets"
      - "./outputs:/app/outputs"
      - "./configuration.json:/app/configuration.json"
    working_dir: /app
    environment:
      # To include environment variables in the format below
      - KEY=${KEY}
    #    entrypoint: /bin/sh -c
    command: python <<project_slug>>/<<main_python_file>>.py
```