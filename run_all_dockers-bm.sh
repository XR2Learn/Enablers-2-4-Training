#!/bin/bash

# Deleting All dataset, output files
#sudo rm -R datasets/*

sudo rm -R outputs/XRoom/shimmer/*

echo "--------------------"
echo "Pre-processing-bm"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm pre-processing-bm
echo "--------------------"
echo "SSL-training-bm"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm ssl-bm
echo "--------------------"
echo "SSL-features-extraction-bm"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm ssl-features-generation-bm
echo "--------------------"
echo "Supervised-training-bm"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm ed-training-bm
echo "--------------------"
