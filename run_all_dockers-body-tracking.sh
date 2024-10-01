#!/bin/bash

# Deleting All dataset, output files
#sudo rm -R datasets/*

sudo rm -R outputs/XRoom/body_tracking/*

echo "--------------------"
echo "Pre-processing-body-tracking"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm pre-processing-body-tracking
echo "--------------------"
# echo "Handcrafted-features-generation-body-tracking"
# echo "--------------------"
# CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm handcrafted-features-generation-body-tracking
# echo "--------------------"
#echo "SSL-training-body-tracking"
#echo "--------------------"
#CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm ssl-body-tracking
#echo "--------------------"
#echo "SSL-features-extraction-body-tracking"
#echo "--------------------"
#CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm ssl-features-generation-body-tracking
#echo "--------------------"
echo "Supervised-training-body-tracking"
echo "--------------------"
CONFIG_FILE_PATH=$CONFIG_FILE_PATH docker compose run --rm ed-training-body-tracking
echo "--------------------"
