#!/bin/bash
clear
echo "Vietnamese RVC Tensorboard"

source "./venv/bin/activate"
python "./main/app/run_tensorboard.py" --open
deactivate