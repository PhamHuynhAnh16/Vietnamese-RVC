#!/bin/bash
clear
echo "Vietnamese RVC (Linux)"

source "./venv/bin/activate"
python "./main/app/app.py" --open
deactivate
