#!/usr/bin/env bash

set -e
echo "Vietnamese RVC By Anh"

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

python main/app/app.py --open