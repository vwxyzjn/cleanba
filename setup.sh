#!/bin/bash

# Check if the argument is passed and it's one of the accepted values
if [ $# -eq 0 ]; then
    echo "No arguments provided, please provide 'backend' argument"
    exit 1
elif [ "$1" != "cpu" -a "$1" != "gpu" -a "$1" != "tpu" ]; then
    echo "Invalid argument provided, please provide either 'cpu', 'gpu' or 'tpu'"
    exit 1
fi

backend=$1

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Use if-elif-else to choose the right JAX version
if [ "$backend" == "cpu" ]; then
    pip install --upgrade jax
elif [ "$backend" == "gpu" ]; then
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [ "$backend" == "tpu" ]; then
    pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi