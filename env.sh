#!/bin/bash

if [ ! -x "$0" ]; then
    echo "Creating executable "$0" file..."
    chmod +x "$0"
    echo "Finished creating executable "$0" file..."
    source "$0"
fi 

if [ "$0" = "$BASH_SOURCE" ]; then
    echo "You need to source this script, please run:
    source env"
else
    source opt_venv/bin/activate
    echo "Activated virtual environment!"
fi

