#!/usr/bin/env bash

# This script is used to run the interactive phishing URL detection tool.
# It activates the virtual environment and runs the detection script.

# Activate the virtual environment
source .venv/bin/activate

# Run the detection script in interactive mode
# detect.py will now handle user input for URLs and model selection if arguments are passed
python detect.py $*
