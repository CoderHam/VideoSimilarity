"""
This file performs similarity inference using 3D CNN.
"""
import os
import numpy as np


"""
python main.py --input input --video_root videos --output output.json
--model resnet-34-kinetics.pth --mode feature --sample_duration 8
"""

def extract_features():
    os.system('cd')
    os.system('cd 3d-cnn')
    os.system('python main.py --input input --video_root videos --output output.json --model resnet-34-kinetics.pth --mode feature --sample_duration 8')

extract_features()
