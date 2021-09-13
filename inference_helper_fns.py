import numpy as np
import json
from typing import Dict

def generate_camera_matrix(fx:float, fy:float, cx:float, cy:float) -> np.ndarray:
    return np.array(( ( fx, 0, cx ), ( 0, fy, cy ), ( 0, 0, 1 ) ))

def load_object_index_map(fp:str) -> Dict[int,str]:
    with open(fp, 'r') as f:
        oim_ = { int(k):v for k,v in json.load(f).items() }

    return oim_
