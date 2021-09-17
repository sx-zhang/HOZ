from .a3c_train import a3c_train
from .a3c_val import a3c_val

trainers = [ 
    'vanilla_train',
    'learned_train',
]

testers = [
    'vanilla_val',
    'learned_val',
]

variables = locals()