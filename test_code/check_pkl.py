import jax
import pickle as pkl
import warnings
#file_path = 'checkpoint_resnetv1-34_41.pkl'
file_path = '/home/amigo/ChaoW/ws/hil-serl/examples/experiments/grab_earphone/demo_data/grab_earphone_10_demos_2025-06-03_15-08-05.pkl'

def print_keys(d, prefix=''):
    for key in d:
        full_key = f"{prefix}.{key}" if prefix else key
        print(full_key)
        if isinstance(d[key], dict):  # �~K��~@�为�~W�~E��~L�~@~R�~R�~D�~P~F
            print_keys(d[key], full_key)

nested_dict = {'user': {'name': 'Alice', 'address': {'city': 'Beijing'}}}
print_keys(nested_dict)
with open(file_path, "rb") as f:
    print('use params in path:', file_path)
    encoder_params = pkl.load(f)
    print_keys(encoder_params)


