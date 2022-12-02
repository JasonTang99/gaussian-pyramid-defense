import torch

def get_ens_adv_model(model_name, model_path, adv_models, args):
    checkpoint = torch.load(adv_model_paths[i])
    if 'state_dict' in checkpoint.keys():
        state = 'state_dict'
    elif 'net' in checkpoint.keys():
        state = 'net'
    adv_models[i].load_state_dict(checkpoint[state])