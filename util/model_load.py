import os,sys, torch
from iGAN import AEGANSynthesizer
from clbn import CLBNSynthesizer
from ctgan import CTGANSynthesizer
from independent import IndependentSynthesizer
from medgan  import MedganSynthesizer
from privbn import PrivBNSynthesizer
from tablegan import TableganSynthesizer
from tvae import TVAESynthesizer
from uniform import UniformSynthesizer
from veegan import VEEGANSynthesizer

synthesizer = {
    "CLBNSynthesizer" : CLBNSynthesizer,
    "CTGANSynthesizer" : CTGANSynthesizer,
    "IndependentSynthesizer" :IndependentSynthesizer,
    "MedganSynthesizer" :MedganSynthesizer,
    "PrivBNSynthesizer" :PrivBNSynthesizer,
    "TableganSynthesizer" :TableganSynthesizer,
    "TVAESynthesizer" :TVAESynthesizer,
    "UniformSynthesizer" :UniformSynthesizer,
    "VEEGANSynthesizer" :VEEGANSynthesizer,
    "AEGANSynthesizer" : AEGANSynthesizer,
}
def model_load(location, GPU_NUM, choosed_model):
    """
    location: pth file location
    GPU_NUM: gpu number to use
    choosed_model: epoch
    """
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu') 
    checkpoint = torch.load(location, map_location = device)
    argument = checkpoint["arg"]
    if "GPU_NUM" in argument:
        argument["GPU_NUM"] = GPU_NUM
    try:
        del argument["excute_time"]
    except:
        pass

    argument["save_arg"] = {}
    
    
    if "name" not in checkpoint: # this exists for previous version
        if "discriminator" in checkpoint["info"][choosed_model]:
            syn = CTGANSynthesizer(**argument, train=False)
            syn.model_load(checkpoint, choosed_model)
        elif 'generator' in checkpoint["info"][choosed_model]:
            argument["sigma_coef"] = 0

            syn = AEGANSynthesizer(**argument, train=False)
            syn.model_load(checkpoint, choosed_model)
        else:
            syn = TVAESynthesizer(**argument, train=False)
            syn.model_load(checkpoint, choosed_model)
    else:
        syn = synthesizer[checkpoint["name"]](**argument, train=False)
        syn.model_load(checkpoint, choosed_model)
    return syn        
