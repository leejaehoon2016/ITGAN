import os,sys, torch
from train_itgan import AEGANSynthesizer
from train_base1_clbn import CLBNSynthesizer
from train_base2_independent import IndependentSynthesizer
from train_base3_privbn import PrivBNSynthesizer
from train_base4_uniform import UniformSynthesizer
from train_base5_tvae import TVAESynthesizer
from train_base6_ctgan import CTGANSynthesizer
from train_base7_tablegan import TableganSynthesizer
from train_base8_veegan import VEEGANSynthesizer
from train_base9_medgan  import MedganSynthesizer






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
