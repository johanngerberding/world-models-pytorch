import torch 

def get_n_params(model: torch.nn.Module) -> int: 
    """Get the number of parameters of model""" 
    pp = 0 
    for p in list(model.parameters()):
        nn = 1 
        for s in list(p.size()):
            nn = nn * s 
        pp += nn 
    
    return pp