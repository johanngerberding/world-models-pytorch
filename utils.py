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

def collate_fn(batch): 
    obss, actions, rewards, terminals, next_obss = [], [], [], [], []
    for sample in batch: 
        obs, action, reward, terminal, next_obs = sample 
        obss.append(torch.tensor(obs))
        actions.append(torch.tensor(action))
        rewards.append(torch.tensor(reward))
        terminals.append(torch.tensor(terminal))
        next_obss.append(torch.tensor(next_obs))

    # cut all to min seq len in batch
    if len(set([ob.shape[0] for ob in obss])) != 1: 
        _seq_len = min([ob.shape[0] for ob in obss]) 
        obss = [ob[:_seq_len, :, :, :] for ob in obss]
        actions = [act[:_seq_len] for act in actions] 
        rewards = [rew[:_seq_len] for rew in rewards] 
        terminals = [ter[:_seq_len] for ter in terminals] 
        next_obss = [nob[:_seq_len, :, :, :] for nob in next_obss] 

    obss = torch.stack(obss)
    actions = torch.stack(actions) 
    rewards = torch.stack(rewards)
    terminals = torch.stack(terminals)
    next_obss = torch.stack(next_obss)
   
    return obss, actions, rewards, terminals, next_obss

def transform_to_latent(obs, model, device, rnn_cfg) -> torch.Tensor:
    seq_len = obs.shape[1] 
    obs = obs / 255. 
    obs = [torch.nn.functional.upsample(
        frame.view(-1, 3, 96, 96), 
        size=64, 
        mode='bilinear', 
        align_corners=True,
    ) for frame in obs]

    mus, sigmas = [], []
    for frame in obs: 
        _, mu, sigma = model(frame.to(device))
        mus.append(mu) 
        sigmas.append(sigma)
    
    latents = [] 
    for mu, sigma in zip(mus, sigmas):
        latent = mu + sigma.exp() * torch.randn_like(mu)
        latents.append(latent) 
    
    latents = torch.stack(latents)
    latents = latents.view(rnn_cfg['batch_size'], seq_len, rnn_cfg['latent_size']) 

    return latents 