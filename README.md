# World Models 

* two components: 
    1. world model (vision model CNN + memory RNN) 
    2. controller model 

* vision model is a VAE, compress image to small latent vector z  


## Dataset Generation 

You can use the `generate_vae_data.py` script to generate a dataset for training the VAE model. Before you do this, make sure you have enough space on your disk (~260 GB).


## Todos 

* create a training script for the VAE 
* add image transforms for the VAE training 
* create the controller model
* create the memory model 

