# World Models 

* two components: 
    1. world model (vision model + memory RNN) 
    2. controller model 

* vision model is a VAE, compress image to small latent vector z  


## Dataset Generation 

You can use the `generate_vae_data.py` script to generate a dataset for training the VAE model. Before you do this, make sure you have enough space on your disk (~260 GB).

## Training 

1. After you have generated your training dataset, you can use it to train your Variational Autoencoder Model. 

2. Training the RNN using the VAE you trained

3. Train the controller model


## Todos 

* add image transforms for the VAE training 
* currently the test dataloader loads to many files? or I iterate through it the wrong way
