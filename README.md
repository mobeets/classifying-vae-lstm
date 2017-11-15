
This is the implementation in Keras of the Classifying VAE and Classifying VAE+LSTM as described in _A Classifying Variational Autoencoder withApplication to Polyphonic Music Generation_ by Jay A. Hennig, Akash Umakantha, and Ryan C. Williamson.

These models extend the standard VAE and VAE+LSTM to the case where there is a latent discrete category. In the case of music generation, for example, we may wish to infer the key of a song, so that we can generate notes that are consistent with that key. These discrete latents are modeled as a Logistic Normal distribution, so that random samples from this distribution can use the reparameterization trick during training.

Training data for the JSB Chorales and Piano-midi corpuses can be found in `data/input`. Songs have been transposed into C major or C minor (`*_Cs.pickle`), for comparison to previous work, or kept in their original keys (`*_all.pickle`).

Samples from the models for the JSB Chorales and Piano-midi corpuses, for songs in their original keys, can be found in `data/samples`.

 <audio controls="controls">
   <source src="data/samples/PMall_Data.wav" type="audio/mpeg">
   Your browser does not support the audio element.
 </audio>
 
