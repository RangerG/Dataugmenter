import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time
import os

# Using GAN generator, acccording to the dataset
def gan(x, y, args):
    cat_var = 'ord'  # 'ord' (Ordinal [no onehot] or 'cat' Categorical [onehot])
    dataset_name = args.dataset
    generator_name = 'generator_' + dataset_name  # Ordinal version (generic)
    generator_path = "generators"
    generator_full_path = os.path.join(generator_path, generator_name)

    # Loading the model
    with tf.device('/cpu:0'):
        generator = tf.keras.models.load_model(generator_full_path, compile=False)
        print("[INFO] Generator loaded!")
        time_steps = x.shape[1]  # Retrieving time steps

        latent_dim = generator._build_input_shape.as_list()[-1] - 1  # Conditional GAN!

        # Retrieving latent space dimension
        if cat_var == 'ord':  # Ordinal version (no onehot)
            latent_dim = generator._build_input_shape.as_list()[-1] - 1  # Conditional GAN!
        elif cat_var == 'cat':  # Categorical version (onehot)
            latent_dim = generator._build_input_shape.as_list()[-1]  # Conditional GAN!
        print("[INFO] Latent space dimension = {}".format(latent_dim))

        # Generating data
        if cat_var == 'ord':  # Ordinal version (no onehot)
            #print("Shape of y:", y.shape)
            #y_ordinal = y + 1
            y_ordinal = tf.argmax(y, axis=1) + 1
            cond_noise = np.zeros(shape=(len(x), time_steps, latent_dim + 1))
            cond_noise[:, :, 0] = tf.expand_dims(y_ordinal, -1).numpy()
            noise = tf.random.normal(shape=(len(x), time_steps, latent_dim))
            cond_noise[:, :, 1:] = noise
        elif cat_var == 'cat':  # Categorical version (onehot)
            n_classes = y.shape[1]
            cond_noise = np.zeros(shape=(len(x), time_steps, latent_dim))
            cond_noise[:, :, :n_classes] = np.expand_dims(y, 1)
            noise = tf.random.normal(shape=(len(x), time_steps, latent_dim - n_classes))
            cond_noise[:, :, n_classes:] = noise

        # Convert to numpy
        gen_data = generator(cond_noise).numpy()

    return gen_data