import tensorflow as tf
from tensorflow.keras import layers
from keras.regularizers import l2

class ActorCritic():        
    def build_actor(self):
        model = tf.keras.Sequential(
            [layers.Input((20, 10, 1)),
            layers.Conv2D(16, kernel_size=(3,3), padding="same", activation='relu', kernel_regularizer=l2(0.001)),
            layers.Conv2D(32, kernel_size=(3,3), padding="same", activation='relu', kernel_regularizer=l2(0.001)),
            layers.Flatten(),
            layers.Dense(16,  activation='relu'),
            layers.Dense(1,  activation=None)],
            name = "Actor"
        )
        opt = tf.keras.optimizers.Adam(0.0003)
        return model, opt
    
    def build_critic(self):
        model = tf.keras.Sequential(
            [layers.Input((20, 10, 1)),
            layers.Conv2D(16, kernel_size=(3,3), padding="same", activation='relu', kernel_regularizer=l2(0.001)),
            layers.Conv2D(32, kernel_size=(3,3), padding="same", activation='relu', kernel_regularizer=l2(0.001)),
            layers.Flatten(),
            layers.Dense(16,  activation='relu'),
            layers.Dense(1,  activation=None)],
            name = "Critic"
        )
        opt = tf.keras.optimizers.Adam(0.0003)
        return model, opt

