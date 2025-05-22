import tensorflow as tf
from tensorflow.keras import layers

class ActorCritic():        
    def build_actor(self):
        model = tf.keras.Sequential(
            [layers.Input((3,)),
            layers.Dense(64, activation='tanh'),
            layers.Dense(64, activation='tanh'),
            layers.Dense(1,  activation=None)],
            name = "Actor"
        )
        opt = tf.keras.optimizers.Adam(0.003)
        return model, opt
    
    def build_critic(self):
        model = tf.keras.Sequential(
            [layers.Input((3,)),
            layers.Dense(64, activation='tanh'),
            layers.Dense(64, activation='tanh'),
            layers.Dense(1,  activation=None)],
            name = "Critic"
        )
        opt = tf.keras.optimizers.Adam(0.003)
        return model, opt

