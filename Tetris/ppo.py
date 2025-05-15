from tetris import Tetris
from dqn import ActorCritic
from gae_lambda import GAE

import numpy as np
import tensorflow as tf
from collections import deque     

# --------PARAMETER-----------
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
ROLL_STEPS = 2048    
EPOCHS = 10
MINI_BATCH = 256
TOTAL_UPDATES = 5000

#----------------------------
dqn = ActorCritic()
actor, opt_pi = dqn.build_actor()
critic, opt_v = dqn.build_critic()
env = Tetris()
buffer = GAE()
obs = env.reset()
update = 0

# -----------Rollout phase-----------
while update < TOTAL_UPDATES:
    # Each episode
    for _ in range(ROLL_STEPS):
        # Sample next state
        obs = env._get_board_props()
        cand_dict = env.get_next_states()
        a = cand_dict.keys()
        feats = np.vstack(list(cand_dict.values()))  
        logits = actor(feats).numpy()[:,0]
        dist = tf.nn.softmax(logits).numpy()
        idx = np.random.choice(len(a), p=dist)
        logp = np.log(dist[idx] + 1e-8)
        v_s = critic(obs[None]).numpy()[0, 0]
        r, done = env.play(a[idx][0], a[idx][1])

        # Store
        buffer.store(obs, idx, logp, r, v_s, done, feats.astype(np.float32))
        if done:
            obs = env.reset()

    # Calculate the last value of episode
    last_v = 0.0 if done else critic(obs[None]).numpy()[0,0]
    buffer.finish(last_v)

# -----------Learning phase--------------
    for _ in range(EPOCHS):
        for (s, a_idx, logp, adv, r, cand_list) in buffer.get(MINI_BATCH):
            # Flatten all actions list and keep track of each
            flat, splits = [], []
            cum = 0
            for c in cand_list:
                flat.append(c)
                cum += len(c)
                splits.append(cum)
            flat = tf.convert_to_tensor(np.vstack(flat))
            logits = actor(flat)[:,0]
            split_logits = tf.split(logits, splits[:-1])
            logp_n = tf.stack([tf.nn.log_softmax(l)[i] for l,i in zip(split_logits,a_idx)])

            # Compute clip and entropy
            ratio = tf.exp(logp_n - logp)
            minrat = tf.clip_by_value(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pi_loss= -tf.reduce_mean(tf.minimum(ratio*adv, minrat*adv))
            entropy= tf.reduce_mean(
                tf.concat([ -tf.nn.softmax(l)*tf.nn.log_softmax(l) for l in split_logits ],0))

            with tf.GradientTape() as tape_pi, tf.GradientTape() as tape_v:
                tape_pi.watch(actor.trainable_variables)
                tape_v.watch(critic.trainable_variables)
                # Compute losses
                v_pred = critic(s).tf.squeeze()
                v_loss = tf.reduce_mean((r - v_pred)**2)
                loss   = pi_loss - ENT_COEF*entropy + VF_COEF*v_loss

            grads_pi = tape_pi.gradient(loss, actor.trainable_variables)
            grads_v = tape_v.gradient(v_loss, critic.trainable_variables)
            opt_pi.apply_gradients(zip(grads_pi, actor.trainable_variables))
            opt_v.apply_gradients(zip(grads_v, critic.trainable_variables))
    
    buffer.clear(); update += 1
    if update % 10 == 0:
        print(f'update {update}: avg reward {np.mean(buffer.r):.1f}')
            
