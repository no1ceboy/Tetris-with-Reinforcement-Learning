from tetris import Tetris
from dqn import ActorCritic
from gae_lambda import GAE

import numpy as np
import tensorflow as tf    

# --------PARAMETER-----------
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
ROLL_STEPS = 512    
EPOCHS = 10
MINI_BATCH = 64
TOTAL_UPDATES = 5000

#----------------------------
dqn = ActorCritic()
actor, opt_pi = dqn.build_actor()
critic, opt_v = dqn.build_critic()
env = Tetris()
buffer = GAE(size=ROLL_STEPS)
obs = env.reset()
update = 0
train_writer = tf.summary.create_file_writer("/kaggle/workingogs/train")
eval_writer = tf.summary.create_file_writer("/kaggle/workinglogs/eval")

# ---------------------------
def evaluate_policy(env, actor, n_games=50):
    scores, lines, pieces = [], [], []
    for _ in range(n_games):
        obs = env.reset(); done=False; total=0; piece_cnt=0
        while not done:
            cand = env.get_next_states()
            feats = tf.convert_to_tensor(np.vstack(list(cand.values())), tf.float32)
            logits = actor(feats)[:,0]
            (x,rot) = list(cand.keys())[int(tf.argmax(logits))]
            reward, done, obs = env.play(x, rot)
            total += reward; piece_cnt += 1
        scores.append(env.get_game_score())   
        lines.append(obs[0])                       
        pieces.append(piece_cnt)
    return np.mean(scores), np.mean(lines), np.mean(pieces)

# -----------Rollout phase-----------
while update < TOTAL_UPDATES:
    # Each episode
    for _ in range(ROLL_STEPS):
        # Sample next state
        cand_dict = env.get_next_states()
        a = list(cand_dict.keys())
        feats = np.vstack(list(cand_dict.values()))  
        logits = actor(feats).numpy()[:,0]
        dist = tf.nn.softmax(logits).numpy()
        assert np.isfinite(dist).all(), "NaNs in softmax"
        assert np.abs(dist.sum() - 1) < 1e-6, "not a prob-vector"
        idx = np.random.choice(len(a), p=dist)
        logp = np.log(dist[idx] + 1e-8)
        v_s = critic(obs[None]).numpy()[0, 0]
        r, done, next_obs = env.play(a[idx][0], a[idx][1])

        # Store
        buffer.store(obs, idx, logp, r, v_s, done, feats.astype(np.float32))
        if done:
            obs = env.reset()
        else:
            obs = next_obs

        # Sample obs from buffer
        # if _ % 50 == 0:
        #     print("step", _, "obs", obs)
        
    # Calculate the last value of episode
    last_v = 0.0 if done else critic(obs[None]).numpy()[0,0]
    buffer.finish(last_v)
    print(np.unique(buffer.s[:10], axis=0).shape)
    print("buffer rewards sample:", buffer.r[:20])
    print("fraction done=True:", buffer.done.mean())

# -----------Learning phase--------------
    for _ in range(EPOCHS):
        for (s, a_idx, logp, adv, r, cand_list) in buffer.get(MINI_BATCH):
            adv = tf.convert_to_tensor(adv)
            r = tf.convert_to_tensor(r)
            logp = tf.convert_to_tensor(logp)
            
            # Flatten all actions list and keep track of each
            flat, splits = [], []
            cum = 0
            entropies, v_losses = [], []
            for c in cand_list:
                flat.append(c)
                cum += len(c)
                splits.append(cum)
            flat = tf.convert_to_tensor(np.vstack(flat))
            sizes = [splits[0]]
            for i in range(len(splits)-1):
                sizes.append(splits[i+1] - splits[i])
            
            with tf.GradientTape() as tape_pi, tf.GradientTape() as tape_v:
                # Compute log prob of actor
                logits = actor(flat)[:,0]
                split_logits = tf.split(logits, sizes)

                logp_n = tf.stack([tf.nn.log_softmax(l)[i] for l,i in zip(split_logits,a_idx)])

                # Compute ratio, entropy
                ratio = tf.exp(logp_n - logp)
                minrat = tf.clip_by_value(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pi_loss= -tf.reduce_mean(tf.minimum(ratio*adv, minrat*adv))
                entropy= tf.reduce_mean(
                    tf.concat([-tf.nn.softmax(l)*tf.nn.log_softmax(l) for l in split_logits], 0))
                
                # Compute value loss
                v_pred = critic(s)
                v_loss = tf.reduce_mean((r - v_pred)**2)
                loss = pi_loss - ENT_COEF*entropy + VF_COEF*v_loss

            grads_pi = tape_pi.gradient(loss, actor.trainable_variables)
            grads_v = tape_v.gradient(v_loss, critic.trainable_variables)
            opt_pi.apply_gradients(zip(grads_pi, actor.trainable_variables))
            opt_v.apply_gradients(zip(grads_v, critic.trainable_variables))

            entropies.append(entropy)
            v_losses.append(v_loss)

    with train_writer.as_default():
        tf.summary.scalar("return", np.mean(buffer.r), step=update)
        tf.summary.scalar("entropy", np.mean(entropies), step=update)
        tf.summary.scalar("value_loss", np.mean(v_losses), step=update)
    
    update += 1
    
    # Evaluate after 10 update
    if update % 10 == 0:
        env_eval = Tetris()
        scores, lines, pieces = evaluate_policy(env_eval, actor, n_games=30)
        with eval_writer.as_default():
            tf.summary.scalar("score", scores, step=update)
            tf.summary.scalar("lines", lines, step=update)
            tf.summary.scalar("pieces", pieces, step=update)
        print(f'Update {update}: Avg reward {np.mean(buffer.r):.1f}')

    buffer.clear()
        
# Save model
actor.save_model("/kaggle/workingactor.keras")
critic.save_model("/kaggle/workingcritic.keras")