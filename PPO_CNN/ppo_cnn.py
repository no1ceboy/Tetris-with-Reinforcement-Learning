from tetris_cnn import Tetris
from dqn_cnn import ActorCritic
from gae_lambda_cnn import GAE

import os
import numpy as np
import tensorflow as tf    
import time
import matplotlib.pyplot as plt

# -------- Parameter -----------
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
ROLL_STEPS = 512    
EPOCHS = 10
MINI_BATCH = 64
TOTAL_UPDATES = 2000

# ------------- Metric lists -------------
train_ret_list = []
train_entropy = []
train_vloss = []

eval_score_list = []
eval_lines_list = []
eval_pieces_list = []
eval_line_types = {1: [], 2: [], 3: [], 4: []} 

#----------------------------
dqn = ActorCritic()
actor, opt_pi = dqn.build_actor()
critic, opt_v = dqn.build_critic()
env = Tetris()
buffer = GAE(size=ROLL_STEPS)
obs = env.reset()
update = 0
train_writer = tf.summary.create_file_writer("PPO_CNN/train")
eval_writer = tf.summary.create_file_writer("PPO_CNN/eval")

# ---------------------------

# Evaluate after 10 update, track mean score, lines cleared, pieces placed and number of 1, 2, 3, 4 line cleared consecutively
def evaluate_policy(env, actor, n_games=20):
    scores, lines, pieces = [], [], []
    line_type_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for _ in range(n_games):
        obs = env.reset(); done=False; total=0; piece_cnt=0; line=0

        while not done:
            cand = env.get_next_states()
            feats = tf.convert_to_tensor(np.vstack(list(cand.values())), tf.float32)
            logits = actor(feats)[:,0]
            (x,rot) = list(cand.keys())[int(tf.argmax(logits))]
            reward, done, line_cleared, obs = env.play(x, rot)
            total += reward; piece_cnt += 1; line += line_cleared
            if 1 <= line_cleared <= 4:
                line_type_counts[line_cleared] += 1

        scores.append(env.get_game_score())   
        lines.append(line)                       
        pieces.append(piece_cnt)
    return np.mean(scores), np.mean(lines), np.mean(pieces), line_type_counts

# -----------Rollout phase-----------
while update < TOTAL_UPDATES:
    # Each episode
    for _ in range(ROLL_STEPS):
        if env.game_over:
            obs = env.reset()
        # Sample next state
        cand_dict = env.get_next_states()
        a = list(cand_dict.keys())
        feats = np.stack(list(cand_dict.values()))  
        feats = feats[..., None].astype(np.float32) 
        logits = actor(feats).numpy()[:,0]
        dist = tf.nn.softmax(logits).numpy()
        assert np.isfinite(dist).all(), "NaNs in softmax"
        assert np.abs(dist.sum() - 1) < 1e-6, "not a prob-vector"
        idx = np.random.choice(len(a), p=dist)
        logp = np.log(dist[idx] + 1e-8)
        v_s = critic(obs[None]).numpy()[0, 0]
        r, done, line_cleared, next_obs = env.play(a[idx][0], a[idx][1])

        # Store
        buffer.store(obs[..., None], idx, r, v_s, logp, done, feats.astype(np.float32))
        if done:
            obs = env.reset()
        else:
            obs = next_obs
        
    # Calculate the last value of episode
    last_v = 0.0 if done else critic(obs[None]).numpy()[0,0]
    buffer.finish(last_v)
    train_ret_list.append(np.mean(buffer.r[:buffer.ptr]))
    print(np.unique(buffer.s[:10], axis=0).shape)
    print("buffer rewards sample:", buffer.r[:20])
    print("fraction done=True:", buffer.done.mean())

# -----------Learning phase--------------
    for _ in range(EPOCHS):
        entropies, v_losses = [], []
        for (s, a_idx, logp, adv, r, cand_list) in buffer.get(MINI_BATCH):
            adv = tf.convert_to_tensor(adv)
            r = tf.convert_to_tensor(r)
            logp = tf.convert_to_tensor(logp)
            
            # Flatten all actions list and keep track of each
            flat, splits = [], []
            cum = 0
            for c in cand_list:
                flat.append(c)
                cum += len(c)
                splits.append(cum)
            flat  = np.concatenate([c[..., None] for c in cand_list], axis=0)
            flat  = tf.convert_to_tensor(flat, dtype=tf.float32)
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

    # Write in train log
    with train_writer.as_default():
        tf.summary.scalar("return", np.mean(buffer.r), step=update)
        tf.summary.scalar("entropy", np.mean(entropies), step=update)
        tf.summary.scalar("value_loss", np.mean(v_losses), step=update)
    
    train_entropy.append(np.mean(entropies))
    train_vloss.append(np.mean(v_losses))

    update += 1
    
    # Evaluate after 50 update
    if update % 50 == 0:
        env_eval = Tetris()
        scores, lines, pieces, line_types = evaluate_policy(env_eval, actor, n_games=10)
        eval_score_list.append(scores)
        eval_lines_list.append(lines)
        eval_pieces_list.append(pieces)
        for i in range(1,5):
            eval_line_types[i].append(line_types.get(i, 0))
        with eval_writer.as_default():
            tf.summary.scalar("score", scores, step=update)
            tf.summary.scalar("lines", lines, step=update)
            tf.summary.scalar("pieces", pieces, step=update)
        print(f'Update {update}: Avg reward {np.mean(buffer.r):.1f}')

    buffer.clear()
        
# ------------ Plot metric -------------
def save_plots(train_ret, train_ent, train_vloss, eval_score, eval_lines, eval_pieces, eval_line_types, eval_every=10, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    def save_curve(x, y, name, title, ylab):
        plt.figure(figsize=(16,10))
        plt.title(title)
        plt.plot(x, y)
        plt.xlabel("update"); plt.ylabel(ylab)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
        plt.close()

    x_tr = np.arange(len(train_ret))
    save_curve(x_tr, train_ret, "train_return", "Train return", "R")
    save_curve(x_tr, train_ent, "train_entropy", "Policy entropy","entropy")
    save_curve(x_tr, train_vloss,"train_vloss", "Value loss", "MSE")

    x_ev = np.arange(len(eval_score)) * eval_every
    save_curve(x_ev, eval_score, "eval_score", "Eval score", "score")
    save_curve(x_ev, eval_lines, "eval_lines", "Eval lines", "lines")
    save_curve(x_ev, eval_pieces, "eval_pieces", "Eval pieces", "pieces")

    plt.figure(figsize=(16,10))
    plt.title("Line-type counts")
    for k, lab in zip(range(1,5), ["1-line","2-line","3-line","4-line"]):
        plt.plot(x_ev, eval_line_types[k], label=lab)
    plt.xlabel("update"); plt.ylabel("count"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_line_types.png"), dpi=150)
    plt.close()

save_plots(train_ret_list, train_entropy, train_vloss, eval_score_list, eval_lines_list, eval_pieces_list, eval_line_types)

# Save model
actor.save("actor_cnn.keras")
critic.save("critic_cnn.keras")