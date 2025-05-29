import time, numpy as np, tensorflow as tf
from tetris import Tetris   # or from tetris import Tetris

actor = tf.keras.models.load_model("PPO/model/actor_modelCNN.keras", compile=False)
env   = Tetris()

obs   = env.reset() 
done  = False
scores, lines, pieces = [], [], []
line_type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
for i in range(50):
    obs   = env.reset(); done=False; total=0; piece_cnt=0; line=0
    while not done:
        cand = env.get_next_states()
        feats = np.stack(list(cand.values()))  
        feats = feats[..., None].astype(np.float32) 
        logits = actor(feats)[:,0]
        (x,rot) = list(cand.keys())[int(tf.argmax(logits))]
        reward, done, line_cleared, obs = env.play(x, rot)
        total += reward; piece_cnt += 1; line += line_cleared
        if 1 <= line_cleared <= 4:
            line_type_counts[line_cleared] += 1

    scores.append(env.get_game_score())   
    lines.append(line)                       
    pieces.append(piece_cnt)

print("score:", np.mean(scores))
print("line:", np.mean(lines))
print("piece:", np.mean(pieces))
print(line_type_counts)