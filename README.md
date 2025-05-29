# Tetris with Reinforcement Learning

This repository contains an implementation of a Reinforcement Learning (RL) agent trained to play **Tetris**.  

---

## Features

- **Environment:** Custom Tetris game implementation compatible with RL training loops.
- **RL Algorithm:** Deep Q Leanring (DQL), Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE).
- **State Representations:**    
  - Full board state (optional CNN-based implementation)
- **Reward shaping:** Penalizes holes, bumpiness, and height to encourage efficient play.
- **Training & Evaluation:** Includes training loop with rollout buffering, PPO updates, and periodic evaluation.
- **Visualization:** Game rendering during evaluation for visual verification of learned policy.
- **Logging:** TensorBoard integration for monitoring training metrics such as return, entropy, and value loss.
