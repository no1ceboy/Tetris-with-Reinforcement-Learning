# A buffer that implement GAE lambda bootstrapping
import numpy as np

np.random.seed(32)

class GAE():
    def __init__(self, size = 2048, GAE_LAMBDA = 0.95, GAMMA = 0.99):
        # s: state, a: index of chosen action, r: reward, v: critic's state value, logp: log prob of action, d: episode done, cand: all possible actions, ptr: current pointer  
        self.s = np.zeros((size,3),  np.float32)
        self.a_idx = np.zeros(size, np.int32)
        self.logp = np.zeros(size, np.float32)
        self.r = np.zeros(size, np.float32)
        self.v = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.bool_)
        self.cands = []              
        self.ptr = 0   
        self.GAMMA = GAMMA
        self.GAE_LAMBDA = GAE_LAMBDA               

    # Store information into the buffer
    def store(self, s, a, r, v, logp, d, cand):
        i = self.ptr
        self.s[i], self.a_idx[i], self.logp[i], self.r[i], self.v[i], self.done[i] = s, a, logp, r, v, d
        self.cands.append(cand)
        self.ptr += 1

    # Call when rollout phase it over
    def finish(self, last_val):
        adv = np.zeros_like(self.r)
        gae = 0
        for t in reversed(range(self.ptr)):
            # Compute GAE-lambda 
            mask = 1 - self.done[t]
            delta = self.r[t] + self.GAMMA * mask * (last_val if t == self.ptr-1 else self.v[t+1]) - self.v[t]
            gae = delta + self.GAMMA * self.GAE_LAMBDA * mask * gae
            adv[t] = gae
        self.ret = adv + self.v[:self.ptr]          # This return is for loss calcultion, we dont use raw reward
        self.adv = (adv - adv.mean()) / (adv.std() + 1e-8)           # Normalize advantage

    # Retrieve training batches
    def get(self, batch_size=32):
        idx = np.arange(self.ptr)
        np.random.shuffle(idx)
        for s in range(0, self.ptr, batch_size):
            j = idx[s:s + batch_size]
            yield (self.s[j], self.a_idx[j], self.logp[j], self.adv[j], self.ret[j], [self.cands[k] for k in j])
    
    # Clear buffer
    def clear(self):
        self.ptr=0; self.cands.clear()
        
