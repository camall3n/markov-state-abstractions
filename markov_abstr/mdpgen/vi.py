import numpy as np

def vi(mdp, pi=None):
    V_max = mdp.R_max/(1-mdp.gamma)
    V_min = mdp.R_min/(1-mdp.gamma)
    q = [V_min * np.ones((mdp.n_states)) for _ in range(mdp.n_actions)]
    for i in range(1000):
        new_q = np.copy(q)
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                q_target = 0# return
                for next_s in range(mdp.n_states):
                    pr_sas = mdp.T[a][s][next_s]
                    r_sas = mdp.R[a][s][next_s]
                    if pi is None:
                        next_v = max([q[a][next_s] for a in range(mdp.n_actions)])
                    else:
                        next_v = q[pi[next_s]][next_s]
                    q_target += pr_sas * (r_sas + mdp.gamma * next_v)
                new_q[a][s] = q_target
        if all(np.allclose(new_q[a], q[a]) for a in range(mdp.n_actions)):
            break
        else:
            q = new_q

    v = np.empty_like(q[0])
    if pi is None:
        pi = np.zeros_like(q[0], dtype=np.int)
        for s in range(mdp.n_states):
            sorted_q = sorted([(q[a][s],a) for a in range(mdp.n_actions)])
            v[s] = sorted_q[-1][0]
            pi[s] = sorted_q[-1][1]
        pi = pi.squeeze()
    else:
        for s in range(mdp.n_states):
            v[s] = q[pi[s]][s]
    v = v.squeeze()
    return v, q, pi
