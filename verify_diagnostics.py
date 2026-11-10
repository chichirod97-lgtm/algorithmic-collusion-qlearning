"""Independent linear-system checks of the infinite-horizon diagnostics."""
import numpy as np
from pathlib import Path
from model import Config,environment
from analyze_strategies import values,mdp
c=Config();grid,profit,_=environment(c);m=c.actions;n=m*m;s=np.arange(n)
policy=np.zeros((2,n),dtype=int)
# Against a constant opponent the optimal repeated response is the static response.
for player in [0,1]:
    v,opt,q,res,ownres=values(policy,player,profit,c.discount)
    rewards=profit[:,0,0] if player==0 else profit[0,:,1]
    assert np.allclose(opt,rewards.max()/(1-c.discount),atol=1e-8)
    # Independently solve (I-delta P)V = reward for the current policy.
    nxt,r=mdp(policy,player,profit,m)
    p=np.zeros((n,n));p[s,nxt[s,policy[player]]]=1
    direct=np.linalg.solve(np.eye(n)-c.discount*p,r[s,policy[player]])
    assert np.allclose(v,direct,atol=1e-8)
    # Evaluate the greedy optimal policy using a linear system too.
    br=(r+c.discount*opt[nxt]).argmax(axis=1)
    p[:]=0;p[s,nxt[s,br]]=1
    direct_opt=np.linalg.solve(np.eye(n)-c.discount*p,r[s,br])
    assert np.allclose(opt,direct_opt,atol=1e-8)
for file in (Path(__file__).parent/'results').glob('seed_*.npz'):
    learned=np.load(file,allow_pickle=False)['policy']
    for player in [0,1]:
        v,opt,q,res,ownres=values(learned,player,profit,c.discount)
        nxt,r=mdp(learned,player,profit,m)
        for choices,expected in [(learned[player],v),((r+c.discount*opt[nxt]).argmax(axis=1),opt)]:
            p=np.zeros((n,n));p[s,nxt[s,choices]]=1
            direct=np.linalg.solve(np.eye(n)-c.discount*p,r[s,choices])
            assert np.allclose(expected,direct,atol=1e-8)
print('PASS: both players, static best-response benchmark and independent policy-value linear systems.')
