"""Independent Python implementation of Calvano et al. (AER 2020), duopoly k=1.

No API calls or business data. The authors' original code has not been audited.
"""
from dataclasses import dataclass, asdict
import math
import time
import numpy as np

@dataclass(frozen=True)
class Config:
    cost: float = 1.0
    quality: float = 2.0
    outside: float = 0.0
    mu: float = 0.25
    discount: float = 0.95
    learning_rate: float = 0.15
    beta: float = 4e-6
    actions: int = 15
    extension: float = 0.1
    stable_periods: int = 100_000

def shares(p1, p2, c):
    e1 = np.exp((c.quality-np.asarray(p1))/c.mu)
    e2 = np.exp((c.quality-np.asarray(p2))/c.mu)
    den = e1+e2+math.exp(c.outside/c.mu)
    return e1/den, e2/den

def bisect(fn, lo, hi):
    if fn(lo)*fn(hi) >= 0: raise ValueError('Root not bracketed')
    for _ in range(100):
        mid=(lo+hi)/2
        if fn(lo)*fn(mid) <= 0: hi=mid
        else: lo=mid
    return (lo+hi)/2

def environment(c):
    # Symmetric continuous-price Nash and joint-profit FOCs.
    nash=bisect(lambda p: p-c.cost-c.mu/(1-shares(p,p,c)[0]),c.cost,5.)
    monopoly=bisect(lambda p: p-c.cost-c.mu/(1-2*shares(p,p,c)[0]),c.cost,5.)
    grid=np.linspace(nash-c.extension*(monopoly-nash),monopoly+c.extension*(monopoly-nash),c.actions)
    a,b=np.meshgrid(grid,grid,indexing='ij')
    q1,q2=shares(a,b,c)
    profit=np.stack([(a-c.cost)*q1,(b-c.cost)*q2],axis=-1)
    pn=float((nash-c.cost)*shares(nash,nash,c)[0])
    pm=float((monopoly-c.cost)*shares(monopoly,monopoly,c)[0])
    return grid,profit,dict(nash_price=nash,monopoly_price=monopoly,nash_profit=pn,monopoly_profit=pm)

def frozen_cycle(policy, state, m):
    seen={}; sequence=[]
    while state not in seen:
        seen[state]=len(sequence)
        a=int(policy[0,state]); b=int(policy[1,state])
        sequence.append((state,a,b)); state=a*m+b
    return sequence[seen[state]:]

def deviation_path(policy, state, grid, profit, periods=30):
    """Freeze learning/exploration; one-period myopic best deviation by firm 1.

    Classroom adaptation, not a claimed exact replication of Figure 6's protocol.
    """
    out=[];m=len(grid)
    for t in range(periods):
        a=int(policy[0,state]);b=int(policy[1,state])
        if t==5: a=int(np.argmax(profit[:,b,0]))
        out.append(dict(period=t,firm1_price=float(grid[a]),firm2_price=float(grid[b]),forced_deviation=t==5))
        state=a*m+b
    return out

def run(c,seed,max_steps=2_000_000,block=10_000,output=None):
    grid,profit,reference=environment(c);m=c.actions;n=m*m
    # Initial Q is discounted profit against a uniformly random rival, all states.
    initial=np.stack([profit[:,:,0].mean(axis=1),profit[:,:,1].mean(axis=0)])/(1-c.discount)
    q=np.repeat(initial[:,None,:],n,axis=1)
    policy=q.argmax(axis=2); value=q.max(axis=2)
    rng=np.random.default_rng(seed);state=int(rng.integers(n));stable=0
    history=[];total_prices=np.zeros(2);total_profits=np.zeros(2);count=0;beg=time.perf_counter()
    for t in range(max_steps):
        if t%block==0: randoms=rng.random((min(block,max_steps-t),4))
        r=randoms[t%block];eps=math.exp(-c.beta*t)
        a=int(r[2]*m) if r[0]<eps else int(policy[0,state])
        b=int(r[3]*m) if r[1]<eps else int(policy[1,state])
        next_state=a*m+b
        # Both targets are computed before either Q-table changes.
        target0=profit[a,b,0]+c.discount*value[0,next_state]
        target1=profit[a,b,1]+c.discount*value[1,next_state]
        changed=False
        for i,action,target in [(0,a,target0),(1,b,target1)]:
            q[i,state,action]+=c.learning_rate*(target-q[i,state,action])
            # argmax selects the lowest price on ties, as specified in the article.
            new=int(q[i,state].argmax());changed|=new!=policy[i,state]
            policy[i,state]=new;value[i,state]=q[i,state,new]
        stable=0 if changed else stable+1
        state=next_state;total_prices+=grid[[a,b]];total_profits+=profit[a,b];count+=1
        if count==block or stable>=c.stable_periods or t==max_steps-1:
            history.append(dict(step=t+1,epsilon=eps,firm1_price=float(total_prices[0]/count),firm2_price=float(total_prices[1]/count),profit_per_firm=float(total_profits.mean()/count)))
            total_prices[:]=0;total_profits[:]=0;count=0
        if stable>=c.stable_periods: break
    cycle=frozen_cycle(policy,state,m)
    avgprofit=float(np.mean([profit[a,b].mean() for _,a,b in cycle]))
    result=dict(config=asdict(c),seed=seed,steps=t+1,max_steps=max_steps,converged=stable>=c.stable_periods,stable_periods_observed=stable,seconds=time.perf_counter()-beg,reference=reference,cycle_length=len(cycle),frozen_mean_price=float(np.mean([grid[[a,b]] for _,a,b in cycle])),frozen_profit_per_firm=avgprofit,profit_gain=(avgprofit-reference['nash_profit'])/(reference['monopoly_profit']-reference['nash_profit']),history=history,deviation=deviation_path(policy,cycle[0][0],grid,profit))
    if output is not None: np.savez_compressed(output,q=q,policy=policy,prices=grid,state=state)
    return result
