"""Exact finite-state best-response diagnostics for saved frozen policies.

Q-learning is disabled. Best responses optimize discounted infinite-horizon
profit against the opponent's fixed memory-one strategy, on the same price grid.
The deviation experiment is an explicitly specified classroom adaptation.
"""
from pathlib import Path
import json,csv
import numpy as np
from model import Config,environment,frozen_cycle

ROOT=Path(__file__).parent

def mdp(policy, player, profit, m):
    other=policy[1-player]
    actions=np.arange(m)[None,:]
    if player==0:
        nxt=actions*m+other[:,None]
        rewards=profit[actions,other[:,None],0]
    else:
        nxt=other[:,None]*m+actions
        rewards=profit[other[:,None],actions,1]
    return nxt,rewards

def values(policy, player, profit, discount, tol=1e-11):
    m=profit.shape[0];states=np.arange(m*m);nxt,rewards=mdp(policy,player,profit,m)
    chosen=policy[player];pi_next=nxt[states,chosen];pi_reward=rewards[states,chosen]
    v=np.zeros(m*m)
    for _ in range(10000):
        new=pi_reward+discount*v[pi_next]
        if np.max(np.abs(new-v))<tol: v=new;break
        v=new
    else: raise RuntimeError('Policy evaluation failed to converge')
    optimum=np.zeros(m*m)
    for _ in range(10000):
        new=np.max(rewards+discount*optimum[nxt],axis=1)
        if np.max(np.abs(new-optimum))<tol: optimum=new;break
        optimum=new
    else: raise RuntimeError('Best-response value iteration failed to converge')
    residual=float(np.max(np.abs(np.max(rewards+discount*optimum[nxt],axis=1)-optimum)))
    q=rewards+discount*v[nxt]
    own_residual=float(np.max(np.abs(pi_reward+discount*v[pi_next]-v)))
    assert np.all(optimum>=v-1e-8)
    return v,optimum,q,residual,own_residual

def paths(policy, start, player, grid, profit, horizon=100):
    """At t=5 force the highest-stage-profit strictly lower price for one period.

    Both players otherwise follow their frozen learned policies. Simultaneous
    actions imply the nondeviator cannot react until t=6.
    """
    m=len(grid);baseline=start;deviated=start;rows=[];cycle_states={s for s,_,_ in frozen_cycle(policy,start,m)}
    return_time=None;forced=None
    for t in range(horizon):
        normal=[int(policy[i,baseline]) for i in [0,1]]
        actual=[int(policy[i,deviated]) for i in [0,1]]
        if t==5:
            current=actual[player]
            if current==0: raise ValueError('No strictly lower price on grid')
            candidate=profit[:current,actual[1],0] if player==0 else profit[actual[0],:current,1]
            actual[player]=int(np.argmax(candidate));forced=actual[player]
        record=dict(period=t,baseline_firm1=float(grid[normal[0]]),baseline_firm2=float(grid[normal[1]]),deviation_firm1=float(grid[actual[0]]),deviation_firm2=float(grid[actual[1]]),forced=t==5,baseline_state=baseline,deviation_state=deviated)
        rows.append(record)
        if t>5 and deviated in cycle_states and return_time is None: return_time=t-5
        baseline=normal[0]*m+normal[1];deviated=actual[0]*m+actual[1]
    before=rows[5][f'baseline_firm{player+1}'];after=rows[5][f'deviation_firm{player+1}']
    other=1-player
    reaction=rows[6][f'deviation_firm{other+1}']-rows[6][f'baseline_firm{other+1}']
    return dict(deviator=player+1,forced_before=before,forced_after=after,
                opponent_next_period_price_change=reaction,return_to_original_cycle_after_periods=return_time,horizon=horizon,rows=rows)

def main():
    all_results=[]
    for f in sorted((ROOT/'results').glob('seed_*.json')):
        r=json.loads(f.read_text(encoding='utf-8'));seed=r['seed'];c=Config(**r['config'])
        saved=np.load(ROOT/'results'/f'seed_{seed}.npz',allow_pickle=False)
        policy=saved['policy'];state=int(saved['state']);grid,profit,ref=environment(c)
        assert np.array_equal(grid,saved['prices'])
        assert np.array_equal(policy,saved['q'].argmax(axis=2))
        cycle=frozen_cycle(policy,state,c.actions);states=np.array([s for s,_,_ in cycle])
        diagnostics=[]
        for i in [0,1]:
            v,opt,q,residual,own_residual=values(policy,i,profit,c.discount)
            gain=opt-v;relative=100*gain/opt
            diagnostics.append(dict(player=i+1,max_discounted_gain_on_cycle=float(gain[states].max()),max_percent_value_loss_on_cycle=float(relative[states].max()),max_percent_value_loss_all_states=float(relative.max()),on_cycle_best_response=bool(gain[states].max()<1e-7),all_states_best_response=bool(gain.max()<1e-7),optimal_bellman_residual=residual,policy_bellman_residual=own_residual,best_response_value_error_bound=residual/(1-c.discount)))
        all_results.append(dict(seed=seed,training_converged=r['converged'],cycle_length=len(cycle),joint_on_cycle_best_response=all(x['on_cycle_best_response'] for x in diagnostics),joint_all_states_best_response=all(x['all_states_best_response'] for x in diagnostics),players=diagnostics,deviations=[paths(policy,cycle[0][0],i,grid,profit) for i in [0,1]]))
    (ROOT/'results/strategy_diagnostics.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
    for r in all_results:
        print(json.dumps({k:v for k,v in r.items() if k!='deviations'},indent=2))
        print('Deviations:',[{k:v for k,v in x.items() if k!='rows'} for x in r['deviations']])
    with (ROOT/'results/strategy_summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['seed','training_converged','player','max_percent_value_loss_on_cycle','max_percent_value_loss_all_states','on_cycle_best_response','all_states_best_response']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in all_results:
            for p in r['players']:w.writerow({k:dict(seed=r['seed'],training_converged=r['training_converged'],**p)[k] for k in fields})

if __name__=='__main__':main()
