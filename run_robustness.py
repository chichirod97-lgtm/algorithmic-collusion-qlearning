"""Fixed stage-3 sensitivity design, three independent sessions per setting.

No rerunning failures automatically. Three processes; raw outputs are preserved.
"""
from pathlib import Path
from dataclasses import replace,asdict
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,hashlib,platform,os,time
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from model import Config,run
from analyze_strategies import values,frozen_cycle

ROOT=Path(__file__).parent
OUT=ROOT/'robustness'
SCENARIOS={'baseline':{},'slower_learning':{'learning_rate':0.10},'faster_exploration_decay':{'beta':8e-6}}
SEEDS=[10,11,12]
CAP=3_000_000

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(job):
    name,seed,protocol_sha=job
    c=replace(Config(),**SCENARIOS[name]);dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'seed_{seed}.json'
    if path.exists():raise FileExistsError(path)
    start=time.perf_counter()
    r=run(c,seed,CAP,output=dest/f'seed_{seed}.npz')
    r['scenario']=name;r['protocol_sha256']=protocol_sha
    r['runtime']=dict(python=platform.python_version(),numpy=np.__version__)
    saved=np.load(dest/f'seed_{seed}.npz',allow_pickle=False)
    from model import environment
    grid,profit,_=environment(c);policy=saved['policy'];state=int(saved['state'])
    cycle=frozen_cycle(policy,state,c.actions);cycle_states=np.array([s for s,_,_ in cycle])
    checks=[]
    for player in [0,1]:
        v,opt,q,res,ownres=values(policy,player,profit,c.discount)
        diff=opt-v
        checks.append(dict(player=player+1,on_cycle_best_response=bool(diff[cycle_states].max()<1e-7),max_percent_value_loss_on_cycle=float((100*diff[cycle_states]/opt[cycle_states]).max()),best_response_bellman_residual=res))
    r['strategy_checks']=checks
    r['joint_on_cycle_best_response']=all(x['on_cycle_best_response'] for x in checks)
    r['worker_seconds']=time.perf_counter()-start
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(r,indent=2),encoding='utf-8');tmp.replace(path)
    return dict(scenario=name,seed=seed,steps=r['steps'],converged=r['converged'],price=r['frozen_mean_price'],gain=r['profit_gain'],joint_best_response=r['joint_on_cycle_best_response'])

def main():
    OUT.mkdir(exist_ok=True);pp=ROOT/'stage3_protocol.json'
    if pp.exists():raise FileExistsError('Protocol already exists. Inspect existing runs; do not silently restart.')
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Descriptive stage-3 sensitivity after inspecting stages 1 and 2; not a preregistration of the full research.',scenarios={name:asdict(replace(Config(),**change)) for name,change in SCENARIOS.items()},seeds=SEEDS,max_steps=CAP,workers=3,source_sha256={name:digest(ROOT/name) for name in ['model.py','analyze_strategies.py','run_robustness.py']},outcomes=['convergence','frozen-cycle price and normalized profit (converged sessions only for headline comparison)','on-cycle joint best response','value losses'],policy='Include all nine sessions and failures. Do not tune or extend settings based on outcomes. Old capped sessions are not pooled with this common-cap comparison. No p-values or population claims from three sessions per setting.')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8');phash=digest(pp)
    jobs=[(name,seed,phash) for name in SCENARIOS for seed in SEEDS]
    print('Protocol written before execution; 9 sessions, 3 workers.',flush=True)
    failed=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs={pool.submit(worker,j):j for j in jobs}
        for f in as_completed(fs):
            try:print(json.dumps(f.result()),flush=True)
            except Exception as e:
                failed.append(dict(job=fs[f][:2],error=repr(e)));print('FAILED '+repr(e),flush=True)
    (OUT/'failures.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
    if failed:raise RuntimeError('Some sessions failed; see failures.json; no automatic retry.')
if __name__=='__main__':main()
