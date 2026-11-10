"""Run a limited classroom reproduction; all settings are saved with the results."""
from pathlib import Path
from dataclasses import replace
import argparse,json,platform
import numpy as np
from model import Config,run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=2_000_000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--output',default='results')
    p.add_argument('--beta',type=float,default=4e-6)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if args.steps<1: p.error('steps must be positive')
    for seed in args.seeds:
        dest=root/f'seed_{seed}.json'
        if dest.exists(): raise FileExistsError(f'Use a new output directory: {dest}')
        r=run(replace(Config(),beta=args.beta),seed,args.steps,output=root/f'seed_{seed}.npz')
        r['runtime']=dict(python=platform.python_version(),numpy=np.__version__)
        dest.write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(f"seed={seed} steps={r['steps']} converged={r['converged']} price={r['frozen_mean_price']:.4f} gain={r['profit_gain']:.4f}",flush=True)
if __name__=='__main__': main()
