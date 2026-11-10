"""Reproduce stage 3 in a new directory without replacing saved evidence.

Install requirements first. This launches nine CPU sessions with three workers;
it may take several minutes. No API calls or external uploads are made.
"""
from pathlib import Path
import shutil,subprocess,sys,time

def main():
    source=Path(__file__).parent
    destination=source/'reproductions'/f'run_{time.time_ns()}'
    destination.mkdir(parents=True,exist_ok=False)
    for f in source.iterdir():
        if f.is_file() and f.suffix in ['.py','.html','.md','.txt','.ipynb']:
            shutil.copy2(f,destination/f.name)
    for name in ['results','figures']:
        if (source/name).exists():shutil.copytree(source/name,destination/name)
    (destination/'figures').mkdir(exist_ok=True)
    print(f'New reproduction directory: {destination}',flush=True)
    subprocess.run([sys.executable,'run_robustness.py'],cwd=destination,check=True)
    subprocess.run([sys.executable,'summarize_robustness.py'],cwd=destination,check=True)
    print(f'Finished: {destination / "index.html"}',flush=True)

if __name__=='__main__':main()
