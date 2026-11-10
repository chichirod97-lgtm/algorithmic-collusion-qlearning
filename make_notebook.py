from pathlib import Path
import json,ast
root=Path(__file__).parent
def md(t):return dict(cell_type='markdown',metadata={},source=t.splitlines(True))
def code(t):return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=t.splitlines(True))
execute='''# 三次训练可能耗时数分钟。 / Three runs may take several minutes.
from pathlib import Path
import json
out=Path("results"); out.mkdir(exist_ok=True)
runs=[]
for seed in [0,1,2]:
    dest=out/f"seed_{seed}.json"
    if dest.exists():
        result=json.loads(dest.read_text())
        assert result['seed']==seed and result['config']==asdict(Config())
        assert result['max_steps']==2_000_000
    else:
        result=run(Config(),seed,2_000_000,output=out/f"seed_{seed}.npz")
        dest.write_text(json.dumps(result,indent=2))
    runs.append(result)
    print(seed,result['converged'],result['frozen_mean_price'],result['profit_gain'])
'''
plot='''import matplotlib.pyplot as plt
fig, ax=plt.subplots(figsize=(10,5))
for r in runs:
    h=r['history']
    ax.plot([x['step'] for x in h],
            [(x['firm1_price']+x['firm2_price'])/2 for x in h],
            label=f"seed {r['seed']}")
ref=runs[0]['reference']
ax.axhline(ref['nash_price'],color='black',ls='--',label='Nash')
ax.axhline(ref['monopoly_price'],color='gray',ls='--',label='Monopoly')
ax.set(xlabel='Training period',ylabel='Mean price per 10,000-period block',
       title='Limited Q-learning reproduction')
ax.legend(); fig.tight_layout()
fig.savefig('price_evolution.png',dpi=200); plt.show()
'''
for t in [execute,plot]:ast.parse(t)
cells=[md('# 算法定价与合谋 / Algorithmic pricing and collusion\nCalvano et al. (AER 2020), DOI 10.1257/aer.20190623.\n部分课堂复现，非完整论文复现。 / Partial classroom reproduction, not a full-paper replication.\n此笔记本不调用付费API。 / No paid API calls.\n'),
       code('%pip -q install numpy matplotlib\n'),code((root/'model.py').read_text(encoding='utf-8')),
       md('## 运行 / Run\n达到步数上限不代表收敛。 / Reaching the step cap does not establish convergence.\n不同参数请使用新的输出目录。 / Use a new output directory for different parameters.\n'),code(execute),code(plot),
       md('## 解释 / Interpretation\n高于竞争价格不足以证明合谋。还需策略稳定性、偏离惩罚及最佳反应检验。\nPrices above Nash alone do not establish collusion. Policy stability, punishment and best-response tests are additional requirements.\n保存JSON、NPZ和图片后再关闭Colab。 / Save JSON, NPZ, and figures before closing Colab.\n')]
doc=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
(root/'Calvano_Qlearning_CN_EN.ipynb').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
print('Notebook created; embedded Python syntax checked.')
