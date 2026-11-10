"""Create publication-exportable diagnostic figures from saved results only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).parent
cn=Path('C:/Windows/Fonts/msyh.ttc')
if cn.exists():
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(str(cn))
    plt.rcParams['font.family']=FontProperties(fname=str(cn)).get_name()
plt.rcParams['axes.unicode_minus']=False
out=ROOT/'figures';out.mkdir(exist_ok=True)
runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'results').glob('seed_*.json'))]
diags=json.loads((ROOT/'results/strategy_diagnostics.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(figsize=(10,5.2))
for r in runs:
    h=r['history'];label=f"Seed {r['seed']} — "+('已稳定 / stable' if r['converged'] else '未收敛 / capped')
    ax.plot([x['step'] for x in h],[(x['firm1_price']+x['firm2_price'])/2 for x in h],label=label)
ref=runs[0]['reference']
ax.axhline(ref['nash_price'],color='black',ls='--',label='竞争 / Nash')
ax.axhline(ref['monopoly_price'],color='gray',ls=':',label='垄断 / Monopoly')
ax.set(xlabel='训练期数 / Training period',ylabel='区间平均价格 / Block mean price',title='价格学习过程 / Price learning dynamics')
ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'price_evolution.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
for row,seed in enumerate([1,2]):
    r=next(x for x in diags if x['seed']==seed)
    for col,d in enumerate(r['deviations']):
        ax=axs[row,col];rows=d['rows'][:25];x=[v['period']-5 for v in rows]
        for player,color in [(1,'#2166ac'),(2,'#b35806')]:
            ax.plot(x,[v[f'deviation_firm{player}'] for v in rows],marker='o',ms=3,label=f'企业 / Firm {player}',color=color)
            ax.plot(x,[v[f'baseline_firm{player}'] for v in rows],ls=':',color=color,alpha=.6)
        ax.axvline(0,color='gray',ls='--',lw=1)
        ax.set_title(f"Seed {seed} · 降价企业 / Deviator {d['deviator']}")
        ax.grid(alpha=.2);ax.set_xlabel('相对降价期 / Period relative to deviation');ax.set_ylabel('价格 / Price')
axs[0,0].legend(fontsize=8)
fig.suptitle('冻结策略下的一期降价测试 / One-period price-cut test with frozen policies')
fig.tight_layout();fig.savefig(out/'deviation_responses.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for offset,player,color in [(-.18,1,'#2166ac'),(.18,2,'#b35806')]:
    vals=[r['players'][player-1]['max_percent_value_loss_on_cycle'] for r in diags]
    bars=ax.bar([r['seed']+offset for r in diags],vals,width=.34,color=color,label=f'企业 / Firm {player}')
    ax.bar_label(bars,fmt='%.2f',padding=3)
ax.set_xticks([0,1,2],['Seed 0\n未收敛 / capped','Seed 1','Seed 2'])
ax.set(ylabel='最优价值损失 / Optimal-value loss (%)',title='稳定不等于最佳反应 / Stable policy does not imply best response',ylim=(0,22))
ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(out/'best_response_gaps.png',dpi=200);plt.close(fig)
print('Three diagnostic figures saved.')
