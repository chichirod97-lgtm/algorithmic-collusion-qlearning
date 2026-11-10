"""Audit and report the fixed stage-3 design; do not train or select sessions."""
from pathlib import Path
from collections import defaultdict
import json,csv,hashlib,math,html
import numpy as np
from model import Config,environment,frozen_cycle
ROOT=Path(__file__).parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    protocol=json.loads((ROOT/'stage3_protocol.json').read_text(encoding='utf-8'))
    phash=digest(ROOT/'stage3_protocol.json')
    for name,value in protocol['source_sha256'].items():assert digest(ROOT/name)==value
    runs=[];records=[];hashes={}
    for scenario,cfg in protocol['scenarios'].items():
        for seed in protocol['seeds']:
            f=ROOT/'robustness'/scenario/f'seed_{seed}.json'
            r=json.loads(f.read_text(encoding='utf-8'));runs.append(r)
            assert r['config']==cfg and r['seed']==seed and r['scenario']==scenario
            assert r['protocol_sha256']==phash and r['max_steps']==protocol['max_steps']
            assert 0<r['steps']<=r['max_steps']
            assert r['converged']==(r['stable_periods_observed']>=cfg['stable_periods'])
            if not r['converged']:assert r['steps']==r['max_steps']
            c=Config(**cfg);grid,profit,refs=environment(c)
            saved=np.load(f.with_suffix('.npz'),allow_pickle=False)
            policy=saved['policy'];assert np.array_equal(policy,saved['q'].argmax(axis=2))
            cycle=frozen_cycle(policy,int(saved['state']),c.actions)
            mp=float(np.mean([grid[[a,b]] for _,a,b in cycle]))
            prof=float(np.mean([profit[a,b].mean() for _,a,b in cycle]))
            gain=(prof-refs['nash_profit'])/(refs['monopoly_profit']-refs['nash_profit'])
            assert math.isclose(mp,r['frozen_mean_price'],abs_tol=1e-12)
            assert math.isclose(gain,r['profit_gain'],abs_tol=1e-12)
            assert all(x['best_response_bellman_residual']<1e-9 for x in r['strategy_checks'])
            maxloss=max(x['max_percent_value_loss_on_cycle'] for x in r['strategy_checks'])
            records.append(dict(scenario=scenario,seed=seed,steps=r['steps'],converged=r['converged'],frozen_price=mp,profit_gain=gain,joint_best_response=r['joint_on_cycle_best_response'],max_player_value_loss_percent=maxloss))
            for p in [f,f.with_suffix('.npz')]:hashes[str(p.relative_to(ROOT))]=digest(p)
    assert len(records)==9
    failures=json.loads((ROOT/'robustness/failures.json').read_text(encoding='utf-8'));assert not failures
    dest=ROOT/'robustness'
    with (dest/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for scenario in protocol['scenarios']:
        rows=[r for r in records if r['scenario']==scenario];conv=[r for r in rows if r['converged']]
        summary.append(dict(scenario=scenario,sessions=len(rows),converged=len(conv),joint_best_response_converged=sum(r['joint_best_response'] for r in conv),mean_price_converged=float(np.mean([r['frozen_price'] for r in conv])) if conv else None,mean_profit_gain_converged=float(np.mean([r['profit_gain'] for r in conv])) if conv else None))
    audit=dict(status='PASS',sessions=9,protocol_sha256=phash,source_hashes_verified=True,policy_equals_argmax=True,cycle_prices_and_gains_recomputed=True,bellman_residuals_below_1e_9=True,output_hashes=hashes)
    (dest/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (dest/'group_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    draw(runs,records,summary)
    update_report(summary,records)
    print(json.dumps(summary,indent=2));print('Stage 3 audit PASS: all 9 sessions and frozen outcomes verified.')

def draw(runs,records,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import matplotlib.font_manager as fm
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():fm.fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    labels=['基准 / Baseline','较慢学习 / Slower learning','较快衰减 / Faster decay']
    fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for i,s in enumerate(summary):
        ax=axs[i]
        for r in runs:
            if r['scenario']!=s['scenario']:continue
            h=r['history'];ax.plot([x['step'] for x in h],[(x['firm1_price']+x['firm2_price'])/2 for x in h],label=f"Seed {r['seed']}"+(' (cap)' if not r['converged'] else ''))
        ref=runs[0]['reference'];ax.axhline(ref['nash_price'],ls='--',color='black',label='Nash');ax.axhline(ref['monopoly_price'],ls=':',color='gray',label='Monopoly')
        ax.set_title(labels[i]);ax.set_xlabel('训练期数 / Training periods');ax.set_xlim(0,3_000_000);ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[0].set_ylabel('区间平均价格 / Block mean price');fig.suptitle('预先固定的九次扩展实验 / Nine fixed-design extension sessions')
    fig.tight_layout();fig.savefig(ROOT/'figures/robustness_learning.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    colors=['#2166ac','#b35806','#26734d']
    for i,s in enumerate(summary):
        data=[r for r in records if r['scenario']==s['scenario']]
        for j,r in enumerate(data):
            x=i+(j-1)*.12
            axs[0].scatter(x,r['profit_gain'],color=colors[i],marker='o' if r['converged'] else 'x',s=65)
            axs[1].scatter(x,r['max_player_value_loss_percent'],color=colors[i],marker='o' if r['converged'] else 'x',s=65)
    for ax in axs:
        ax.set_xticks(range(3),labels,fontsize=8);ax.grid(axis='y',alpha=.2)
    axs[0].set(title='利润增益 / Normalized profit gain',ylabel='Δ = (π − πN)/(πM − πN)')
    axs[1].set(title='两企业中最大价值损失 / Maximum player value loss',ylabel='100 × (V* − Vpolicy)/V* (%)')
    fig.suptitle('每点为一次试验；圆点收敛，叉号未收敛 / One session per point; circle: converged, cross: capped')
    fig.tight_layout();fig.savefig(ROOT/'figures/robustness_outcomes.png',dpi=200);plt.close(fig)

def update_report(summary,records):
    p=ROOT/'index.html';text=p.read_text(encoding='utf-8')
    if '<section id="stage3">' in text:
        start=text.index('<section id="stage3">');end=text.index('</section>',start)+len('</section>')
        text=text[:start]+text[end:]
    rows=''
    for r in summary:
        val=lambda k:'—' if r[k] is None else f'{r[k]:.4f}'
        rows+=f"<tr><td>{html.escape(r['scenario'])}</td><td>{r['converged']}/{r['sessions']}</td><td>{val('mean_price_converged')}</td><td>{val('mean_profit_gain_converged')}</td><td>{r['joint_best_response_converged']}/{r['converged']}</td></tr>"
    allrows=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]) if not isinstance(r[k],float) else f'{r[k]:.6f}')+'</td>' for k in records[0])+'</tr>' for r in records)
    sec='''<section id="stage3"><h2>第三阶段：固定方案的敏感性实验 / Stage 3: fixed-design sensitivity</h2>
<p>在已观察前两阶段后，先保存方案与代码哈希，再运行九次新试验。这是后续探索的固定扩展方案，不是全研究事前注册。 / After inspecting stages 1–2, the design and code hashes were saved before nine new runs. This is a fixed follow-up design, not preregistration of the entire study.</p>
<p>所有组使用新种子10、11、12，每次最多3,000,000期。基准α=.15、β=4×10⁻⁶；较慢学习仅把α改为.10；较快探索衰减仅把β改为8×10⁻⁶。其余经济与算法设置不变。同一种子在各组提供匹配的随机数流。 / New seeds 10, 11, 12 and a common 3,000,000-period cap are used throughout. Baseline: α=.15, β=4×10⁻⁶. Slower learning changes only α to .10; faster exploration decay changes only β to 8×10⁻⁶. All other settings are unchanged. Seeds provide matched random streams across groups.</p>
<p>表中价格和利润均值仅使用收敛试验。全部试验，包括达到上限未收敛者，均保存在原始结果和下表。 / Headline price and profit averages use converged sessions only. All runs, including capped nonconverged sessions, are retained in the raw results and session table.</p>
<div class="scroll"><table><tr><th>Setting</th><th>Converged</th><th>Mean price, converged</th><th>Mean Δ, converged</th><th>Joint best response, converged</th></tr>GROUPROWS</table></div>
<img style="width:100%;height:auto" src="figures/robustness_learning.png" alt="Learning histories for all nine extension sessions"><img style="width:100%;height:auto" src="figures/robustness_outcomes.png" alt="All nine profit gains and value losses with capped runs marked">
<p>每组仅三次试验，不进行显著性检验，也不把观察到的差异当作总体因果效应或可靠的合谋发生率。高价、策略稳定和最佳反应为不同指标，不能互相替代。 / With only three sessions per setting, no significance tests are performed. Differences do not establish population causal effects or a reliable collusion frequency. High prices, policy stability, and best responses are distinct outcomes.</p>
<p>这是学习参数敏感性分析，尚未复现原文关于企业数量、不对称成本、需求不确定性等经济环境稳健性检验。早期2,000,000期试验未与新上限组混合计算均值。 / This tests learning-parameter sensitivity, not the paper's economic robustness exercises on firm count, asymmetric costs, or demand uncertainty. Earlier sessions with a 2,000,000-period cap are not pooled with this common-cap comparison.</p>
<h3>全部九次试验 / All nine sessions</h3><div class="scroll"><table><tr>HEADERS</tr>ALLROWS</table></div>
<p>审计通过：全部配置、种子、代码和方案哈希一致；从保存的Q表复核贪婪策略、冻结循环价格与利润增益；最佳反应Bellman残差均低于10⁻⁹。 / Audit passed: configurations, seeds, source and protocol hashes match; greedy policies, cycle prices and gains were recomputed from saved Q tables; best-response Bellman residuals are below 10⁻⁹.</p>
<p><a href="stage3_protocol.json">方案 / Protocol</a> · <a href="robustness/summary.csv">九次结果 / Session CSV</a> · <a href="robustness/group_summary.json">分组汇总 / Group summary</a> · <a href="robustness/audit.json">审计 / Audit</a> · <a href="run_robustness.py">运行代码 / Runner</a> · <a href="summarize_robustness.py">汇总代码 / Summary code</a></p></section>'''
    sec=sec.replace('GROUPROWS',rows).replace('ALLROWS',allrows).replace('HEADERS',''.join('<th>'+k+'</th>' for k in records[0]))
    p.write_text(text.replace('</body>',sec+'</body>'),encoding='utf-8')
if __name__=='__main__':main()
