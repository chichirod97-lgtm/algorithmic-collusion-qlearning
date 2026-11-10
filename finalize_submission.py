"""Finalize the public report and package the recorded classroom experiment.

Does not train, alter raw results, or publish anything.
"""
from pathlib import Path
from html.parser import HTMLParser
import ast,hashlib,json,zipfile
ROOT=Path(__file__).parent

def main():
    page=ROOT/'index.html';text=page.read_text(encoding='utf-8')
    text=text.replace('Q-learning collusion — first experiment','算法定价与合谋 | Q-learning replication study')
    text=text.replace('第一阶段：三随机种子的有限实验 / Stage 1: a limited three-seed experiment','课程报告：部分复现、策略核验与敏感性分析 / Course report: partial reproduction, strategy checks and sensitivity analysis')
    if '<section id="finalsummary">' not in text:
        intro='''<section id="finalsummary"><h2>摘要与主要结论 / Abstract and main findings</h2>
<p>本研究用独立Python实现，对Calvano等（2020）的对称双寡头Q-learning定价模型进行缩减规模复现。首先运行三个初始试验，随后保存扩展方案并完成三个设置、各三个新种子的九次敏感性试验。研究同时检查价格、策略稳定性、偏离响应和最佳反应，避免把高价或收敛等同于合谋均衡。 / This study independently implements a reduced-scale reproduction of the symmetric duopoly Q-learning pricing model of Calvano et al. (2020). Three initial sessions are followed by a fixed extension with three settings and three new seeds per setting. Prices, policy stability, deviation responses and best responses are checked separately.</p>
<p>新九次试验均满足策略稳定标准。基准、较慢学习和较快探索衰减的平均冻结价格分别为1.7894、1.8791、1.8776，高于连续价格Nash参考1.4729。但九次试验均未通过联合最佳反应检验。初始两次收敛试验也未通过该检验。因此，这些运行显示高价与降价响应，而非已验证的合谋均衡。 / All nine extension sessions meet the policy-stability criterion. Mean frozen prices are 1.7894, 1.8791 and 1.8776 for baseline, slower learning and faster exploration decay, above the continuous-price Nash reference of 1.4729. None passes the joint best-response test; neither do the two converged initial sessions. These runs show high prices and price-cut responses, rather than a verified collusive equilibrium.</p>
<p>本文是课堂部分复现，不复现原文完整参数网格或合谋发生率。每组仅三次新试验，比较为描述性，不作总体因果或显著性推断。 / This is a partial classroom reproduction, not a replication of the full parameter grid or collusion frequency. Three new sessions per setting support descriptive comparisons only.</p>
<p><strong>作者 / Author:</strong> RICHINA ORTANCIA DOMINIQUE（杜米妮）<br><strong>课程 / Course:</strong> 人工智能与经济管理研究 / Artificial Intelligence and Economic Management Research</p>
<nav aria-label="Report contents"><a href="#finalsummary">摘要 / Abstract</a> · <a href="#stage2">策略诊断 / Strategy diagnostics</a> · <a href="#stage3">敏感性 / Sensitivity</a> · <a href="#reproduce">复现 / Reproduction</a> · <a href="#aiworkflow">AI过程 / AI workflow</a></nav></section>'''
        text=text.replace('<h2>研究问题 / Research question</h2>',intro+'<h2>研究问题 / Research question</h2>',1)
    if '<section id="reproduce">' not in text:
        ending='''<section id="reproduce"><h2>提交文件与复现 / Submission files and reproduction</h2>
<p>报告、五张图、全部十二次试验的JSON/NPZ模型、CSV汇总、固定扩展方案及校验代码一并提供。原文PDF不重新分发。 / Included: this report, five figures, JSON/NPZ outputs for all twelve sessions, CSV summaries, the fixed extension protocol and verification code. The article PDF is not redistributed.</p>
<h3>先检查已有结果 / Audit the stored results first</h3><pre>python -m pip install -r requirements.txt
python verify.py
python verify_diagnostics.py
python summarize_robustness.py</pre>
<p>这些命令不重新运行长训练；verify.py含两次1000期的微型重复性检查。 / These commands do not repeat the long training runs; verify.py includes two 1,000-period repeatability checks.</p>
<h3>从头复跑九次扩展 / Repeat the nine extension sessions</h3><pre>python reproduce_stage3.py</pre>
<p>该命令创建新目录、用三个CPU进程运行全部九次试验并生成报告，不覆盖已有证据。可能耗时数分钟。 / This creates a new directory, runs all nine sessions with three CPU processes, and generates a report without replacing stored evidence. It may take several minutes.</p>
<p><a href="README.md">完整说明 / Full README</a> · <a href="Calvano_Qlearning_CN_EN.ipynb">阶段1–2笔记本 / Stages 1–2 notebook</a> · <a href="requirements.txt">依赖 / Dependencies</a> · <a href="stage3_protocol.json">阶段3方案 / Stage 3 protocol</a> · <a href="robustness/audit.json">核验 / Audit</a></p>
<p>每次运行JSON记录Python/NumPy版本；各平台和依赖版本可能影响数值重现。固定随机数并不保证跨所有版本逐位一致。 / Run JSON files record Python/NumPy versions. Platform and dependency differences may affect numerical reproduction; fixed seeds do not guarantee bitwise equality across all versions.</p></section>
<section id="aiworkflow"><h2>AI辅助过程与人工责任 / AI workflow and human responsibility</h2>
<p>用户提供课程笔记与本地论文，并确认开始复现实验。实际简短指令包括“OK COMMENCONS ALORS”和后续“prochaine étape”；AI结合论文、先前讨论的实验目标和中英文报告要求，逐步构建代码与分析。这些简短指令依赖完整上下文，不是独立可复用的完整实验提示词。 / The user supplied course notes and a local article and confirmed the experiment. Actual short instructions included “OK COMMENCONS ALORS” and subsequent “prochaine étape” requests. AI used the article and prior context to build the code and analyses step by step; the short requests alone are not complete reusable experiment prompts.</p>
<ol><li>阅读论文并记录经济参数、初始化及收敛标准。 / Read the article and specify economic parameters, initialization and convergence.</li><li>编写独立Python实现，检查对称性、Nash价格和随机种子重复性。 / Implement Python and check symmetry, Nash prices and seeded repeatability.</li><li>保存三次初始运行，明确一个未收敛结果。 / Retain three initial sessions, including one capped nonconverged session.</li><li>用数值动态规划检查最佳反应，再以线性方程组独立验证价值。 / Compute best responses by dynamic programming and independently check values using linear systems.</li><li>在新运行前保存九次扩展方案、种子与代码哈希。 / Save the nine-session follow-up protocol, seeds and code hashes before execution.</li><li>保留全部结果，更新报告，并明确未通过最佳反应检验的发现。 / Retain all outcomes and report failed best-response tests explicitly.</li></ol>
<p>AI完成代码生成、执行、图表与报告整理；提交者应阅读公式、运行记录和结论，并能解释为何“稳定”不等于“最优”或“合谋均衡”。 / AI generated and executed code and prepared figures and reporting. The submitter should review the formulas, records and conclusions and explain why stability does not establish optimality or collusive equilibrium.</p>
<h3>可复用任务说明 / Reusable task specification</h3><p>以下为根据实际工作整理的说明，并非原始对话的逐字记录。 / The following specification is reconstructed from the work, not a verbatim original prompt.</p>
<blockquote>基于Calvano等（2020），实现两企业、15价格、一期记忆的Q-learning模型。保持论文基准经济参数，记录随机种子与收敛状态；分开报告高价、策略稳定、降价响应及最佳反应。保存所有配置与结果，明确有限运行、独立实现和小样本限制，输出中英双语HTML报告。<br>Implement the Calvano et al. (2020) duopoly Q-learning baseline with 15 prices and one-period memory. Record seeds and convergence; distinguish high prices, policy stability, price-cut responses and best responses. Retain all settings and results, disclose independent-implementation and small-sample limits, and produce a Chinese–English HTML report.</blockquote></section>
<section id="conclusion"><h2>结论与复现边界 / Conclusion and replication boundaries</h2><p>本项目完成了可运行、可核查的缩减复现实验与学习参数敏感性分析，但未复现原文合谋均衡的发生率。没有核对作者原始Fortran实现，亦未运行完整100×100学习参数网格、每组1000次试验，或企业数量、成本不对称和随机需求的经济稳健性实验。结果与原文不同应作为待解释的复现发现保留，而非隐藏或通过挑选参数消除。 / This project provides runnable, auditable reduced-scale experiments and learning-parameter sensitivity, but does not reproduce the paper's collusive-equilibrium frequency. The authors' original Fortran implementation has not been audited. The full 100×100 parameter grid, 1,000 sessions per experiment and economic robustness exercises remain outside scope. Differences from the paper are retained as findings to investigate, rather than removed through selective parameter choices.</p><p><a href="https://doi.org/10.1257/aer.20190623">Calvano, Calzolari, Denicolò and Pastorello (2020), American Economic Review 110(10), 3267–3297.</a></p></section>'''
        text=text.replace('</body>',ending+'</body>')
    text=text.replace('<style>','<style>nav{padding:16px 0;border-bottom:1px solid #cad5da;margin-bottom:30px}section{scroll-margin-top:24px}blockquote{margin:20px 0;padding:0 20px;border-left:3px solid #7595a4} @media print{body{background:white;margin:0;max-width:none}nav{display:none}table,img{break-inside:avoid}} ')
    page.write_text(text,encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[];self.ids=set()
        def handle_starttag(self,t,a):
            a=dict(a)
            if 'id' in a:self.ids.add(a['id'])
            for k in ['href','src']:
                if k in a:self.refs.append(a[k])
    links=Links();links.feed(text)
    for v in links.refs:
        if v.startswith('#'):assert v[1:] in links.ids,v
        elif not v.startswith(('http:','https:','data:')):assert (ROOT/v).is_file(),v
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not {'__pycache__','reproductions'}.intersection(p.parts)]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='submission_manifest.json'}
    (ROOT/'submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=ROOT.parent/'Algorithmic_Collusion_Submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if p.is_file() and not {'__pycache__','reproductions'}.intersection(p.parts):z.write(p,p.relative_to(ROOT))
    print(f'Final report references and Python syntax checked; {len(manifest)} files hashed.')
    print(archive)
if __name__=='__main__':main()
