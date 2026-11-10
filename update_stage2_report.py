from pathlib import Path
import json,html
root=Path(__file__).parent
d=json.loads((root/'results/strategy_diagnostics.json').read_text(encoding='utf-8'))
rows=''
for r in d:
    rows+='<tr><td>'+str(r['seed'])+'</td><td>'+('是 / Yes' if r['training_converged'] else '否 / No')+'</td>'
    rows+=''.join(f"<td>{p['max_percent_value_loss_on_cycle']:.3f}%</td>" for p in r['players'])+'<td>未通过 / Not passed</td></tr>'
section='''<section id="stage2"><h2>第二阶段：最佳反应与降价测试 / Stage 2: best responses and deviations</h2>
<p>对手策略固定为已保存的记忆一期策略，在同一15价格网格上用价值迭代求最佳反应。当前策略的折现价值通过策略评估计算。所有测试关闭学习与探索。 / Holding the opponent's saved memory-one policy fixed, value iteration computes a best response on the same 15-price grid. Policy evaluation computes the current policy's discounted value. Learning and exploration are disabled.</p>
<p>价值损失 = 100 × (V*−V策略)/V*，在最终循环状态取最大值。数值Bellman残差低于10⁻¹¹，最佳反应价值误差界低于2×10⁻¹⁰。 / Value loss is 100 × (V*−Vpolicy)/V*, maximized over states on the final cycle. Bellman residuals are below 10⁻¹¹; optimal-value error bounds are below 2×10⁻¹⁰.</p>
<div class="scroll"><table><tr><th>Seed</th><th>训练收敛 / Stable</th><th>企业1损失 / Firm 1 loss</th><th>企业2损失 / Firm 2 loss</th><th>最佳反应 / Best response</th></tr>ROWS</table></div>
<p><strong>两个收敛试验均未通过最佳反应检验。</strong> / Both converged runs fail the best-response test. This distinguishes observed strategy stability from an equilibrium of the restricted repeated game. Three runs do not estimate the paper's equilibrium frequency.</p>
<h3>训练价格 / Training prices</h3><img style="width:100%;height:auto" src="figures/price_evolution.png" alt="Price learning curves for the three seeds">
<h3>一次降价后的响应 / Response to a one-period price cut</h3>
<p>在t=0强制一家企业选择严格低于原价、当期利润最高的网格价格，仅持续一期。另一企业同时行动，因此最早在下一期响应。此后两家恢复冻结策略。分别对两家实施，绘图虚线表示未干预基线。 / At t=0 force one firm to choose the stage-profit-maximizing strictly lower grid price for one period. Simultaneous play means its rival first reacts in the next period. Both then follow frozen policies. Each firm is tested separately; dotted lines show the untreated baseline.</p>
<p>种子1中，原循环在降价后6期或8期恢复；种子2为7期或8期。对手下一期均降价，但此类响应并未排除其他有利偏离。该测试为课堂改编，不声称精确复现原文Figure 6。 / Seed 1 returns to its original cycle after 6 or 8 periods; seed 2 after 7 or 8. The rival cuts its next-period price in all four tests, but these responses do not rule out other profitable deviations. This is a classroom adaptation, not an exact Figure 6 replication.</p>
<img style="width:100%;height:auto" src="figures/deviation_responses.png" alt="Four frozen-policy price-cut responses"><img style="width:100%;height:auto" src="figures/best_response_gaps.png" alt="Both firms have positive optimal-value losses">
<p>独立验证：对固定对手的最佳反应退化为静态最佳反应；通过线性方程组独立验证策略价值与最优价值。 / Independent checks: against a constant opponent, the repeated best response equals the static best response; linear-system solutions independently confirm policy and optimal values.</p>
<p>结论：观察到高价、稳定和降价响应，但当前样本不能认定达到合谋均衡。下一阶段应固定扩展方案，增加独立种子和稳健性设置，而不是为获得预期结果反复挑选参数。 / Conclusion: high prices, stability and price-cut responses are observed, but these runs do not establish a collusive equilibrium. Prespecify additional seeds and robustness settings rather than selecting parameters to obtain a desired result.</p>
<p><a href="results/strategy_summary.csv">策略核验CSV / Strategy summary</a> · <a href="results/strategy_diagnostics.json">完整诊断 / Full diagnostics</a> · <a href="analyze_strategies.py">分析代码 / Analysis code</a> · <a href="verify_diagnostics.py">独立核验 / Independent checks</a> · <a href="plot_diagnostics.py">绘图代码 / Plotting code</a></p></section>'''.replace('ROWS',rows)
p=root/'index.html';text=p.read_text(encoding='utf-8')
if '<section id="stage2">' in text:
    start=text.index('<section id="stage2">');end=text.index('</section>',start)+len('</section>')
    text=text[:start]+text[end:]
p.write_text(text.replace('</body>',section+'</body>'),encoding='utf-8')
print('Stage 2 added to bilingual HTML report.')
