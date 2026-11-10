# 算法定价与合谋 / Algorithmic Pricing and Collusion

Independent Python implementation of the symmetric duopoly, memory-one baseline in Calvano, Calzolari, Denicolò and Pastorello (2020), AER 110(10), 3267–3297. DOI: https://doi.org/10.1257/aer.20190623

This is a limited classroom reproduction, not the original Fortran package or full paper replication. No API calls are made. No student ID is included.

## Run
```sh
python -m pip install -r requirements.txt
python verify.py
python run_experiment.py --steps 2000000 --seeds 0 1 2 --output results
```

The reference setup uses n=2, c=1, a=2, a0=0, mu=.25, discount=.95, 15 grid prices, grid extension .1 and memory k=1. The representative learning settings are alpha=.15, beta=4e-6. Initialization is discounted expected stage profit against a uniform rival. Both agents act simultaneously; lowest-price argmax resolves ties.

Convergence means the greedy policy at every state remains unchanged for 100000 consecutive periods. A limited run that reaches its step cap is explicitly marked not converged. Frozen-policy outcomes at a cap are diagnostics, not converged paper estimates. A supracompetitive price alone is not proof of tacit collusion; equilibrium and punishment diagnostics are additional work.

The article uses 1000 sessions per experiment and a cap of one billion periods. This project starts with three seeds and a two-million-step cap. The full alpha/beta grid and robustness exercises are not yet run. Independent NumPy RNG streams differ from the authors' implementation.

The one-period deviation diagnostic freezes learning and exploration and forces a myopic best response at period 5. It is an explicitly labeled classroom adaptation, not a replication claim for Figure 6.

## Stage 2: best responses and price-cut diagnostics
```sh
python analyze_strategies.py
python verify_diagnostics.py
python plot_diagnostics.py
python update_stage2_report.py
```
The stage-2 price-cut experiment restricts the forced action to a strictly lower price. Both firms are tested separately, learning and exploration frozen. The nondeviator acts simultaneously and first reacts in the following period. This refines the initial stage-1 diagnostic.

Best-response value iteration solves each firm's discounted MDP against the rival's fixed policy on the same grid. Current-policy evaluation and optimal-policy values are independently checked by linear systems, including all three saved policy pairs. Both converged sessions fail the best-response test despite transient price-cut responses and returns to their original cycles. This is not evidence of a collusive equilibrium, nor a replication of the article's equilibrium frequencies. Full results are in `results/strategy_diagnostics.json`; all three figures are in `figures/`.

Saved NPZ files contain Q tables, final policies, the grid and state. JSON files contain settings, runtimes, block averages, convergence and frozen-cycle diagnostics. Reuse a different output directory for new runs; existing outputs are never silently overwritten.

## Stage 3: fixed follow-up sensitivity

Nine new sessions use seeds 10, 11, 12 and a common three-million-period cap. Settings are baseline (alpha=.15, beta=4e-6), slower learning (alpha=.10 only), and faster exploration decay (beta=8e-6 only). This design was fixed after inspecting stages 1–2, before its nine runs; it is not a preregistration of the whole project. Source/protocol hashes are saved in `stage3_protocol.json`.

All nine runs reached the policy-stability criterion; none passed the joint on-cycle best-response test. Mean frozen prices are 1.7894, 1.8791 and 1.8776 respectively. These are descriptive three-session comparisons, not significance tests or evidence that every stable high-price policy is an equilibrium. The original three shorter-cap sessions are kept separate.

- `robustness/summary.csv`: every new session.
- `robustness/group_summary.json`: converged-session group means.
- `robustness/audit.json`: recomputation and source-hash audit.
- `robustness/<setting>/`: JSON and NPZ outputs for all nine sessions.
- `figures/robustness_*.png`: learning histories and outcome comparisons.

To audit/rebuild the stage-3 summary without training:
```sh
python summarize_robustness.py
```
To repeat the complete nine-session experiment in a NEW directory without replacing stored evidence:
```sh
python -m pip install -r requirements.txt
python reproduce_stage3.py
```
This launches three CPU workers and may take several minutes. The new directory is printed and saved below `reproductions/`. Python/NumPy versions for the recorded runs are in each JSON. The source runner refuses to overwrite an existing protocol; use the reproduction launcher for new runs.

This is learning-parameter sensitivity. Firm-count, cost-asymmetry and stochastic-demand robustness from the paper have not been implemented.

## Final submission

Open `index.html` for the Chinese–English report. It includes the research question, actual AI-assisted workflow, all three stages, diagnostics and limitations. The report identifies RICHINA ORTANCIA DOMINIQUE（杜米妮）; no student ID is published.

The complete project includes 12 recorded sessions and five figures. `submission_manifest.json` records file SHA-256 hashes. After rebuilding any report sections, run `python finalize_submission.py` to finalize the report and refresh the submission archive and manifest. This does not retrain models.

For GitHub Pages, publish branch `main` from `/ (root)` in repository Settings → Pages. The repository itself contains the files even without Pages. The article PDF is excluded from redistribution.

## Windows line endings

`.gitattributes` disables automatic line-ending conversion so source and protocol hashes remain valid on Windows. If an earlier checkout converted files to CRLF, make a fresh clone after this update before running hash-based audits. The repository files and recorded numerical results are unchanged.
