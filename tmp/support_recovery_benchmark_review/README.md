# Benchmark and oracle review

Date: 2026-10-12. Input: `experiments/output/fixed_support_scaling_n4_upper_scope_fit_omega_ref_true_greedy`.
Only new diagnostic files were written; original simulation results and estimator code were unchanged.

## What the saved results establish

The model has p=4, three true off-diagonal edges, and six eligible strictly
upper-triangle edges. Its true support is {(0,3),(1,3),(2,3)}. The saved
configuration has `search_option=greedy`, `nested_supports=False`, estimated
fixed omega, and 10 Halton restarts. In the current implementation, this mode
searches supports independently at each cardinality, rather than following a
nested forward path. Enumeration of masks does not certify global continuous
optimization within each mask.

| N | Proposed exact /10 | Known-size saved candidate exact /10 | Correlation baseline exact /10 | Proposed mean recall | Proposed mean FP+FN |
|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 2 | 0 | .3667 | 2.7 |
| 1000 | 1 | 4 | 0 | .5667 | 1.4 |
| 10000 | 9 | 10 | 8 | 1 | .1 |
| 1000000 | 10 | 10 | 10 | 1 | 0 |

The true-size candidate is in the plateau shortlist in 6, 7, 9, and 10 trials,
respectively. Thus the selection pipeline includes both screening and testing;
the oracle gap cannot be attributed solely to the bootstrap tests. At N=10000
the one incorrect final choice excludes the true size from its shortlist.

The known-size benchmark uses the already fitted candidate at D_m=4; it adds
truth only to choose the size. It is not a certified optimum or a universal
upper bound on support recovery. The best-on-candidate-list Hamming oracle has
mean errors 1.3, .7, 0, 0, respectively; this oracle uses the actual true support
to choose among candidates. It is an optimistic reference for this finite
candidate list only.

The exploratory correlation baseline selects eligible pairs with
sqrt(N-3)*abs(atanh(r_ij)) > 2.63825727347675, corresponding to approximate
two-sided Fisher-z testing with Bonferroni alpha=.05 over six pairs. It uses
saved known-zero-mean second moments, so it is an approximation rather than an
exact centered-correlation test. Direction is supplied by the same upper-order
restriction as the proposed method. The threshold was not optimized using truth.
This star model has exactly the same off-diagonal nonzero pattern in population
Sigma and Lambda; this convenience need not hold for other graphs.

Ten trials give weak precision for comparing exact recovery probabilities.
For example, the two-sided 95% exact binomial interval for 10/10 is approximately
[.692,1], assuming independent random trials from this fixed scenario.
Repeated seeds should be paired across methods; vary graphs and coefficient
draws as well as observation seeds.

## AIC/BIC are feasible, with an important distinction

The simulator draws independent zero-mean Gaussian observations from stationary
Sigma. For stable Lambda and positive omega, define C(Lambda,omega) by

    C = Lambda.T @ C @ Lambda + omega I.

Up to constants, the negative twice-log-likelihood is

    N * [logdet(C) + trace(C^{-1} Sigma_hat)].

For each candidate support S, maximize this likelihood within its stable
parameter space. Score the maximum with AIC= -2 loglik + 2 q_S or
BIC= -2 loglik + q_S log N. The difficult step is likelihood fitting, not
deriving either of these two penalties for the Frobenius residual objective.
Adding 2 q or q log N directly to the raw squared Lyapunov residual is not
ordinary AIC/BIC. Simply evaluating likelihood at a residual minimizer likewise
does not produce the usual maximized-likelihood criterion.

Here D_m=1+|S| is a sparsity index. It is not generally the number q_S of
identifiable likelihood parameters. Under genuinely fixed omega and a regular,
locally full-rank covariance parametrization, q_S=p+|S|, including free diagonal
entries. Jointly fitting omega requires checking identifiable model dimension;
one cannot automatically add one in a redundant parametrization. An omega
estimated from the same data and then held fixed also requires care if claiming
the textbook information-criterion derivation. Use identifiable covariance
coordinates or study the plug-in criterion as an explicitly specified heuristic.

For p=4 there are only 64 upper-triangle masks, including 20 three-edge masks.
An exhaustive mask benchmark is practical. Verify numerical fitting using an
independent optimizer and multiple starts; call it an exhaustive numerical
benchmark unless within-mask global optimization is certified.

## Useful comparisons in priority order

1. Known-size saved candidate and candidate-list Hamming oracle, already
   available without refitting. These diagnose the cost of selecting size.
2. Cross-validation/held-out Gaussian likelihood selecting supports fitted by
   the proposed search. Refit the entire search, omega reference, and tuning
   inside training folds. This compares selection rules on the same estimator.
3. Exhaustive model-specific likelihood + BIC, with AIC as a secondary
   prediction-oriented comparison. Consider model-space penalties at larger p,
   without transplanting precision-graph EBIC formulas or consistency theorems.
4. Off-diagonal l1-penalized discrete Lyapunov residual fitting, and/or penalized
   Gaussian likelihood in the same discrete model, tuned without truth and
   refitted on selected masks. These are model-specific adaptations of standard
   sparse selection ideas, not established discrete-model consistency results.
5. Correlation thresholding as an inexpensive basic comparator, plus a
   stability-selection wrapper if repeated-sample stability is part of the claim.

Graphical lasso estimates support of Sigma^{-1}, not support of Lambda.
Cross-sectional regression Lasso and static DAG discovery likewise target
different structures. They can be secondary applied baselines with explicit
target differences. VAR regression requires time-lagged observations, absent
from this simulation. Give every primary method the same information, including
the upper-order restriction. Evaluate unknown-order recovery separately.

## Oracle definitions

Let E be the eligible edges, k*=|S*|, and Theta_S the stable parameter set with
off-diagonal entries zero outside S. Write

    Q_N(Lambda,omega) = ||Sigma_hat - Lambda.T Sigma_hat Lambda - omega I||_F^2.

For a matched known-size oracle, compute omega_hat by the same rule as the
proposed estimator and minimize Q_N(Lambda,omega_hat) over S subset E with
|S|=k* and Lambda in Theta_S. This knows sparsity but must infer which edges.
The saved D_m=4 candidate approximates that reference using the existing fitter.

A support oracle minimizes the same fitting loss in Theta_{S*}; it knows which
edges exist. Its support recovery is trivial, so compare covariance/KL risk or
identifiable parameter error, not its support accuracy. A known-omega oracle
is a separate benchmark: supplying omega*=1 adds information beyond support.

A standard risk oracle is S_or = argmin_S E_theta* L(theta*, theta_hat_S),
where the candidate estimation rule, deterministic support family, and loss L
are fixed beforehand. It may favor a smaller model than S* in finite samples.
An oracle inequality would bound the proposed estimator's risk by a constant
times this best candidate risk plus a remainder. Such a bound does not by itself
prove exact support recovery or minimax optimality.

For support loss, genuine minimax optimality refers to

    inf_delta sup_theta in Theta E_theta |delta(X) symmetric_difference S(theta)|.

Theta must specify sparsity, identifiable structure/order, stability margin,
noise/covariance bounds, and minimum distinguishable signal. Establishing
optimality requires a lower bound and a matching achievable upper bound.
Known-size estimation alone supplies neither.

## Identifiability and calibration

For unrestricted Lambda, a stationary covariance alone does not identify an
arbitrary transition matrix. For SPD Sigma and 0<omega<lambda_min(Sigma),

    Lambda = Sigma^{-1/2} U (Sigma-omega I)^{1/2}, for orthogonal U,

always solves the same equation. Structure and sign restrictions must justify
the target; exact parameters, exact support, and maximal-class recovery are
different claims. Equality of maximal-class families should not be called
equality of statistical covariance models without a theorem proving that link.

The current reference omega_hat=.93*lambda_min(Sigma_hat) converges to about
.9360864 for this truth, whereas omega*=1. It is a fitting reference, not a
consistent estimate of the generating noise in this scenario. Match this rule
across residual-method benchmarks; evaluate true-noise oracles separately.

For theory, first seek oracle risk control or consistency within an identifiable
restricted model. Exact support recovery additionally requires separation of
incorrect sparse models from the true covariance model, sufficient signal,
complexity control for overfitting, and bounded numerical error. If using the
plug-in omega, formulate the population limit with its actual limiting value.

The present bootstrap refits two selected masks without reconstructing the
support search or plateau screening, and holds the observed omega reference
fixed. A nominal alpha=.05 is not by itself an established full-procedure
false-positive guarantee. Validate calibration by simulating the entire
procedure under specified nulls or develop a justified selection-aware scheme.

## Research sources

- Young, Yeung and Raftery (2019), [Identifying dynamical time series model
  parameters from equilibrium samples](https://journals.sagepub.com/doi/full/10.1177/1471082X18776577).
  Direct discrete-VAR equilibrium precedent; likelihood fitting, identifiability,
  and BIC discussion. Their noise model and restrictions differ, so their
  parameter count and theorems must not be copied without checking assumptions.
- Varando and Hansen (2020), [Graphical continuous Lyapunov models](https://proceedings.mlr.press/v124/varando20a.html).
  Related continuous-time work comparing likelihood/Frobenius fitting,
  l1 regularization, and simpler graph-learning baselines. Its continuous
  Lyapunov estimator is not directly interchangeable with this discrete model.
- Olesen Recke et al. (2026), [Identifiability in Graphical Discrete Lyapunov
  Models](https://arxiv.org/abs/2601.21818). Main positive identifiability results
  use non-Gaussian higher cumulants and do not automatically apply here.
- Konishi and Kitagawa (1996), [Generalised information criteria in model
  selection](https://bemlar.ism.ac.jp/zhuang/Refs/Refs/kitagawa1996biometrika.pdf).
  Relevant if deriving a criterion specifically for a non-MLE fitting rule.
- Barron, Birge and Massart (1999), [Risk bounds for model selection via
  penalization](https://www.stat.yale.edu/~arb4/publications_files/BarronBirgeMassart.pdf).
  General risk-oracle framework, not a ready-made theorem for this objective.
- Shah and Samworth (2013), [Variable selection with error control](https://arxiv.org/abs/1105.5578).
  Stability-selection framework; its controlled quantities should not be
  conflated with arbitrary directed-support false discovery guarantees.

Reproduce diagnostics from the repository root with
`python tmp/support_recovery_benchmark_review/analyze.py`.
`trials.csv` and `summary.json` contain all calculated comparisons.
