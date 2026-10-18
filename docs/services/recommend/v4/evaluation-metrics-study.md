# Evaluation Metrics for the LP Budget Allocation System

Oct 16, 2026 · Charles Nathaniel Togle

## Overview

We evaluate the recommender with four metrics: one checks that plans are valid, one checks how well Essentials are funded, and two check efficiency and priority order. Each is computed per household plan and summarized across the FIES 2023 simulation, overall and by income shock and rule setting.

Notation used in every formula:

| Symbol | Meaning |
| --- | --- |
| $B$ | Available funds of the household for the month |
| $x_i$ | Amount the LP allocates to category $i$ |
| $T_i$ | Target for category $i$ from the SARIMA forecast |
| $m_i$ | Minimum (floor) for category $i$; 0 if FLEXIBLE |
| $M_i$ | Maximum (ceiling) for category $i$ |
| $E$ | The 8 Essential categories |
| $N$ | The 4 Non-Essential categories |

Rates can also be reported survey-weighted: replace each count with the sum of the households' FIES weights.

## 1. Constraint Satisfaction Rate

**Purpose.** Shows that every plan the solver returns is valid: within budget and within every user rule.

**Source.** Machfud and Martianto (2023) evaluated their LP food plans by the share of requirements met (90–110% of the RDA for energy and protein) at a cost below actual spending.

**How it was used in the study.** Machfud and Martianto (2023) built 4 LP models on Susenas 2021 data for 5,890 Jakarta households and checked each against energy, protein and cost limits.

**What we are using.** We count a plan as valid only if it passes every hard rule of our model, then report the share of feasible plans that are valid. It should be 100%; anything less signals a solver or modeling bug.

$$
CSR = \frac{V}{P} \times 100\%
$$

where:

- $V$ = number of plans that break no rule
- $P$ = total number of feasible plans

A plan breaks no rule when:

- the total allocated does not exceed the available funds: $\sum_i x_i \le B$
- every category gets at least its minimum and at most its maximum: $m_i \le x_i \le M_i$
- every FIXED category gets exactly its fixed amount

## 2. Essential Coverage

**Purpose.** Measures how much of the Essential targets the plan actually funds, which is the main promise of the system.

**Source.** Alem et al. (2022) defined effectiveness as the fraction of total demand served in a scenario. Mostajabdaveh et al. (2025) used its complement, the unsatisfied demand ratio $1 - v_i/d_i$, where $v_i$ is the amount delivered and $d_i$ the demand.

**How it was used in the study.** Alem et al. (2022) maximized coverage times equity in a relief location-allocation model and tested it on floods and landslides in Rio de Janeiro state, Brazil. Mostajabdaveh et al. (2025) minimized average unsatisfied demand plus a Gini term, on real data from the Van earthquake and Istanbul's Kartal district. They found that putting coverage first, ahead of fairness, works well when time limits are very loose or very tight.

**What we are using.** The share of the total Essential target that the plan funds, with each category capped at its target so overfunding one category cannot hide a shortfall in another. 100% means every Essential is fully funded.

$$
EC = \frac{\sum_{i \in E} \min(x_i,\, T_i)}{\sum_{i \in E} T_i} \times 100\%
$$

where:

- $E$ = the 8 Essential categories
- $x_i$ = amount allocated to Essential category $i$
- $T_i$ = SARIMA target for Essential category $i$
- $\min(x_i, T_i)$ = the funded part of the target, so anything above the target is not counted

For per-category reporting, use the attainment ratio $c_i = \min(x_i, T_i) / T_i$ for each Essential.

## 3. Idle Funds Rate

**Purpose.** Catches money the plan leaves unallocated even though some category is still below its target, which would make a valid plan wasteful.

**Source.** Banerjee et al. (2023) note that priority rules alone ignore efficiency: allocating nothing at all satisfies every priority rule, so they add Pareto efficiency as a separate requirement. They measure efficiency loss as the largest possible allocation minus what was actually allocated.

**How it was used in the study.** Banerjee et al. (2023) studied priority-based rationing with category quotas, motivated by cases like vaccine and school-seat allocation. They proved an online policy keeps efficiency loss plus priority violations bounded, independent of the number of arriving agents.

**What we are using.** The share of the budget left unallocated, counted only when at least one category is still below its target. If every category has reached its target, leftover money is surplus and is not counted.

$$
IFR = \frac{U}{B} \times 100\%, \qquad
U =
\begin{cases}
B - \sum_i x_i & \text{if some } x_i < T_i \\
0 & \text{if every } x_i = T_i
\end{cases}
$$

where:

- $U$ = idle funds: money left over while some category is still below its target
- $B$ = available funds of the household
- $x_i$ = amount allocated to category $i$ (all 12 categories)
- $T_i$ = target for category $i$

It should be 0% for every feasible plan; anything above 0 means the LP stopped short.

## 4. Priority Validation Rate

**Purpose.** Confirms the core design rule: no Non-Essential gets money beyond its minimum while an Essential is still short of its target.

**Source.** Banerjee et al. (2023) define a priority-respecting allocation: a category may give to an agent only after all higher-priority agents are fully served. They count violations as priority loss, the number of unserved agents who were passed over for lower-priority ones. Our metric adapts that count to categories instead of agents.

**How it was used in the study.** Banerjee et al. (2023) used priority loss alongside efficiency loss to evaluate online allocation policies. They showed that forbidding all priority violations costs efficiency that grows with the number of agents, while their policy keeps the sum of both losses bounded.

**What we are using.** For each plan we count priority violations, then report the share of plans with none. It should be 100%; any plan below that breaks the Essentials-first rule.

$$
PVR = \frac{H_0}{P} \times 100\%
$$

where:

- $H_0$ = number of feasible plans with zero priority violations
- $P$ = total number of feasible plans

A plan has a priority violation when both of these hold:

- some Non-Essential category $j$ gets more than its minimum: $x_j > m_j$
- some Essential category $i$ is below its target and could still receive more: $x_i < T_i$ and $x_i < M_i$

For a FLEXIBLE Non-Essential with no floor ($m_j = 0$), any amount above 0 counts as a violation, but only while some Essential is still below its target.

An Essential held back by its own ceiling ($x_i = M_i$) is not a violation, since the plan cannot fund it further.

## References

Alem, D., Caunhye, A. M., & Moreno, A. (2022). Revisiting Gini for equitable humanitarian logistics. *Socio-Economic Planning Sciences, 82*, Article 101312. https://arxiv.org/abs/2106.12686

Banerjee, S., Eichhorn, M., & Kempe, D. (2023). *Allocating with priorities and quotas: Algorithms, complexity, and dynamics* (arXiv:2204.13019) [Preprint]. arXiv. https://arxiv.org/abs/2204.13019


Machfud, E. F. K., & Martianto, D. (2023). Optimalisasi konsumsi pangan pada rumah tangga berpendapatan rendah di DKI Jakarta [Optimization of food consumption among low income households in DKI Jakarta]. *Jurnal Ilmu Gizi dan Dietetik, 2*(1), 37–44. https://doi.org/10.25182/jigd.2023.2.1.37-44

Mostajabdaveh, M., Salman, F. S., & Gutjahr, W. J. (2025). *A branch-and-price algorithm for fast and equitable last-mile relief aid distribution* (arXiv:2512.19882) [Preprint]. arXiv. https://arxiv.org/abs/2512.19882



> Before submission, look for published versions of the two arXiv preprints.
