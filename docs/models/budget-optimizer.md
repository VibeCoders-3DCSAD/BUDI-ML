# Budget Optimizer — Plain-Language Explanation

> **What it does:** Takes a user's available money and distributes it across spending categories in the best possible way, respecting their constraints and preferences.

---

## In one sentence

The Budget Optimizer is a mathematical planner that figures out how to split a user's money across categories like food, transport, bills, and savings — doing the best job possible while following the rules the user sets.

---

## Why it exists

Most people know roughly how much they earn and what they spend on, but they struggle with the *allocation problem*: given ₱30,000 this month, how much should go to food vs. transport vs. savings vs. debt repayment? The optimizer solves this by treating it as a formal mathematical problem — finding the best possible split given real constraints.

---

## Key concept: This is NOT a learned model

Unlike the other three models, the Budget Optimizer has **no training phase** and **no machine learning**. It's a **deterministic mathematical optimization** — a linear program solved by `scipy.optimize.linprog` (using the HiGHS solver). The same inputs always produce the same outputs. This makes it fast, transparent, and fully explainable.

---

## Input — What goes in

The optimizer receives a structured request with these components:

### 1. Available funds

The total money available for allocation this period (e.g., ₱30,000 for this month). This is the user's income minus any already-committed amounts.

### 2. Categories with restrictions

Each spending category (food, transport, bills, savings, etc.) has:

| Field | What it means |
|---|---|
| **Restriction level** | One of three levels (see below). |
| **Floor** | Minimum amount that MUST go to this category. |
| **Ceiling** | Maximum amount that CAN go to this category. |
| **Current spend** | Tracking context only. It does not change a configured allocation rule. |
| **Priority weight** (0–1) | How important is it to the user that this category gets its preferred amount? |

### 3. Target ratios

The user's *preferred* spending proportions — e.g., "I want 40% on food, 20% on transport, 15% on savings." These are goals, not hard rules; the optimizer tries to get as close as possible.

### 4. Optional context

- Transaction history (for context).
- Forecast data (so the optimizer can plan ahead).
- A flag for whether to include reasoning in the response.

---

## The three restriction levels — What they mean

| Level | Rule | Example |
|---|---|---|
| **FIXED** | Exact configured amount. Floor and ceiling must be equal. | Rent: exactly ₱8,000. |
| **MINIMUM** | Non-reducible configured floor. Can receive more within its ceiling. | Food: at least ₱5,000. |
| **FLEXIBLE** | Zero or configured floor, up to its ceiling. | Entertainment: between ₱0 and ₱3,000. |

---

## Process — How it thinks

### 1. Problem setup

The optimizer frames budget allocation as a **linear program** — a mathematical problem with:

- **Decision variables**: One number per category (how much to allocate), plus one "deviation" variable per category (how far the allocation is from the target ratio).
- **Objective**: Minimize the total weighted deviation from the user's target ratios. Categories with higher `priority_weight` get penalized more for missing their target.
- **Constraints** (the rules it must follow):

| Constraint | Mathematical form | Plain meaning |
|---|---|---|
| **Budget sum** | Σ allocations = available funds | Every peso must be allocated somewhere. |
| **FIXED** | Allocation = floor = ceiling | Fixed categories get their configured exact amount. |
| **MINIMUM** | Allocation ≥ floor | Minimum categories cannot be reduced below their configured floor. |
| **FLEXIBLE** | floor ≤ allocation ≤ ceiling | Flexible categories stay within their bounds and may receive zero. |
| **Deviation** | deviation ≥ allocation − target AND deviation ≥ target − allocation | Deviation captures the absolute gap between what was allocated and what was desired. |

### 2. Solving

The problem is passed to `scipy.optimize.linprog` with the **HiGHS** solver — a fast, production-grade linear programming solver. It finds the allocation that minimizes total deviation while satisfying all constraints.

### 3. Feasibility check

Not all requests can be satisfied. The optimizer reports one of two states:

| State | What it means |
|---|---|
| **FEASIBLE** | All constraints can be met simultaneously. The solution is optimal. |
| **INFEASIBLE** | No valid allocation satisfies every hard constraint. The optimizer never relaxes Fixed or Minimum rules. It returns the required total, available funds, shortfall, and affected categories so the user can decide what to change. |

---

## Output — What comes out

The API returns a structured recommendation:

| Field | Description |
|---|---|
| **Allocations** (list) | For each category: the recommended amount, floor, ceiling, target, and how much deviation was accepted. |
| **Budget utilization** (0–1) | What fraction of available funds is allocated (ideally 1.0). |
| **Constraint satisfaction** (0–1) | What fraction of constraints are fully met (1.0 = all constraints satisfied). |
| **Feasibility status** | `FEASIBLE` or `INFEASIBLE`. |
| **Per-category explanations** | A plain-language reason for each allocation: "Food received ₱12,000 (target was ₱12,000 — exact match)" or "Transport was reduced to ₱4,500 to stay within available funds." |
| **Overall explanation** | A summary of the allocation strategy. |

### Example output

```
Status: FEASIBLE
Available funds: ₱30,000
Utilization: 100%
Constraint satisfaction: 100%

Allocations:
  Food:         ₱12,000  (target ₱12,000, deviation: 0)     ← exact match
  Transport:    ₱ 6,000  (target ₱ 6,000, deviation: 0)     ← exact match
  Bills:        ₱ 8,000  (FIXED — configured exact amount)   ← no choice
  Savings:      ₱ 3,000  (target ₱ 4,500, deviation: ₱1,500) ← reduced to fit
  Entertainment:₱ 1,000  (target ₱ 1,500, deviation: ₱  500) ← reduced to fit

Explanation: All constraints satisfied. Savings and Entertainment
were slightly reduced from their target ratios to accommodate the
locked Bills allocation, but stayed within their allowed bounds.
```

---

## How other modules use this

- **Forecasting** provides the estimated category-level spending used as the initial allocation target for Flexible categories. It can raise, but never lower, a Minimum target; Fixed categories remain exact.
- **Anomaly Detector** can flag transactions that deviate from the optimizer's plan ("You've overspent in Food by ₱2,000 relative to your budget").
- **PFP Classifier** informs initial restriction levels — an "At-Risk" user may have tighter Fixed/Minimum constraints, while a "Tolerant" user gets more Flexible categories.
- **User dashboard** displays the allocation as a visual budget card, with real-time tracking of how actual spending compares to the plan.

---

## Evaluation

The optimizer was evaluated on **600 synthetic personas** with varying income levels, obligation loads, and preference profiles:

| Metric | Value | What it means |
|---|---|---|
| Constraint Satisfaction Rate | 1.0 (100%) | When the problem is feasible, every constraint is met. |
| Budget Utilization Rate | 1.0 (100%) | Every peso of available funds is allocated. |
| Mean deviation from preferences | 0.041 (4.1%) | On average, allocations are within ~4% of the user's target ratios. |

Of the 600 personas: **541 were FEASIBLE** (90.2%), **59 were INFEASIBLE** (9.8%) — meaning about 1 in 10 synthetic users requested allocations that couldn't be fully satisfied with their available funds.

---

## Honest limitations

- **No learning** — the optimizer doesn't improve over time or learn from past allocations. It's a pure mathematical solver. Any improvements come from better inputs (more accurate forecasts, better category definitions) rather than model updates.
- **Linearity assumption** — the model assumes that spending relationships are linear (doubling the budget doubles each category equally). In reality, some categories have thresholds (you need at least ₱X for transport, but beyond that, more doesn't help).
- **No temporal awareness** — the optimizer allocates for one period at a time. It doesn't plan across months (e.g., "save more this month for a big expense next month").
- **INFEASIBLE outcomes** require user intervention — the optimizer can't create money that isn't there. When constraints conflict with available funds, the user must adjust their expectations.
