# Unusual Spending Alerts V2 — IQR Methodology

```json
{
  "document-type": "system-spec",
  "version": "2.3.0",
  "date": "2026.10.11",
  "authors": ["Group 4, III-DCSAD"]
}
```

> **Purpose:** This specification defines the final V2 methodology for detecting statistically unusual expense transactions from each user's own prior history. The production algorithm is the personalized transaction-level IQR rule. It does not use a trained anomaly-classification model, a forecast-residual detector, a global anomaly baseline, synthetic transactions, or another user's financial data for production anomaly decisions.

---

## Methodology Overview

V2 uses a Personalized Transaction-Level IQR Detector as its final production algorithm:

```text
Earlier expense transactions for one user and category
                         |
                Transaction IQR
                         |
                  Anomaly alert
```

The system identifies statistical deviations from a user's historical behaviour. An alert does not mean that a transaction is fraudulent, irresponsible, or unauthorized.

---

## Production Data Boundary

Every production anomaly decision is based only on the current user's chronologically earlier expense transactions in the candidate transaction's category.

The ML service accepts the user identity only through `X-Odin-User-Id`; it does not accept a body
`user_id`. The authenticated Odin API gateway must retrieve and scope the transaction history to
that identity, set `X-Odin-History-Source: odin_api`, and HMAC-SHA256 sign
`odin_api:<user_id>:<raw request body>` into `X-Odin-History-Signature` with the configured
`ODIN_TRUSTED_HISTORY_SHARED_SECRET`. The ML service rejects unsigned or invalid payloads and is
unavailable for anomaly scoring when that secret is not configured. Direct public access to these
headers is not a trusted deployment boundary. The same signed-gateway boundary protects the PFP,
forecast, and budget inference endpoints; health, readiness, and metrics remain unsigned for
infrastructure probes.

`/api/v1/anomaly/detect/batch` is a single-user convenience endpoint: every request in the batch
is covered by the same signed identity and history source. A gateway processing several users must
send separately signed calls for each user.

| Input | Permitted in production anomaly decision | Notes |
| :--- | :---: | :--- |
| Current user's prior transactions | Yes | The personal transaction-IQR baseline. |
| Other users' transactions | No | No population-wide anomaly reference is used. |
| Synthetic transactions | No | They must not form a production personal baseline. |
| Precomputed anomaly-model thresholds | No | Boundaries are calculated from the user history. |

Evaluation data remains separate from production anomaly data. It must not become a substitute for the user's personal anomaly baseline.

### Definition and Scope of Anomaly Labels

An anomaly is an observation that does not conform to expected behaviour. In V2, a statistical anomaly is an individual expense amount outside its personal, category-specific historical bounds. These flags are detector outputs, not independently established fraud, financial-harm, or behavioural labels.

Category membership alone does not make a transaction anomalous. A luxury, gambling, or investment category is not automatically anomalous, and a newly used category without sufficient history is an insufficient-baseline case rather than an established outlier. A large legitimate expense may also be statistically unusual without being harmful.

For evaluation, an expected positive label means that a specified controlled deviation was introduced. It does not establish fraud or financial harm. Expected labels must be independently specified and must not be derived from the IQR fences under evaluation.

---

## Cold Start Policy

Anomaly alerts are disabled for a user-category pair until it has both sufficient usable expense-transaction history and history duration for stable quartile estimates:

```text
n_(u,c) < 6 OR history_span_(u,c) < 3 calendar months
    -> anomaly alerts disabled for category c
```

Where `n_(u,c)` is the usable prior expense-observation count for user `u` and category `c`. `history_span_(u,c)` is the elapsed period from the earliest eligible baseline transaction to the candidate transaction. The initial production requirement is at least six prior same-category expenses spanning at least three calendar months. This requirement must be revalidated experimentally before adjustment.

For this rule, three calendar months means the candidate's `(year * 12 + month)` is at least
three greater than the earliest earlier baseline expense's equivalent value. ISO date-only values
mean midnight UTC; timestamp values must carry a timezone offset and are normalized to UTC.

When history is insufficient, the service must return an explicit insufficient-history state and must not manufacture an alert, a normal result, or a population-based fallback.

---

## Transaction-Level IQR

### Purpose

This detector answers:

> Is this transaction unusual compared with this user's earlier transactions in the same category?

Represent an expense transaction as:

```text
T_i = (a_i, c_i, t_i, id_i)
```

Where `a_i` is amount, `c_i` is category, `t_i` is timestamp, and `id_i` is transaction ID. Transactions are ordered by ascending `(t_i, id_i)`; `id_i` is the deterministic tie-breaker when timestamps match.

### Personal Category Baseline

For user `u` and category `c`, the historical amount set is:

```text
A_(u,c) = {a_1, a_2, ..., a_n}
```

The detector uses category-specific personal IQRs, `IQR_(u,c)`, rather than comparing categories with materially different expected scales, such as housing and transport. The baseline must contain only valid expense transactions ordered before the candidate transaction. Income, transfers, refunds, reversals, and non-positive amounts are excluded. Categories must be normalized to Odin's canonical category identifiers before baseline selection.

### Log Transformation

Because transaction amounts may be positively skewed, the detector transforms each amount before computing quartiles:

```text
x_i = ln(1 + a_i)
```

The transformed personal category history is:

```text
X_(u,c) = {x_1, x_2, ..., x_n}
```

### IQR Boundaries

For each eligible user-category history:

```text
Q1_(u,c) = P25(X_(u,c))
Q3_(u,c) = P75(X_(u,c))
IQR_(u,c) = Q3_(u,c) - Q1_(u,c)

L_(u,c) = Q1_(u,c) - 1.5 * IQR_(u,c)
U_(u,c) = Q3_(u,c) + 1.5 * IQR_(u,c)
```

Quartiles use NumPy's `linear` interpolation method.

The candidate transaction is anomalous when:

```text
A_i_txn = 1 when x_i < L_(u,c) or x_i > U_(u,c)
A_i_txn = 0 otherwise
```

### Severity

The detector may calculate a continuous severity for prioritization and explanation:

```text
D_i_txn = max(L_(u,c) - x_i, 0, x_i - U_(u,c)) / (IQR_(u,c) + epsilon)
```

`D_i_txn = 0` means the transaction is within the personal historical range. Larger values indicate increasing distance beyond the applicable IQR fence.

When `IQR_(u,c) = 0`, the detector must return an insufficient-variability state rather than treating every non-identical amount as anomalous. Severity is unavailable in this state.

After scoring, every valid expense transaction, including a flagged transaction, is added to its user-category baseline. This permits the baseline to adapt when a user's spending pattern changes over time.

## Alert Language

Alerts must describe statistical deviation rather than motive or blame.

| Detector | Recommended language |
| :--- | :--- |
| Transaction-level IQR | "This transaction is outside your usual spending range for this category." |

The system must not claim that a transaction is fraudulent, that a user is spending irresponsibly, that a purchase is impulsive, or that money was stolen.

---

## API Response Contract

Each evaluated transaction must return one of the following states:

| State | Meaning |
| :--- | :--- |
| `ANOMALOUS` | The transaction is outside its eligible personal category IQR fence. |
| `NORMAL` | The transaction is within its eligible personal category IQR fence. |
| `INSUFFICIENT_HISTORY` | The category has fewer than six prior valid expenses or less than three calendar months of prior history. |
| `INSUFFICIENT_VARIABILITY` | The eligible category baseline has zero IQR. |

Every response must include the canonical category, usable-history count, state, and amount-scale lower and upper bounds when bounds are available. Amount-scale bounds are calculated as `exp(L_(u,c)) - 1` and `exp(U_(u,c)) - 1`. Every `ANOMALOUS` or `NORMAL` response must include severity; severity is `0` for `NORMAL` and is unavailable for insufficient-history or insufficient-variability states.

Requests require globally unique, non-null `transaction_id` values and canonical lowercase
category identifiers (`[a-z][a-z0-9_]*`). The response has `ruleset_version`
`personal-transaction-iqr-v2.3.0`; lower bound, upper bound, and severity are all `null` for
both insufficient states. A candidate strictly equal to either fence is `NORMAL`.

---

## Evaluation and Dataset Separation

Maintain separate datasets and purposes:

| Dataset | Purpose |
| :--- | :--- |
| `anomaly_test` | Evaluate the transaction-IQR rule with controlled deviations and known labels. |
| Production user history | Build each user's live category-specific transaction-IQR baseline. |

Evaluation must be chronological:

```text
Historical observations
|
+-- baseline: estimate initial category-specific transaction IQRs
`-- test: evaluate transaction anomaly detection
```

For each test transaction, calculate bounds only from its earlier eligible history. The candidate transaction must not affect its own bounds; after scoring, every valid expense transaction is added to the baseline.

### Transaction-Level Test Injection

The transaction-level evaluation dataset must use eligible held-out synthetic transactions designated as normal by the generator. Split the data chronologically before injection, define all detector settings before scoring, and calculate each candidate's category-specific bounds from only its permitted earlier observations. This synthetic evaluation does not establish performance on naturally occurring anomalous spending.

For selected eligible test transactions, draw `k ~ Uniform(3, 5)` and apply:

```text
modified_amount = original_amount * k
expected_label = 1
injection_type = amount_spike
```

Retain unchanged synthetic transactions designated as normal by the generator with `expected_label = 0`. Preserve provenance for every selected record: `source_id`, `user_id`, `category`, `timestamp`, `original_amount`, `modified_amount`, `factor`, `injection_type`, and `expected_label`.

The V2 amount-based transaction-IQR evaluation does not treat category mismatch, new merchant, or frequency-change modifications as positive labels. These scenarios require separate independently defined detectors and labels.

Evaluation may report separate results at experimental 3%, 5%, 10%, and 25% injected-positive proportions after defining the eligible observations and sampling unit for each rate. These rates are comparison scenarios, not estimates of real anomaly prevalence.

### Evaluation Outputs and Metrics

Evaluate the transaction-IQR output using exactly four classification metrics: Accuracy, Precision, Recall, and F1-score.

| Detector output | Evaluation unit | Positive prediction |
| :--- | :--- | :--- |
| Transaction IQR | Individual transaction | The transaction falls outside its personal category IQR fence. |

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 * TP / (2 * TP + FP + FN)
```

Report the evaluated unit, positive-label definition, sample size, and injected-positive proportion with each metric set. Precision, Recall, and F1-score are primary because anomalous cases are uncommon; Accuracy is secondary. When a metric denominator is zero, report the metric as undefined or not estimable and state the reason rather than reporting a perfect score.

Implementation tests must separately verify expense filtering, category scoping, timestamp and transaction-ID ordering, NumPy linear quartiles, log transformation, lower- and upper-fence equality, zero-IQR handling, baseline updates after scoring, and chronological leakage prevention. These checks establish mathematical correctness; the four classification metrics measure agreement with independently specified labels.

---

## Estimation Clarification

IQR does not train like an SVM, random forest, neural network, or global anomaly classifier. It statistically estimates quartiles and Tukey fences from a user's own eligible history:

```text
user history -> Q1, Q3, IQR, lower fence, upper fence -> alert decision
```

This per-user boundary estimation is sometimes informally called IQR "training," but it is not machine-learning model training. The system does not save global IQR bounds and apply them to unrelated users.

---

## Validation Limits

The following decisions require empirical validation before production use:

1. `N_min` is category-specific. Candidate counts such as 6, 9, 12, 18, and 24 prior transactions are candidates, not established sufficiency.
2. Real individual-user transaction history is required to establish a production personal baseline; synthetic data cannot replace it.
3. Controlled 3-5x amount-spike interventions are injected only into the held-out `anomaly_test` dataset. IQR has no anomaly-training or forecasting-training dataset, and injected records never form a production personal baseline.
4. Controlled interventions evaluate the declared simulated scenarios only. They do not establish performance on naturally occurring anomalous spending among Filipino users.
5. The 3-5x amount multiplier and any selected test proportions are experimental design choices. They do not estimate real anomaly prevalence.
6. Neither category membership nor an IQR flag establishes fraud, irresponsible behaviour, or financial harm.

---

## Final Methodology Label

**Personalized Expenditure Anomaly Detection using Transaction-Level IQR**

This methodology identifies individual transaction deviations using only each user's chronologically earlier expense transactions in the same category, while preserving the distinction between anomaly-evaluation data and actual production user history.

The IQR decision rule is deterministic: the same eligible transaction history and candidate transaction always produce the same result. It is a statistical outlier detector, not a trained machine-learning model.

---

## References

Azamuke, D., Katarahweire, M., & Bainomugisha, E. (2025). A labeled synthetic mobile money transaction dataset. *Data in Brief, 60*, Article 111534. https://doi.org/10.1016/j.dib.2025.111534

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys, 41*(3), Article 15. https://doi.org/10.1145/1541880.1541882

Fisch, A. T. M., Eckley, I. A., & Fearnhead, P. (2022). A linear time method for the detection of collective and point anomalies. *Statistical Analysis and Data Mining: The ASA Data Science Journal*. https://doi.org/10.1002/sam.11586

Mashiko, S., Kawamata, Y., Nakayama, T., Sakurai, T., & Okada, Y. (2025). Anomaly detection in double-entry bookkeeping data by federated learning system with non-model sharing approach. *Scientific Reports, 15*, Article 42208. https://doi.org/10.1038/s41598-025-26120-y

National Institute of Standards and Technology. (n.d.). *7.1.6. What are outliers in the data?* NIST/SEMATECH e-Handbook of Statistical Methods. https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm

scikit-learn developers. (n.d.). *Precision, recall, F-measure and support*. scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
