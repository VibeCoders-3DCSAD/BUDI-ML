# Spending Forecaster V1

> **What it does:** Projects a user's next-month category spend from their
> completed category history and the national HFCE category trend.

## API Contract

`POST /api/v1/forecast/predict` accepts a signed request containing a user ID,
historical transactions, and one or more supported categories. It returns one
forecast per category for the next calendar month, using the HFCE quarter
containing that month.

```json
{
  "user_id": "user-123",
  "categories": ["food", "transport"],
  "historical_transactions": [
    {
      "date": "2026-01-10",
      "amount": 1200,
      "category": "food",
      "transaction_type": "expense"
    }
  ]
}
```

The previous monthly, weekly, yearly, and batch forecast API is removed.

## Forecast Method

For each requested category, the service:

1. Excludes transactions after the current date and the in-progress calendar month.
2. Calculates a recency-weighted average (weights 1, 2, 3, ...) of the user's monthly category totals for the last 12 completed months.
3. Gets the HFCE category projection for the quarter containing the next calendar month.
4. Divides that projection by the HFCE value of each baseline month's quarter, weighted with the same recency weights. Quarters after the artifact's last observation use the model's projection.
5. Multiplies the user's baseline by that HFCE trend multiplier.

```text
forecast_php = weighted_average(user_month_totals, w)
               * (hfce[next_month_quarter] / weighted_average(hfce[quarter_of(user_months)], w))
```

Matching the denominator to the user's own baseline months means the multiplier
scales the user's recent level to the target quarter. A September-only baseline,
for example, is scaled by `hfce[Q4] / hfce[Q3]`.

The response reports the user's baseline, the HFCE multiplier, and the national
HFCE forecast amount so the result remains explainable.

## Requirements And Limits

- A request must include at least one expense transaction.
- Each requested category needs expense history in at least one completed month within the last 12 months.
- A category whose latest completed-month expense is 6 or more months old is not forecast.
- The current month is excluded because it is incomplete.
- HFCE is a population-level category trend, not an individual household forecast.
- Sparse personal history results in a less representative baseline. The API does
  not fabricate a cold-start category value.

## Related Artifacts

- `models/hfce_sarima_v4/` contains one quarterly SARIMA artifact per supported category.
- `models/hfce_sarima_v4/evaluation_report.md` records the category-specific model selection, chronological validation design, and held-out results.
- `training/scripts/modeling/train_hfce_sarima_v4.py` trains and evaluates those artifacts using the reproducible quarterly search.
- `docs/services/forecast/v2/methodology.md` documents the retired pooled monthly SARIMA research artifact.
