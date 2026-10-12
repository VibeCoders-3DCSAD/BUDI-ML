# Service Documentation

This directory contains documents for FastAPI service modules. Each module uses
chronological `vN` directories. Create the next numbered directory only when
the service contract, methodology, or release evidence changes; retain older
directories unchanged. Each document's metadata supplies its publication date.

| Module | Endpoint | Available versions |
| :--- | :--- | :--- |
| `classification/` | `/api/v1/classification/v2` | `v2/` |
| `iqr/` | `/api/v1/anomaly/detect` | `v1/`, `v2/` |
| `recommend/` | `/api/v1/budget/recommend` | `v2/`, `v3/`, `v4/` |
| `forecast/` | `/api/v1/forecast/predict` | `v1/`, `v2/`, `v3/` |

The version number records the chronological service-document generation. It
does not imply that every model family shares the same version number.
