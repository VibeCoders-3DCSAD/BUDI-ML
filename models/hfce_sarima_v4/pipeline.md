
# Household Final Consumption Expenditure Forecasting Methodology

## Data Source and Forecasting Scope

The forecasting component uses quarterly Philippine **Household Final Consumption Expenditure (HFCE) by purpose** obtained from the Philippine Statistics Authority (PSA). The source series covers **2000-Q1 to 2026-Q2**, corresponding to **106 quarterly observations per expenditure category**. The PSA publishes the linked quarterly HFCE series at both current and constant 2018 prices, with the latest dataset used in this study dated **August 7, 2026** (Philippine Statistics Authority [PSA], 2026a).

The forecasting target is:

`current_price_million_php`

which represents national HFCE at **current prices in million Philippine pesos**.

The following twelve expenditure categories are modeled independently:

1.  Food and non-alcoholic beverages

2.  Alcoholic beverages and tobacco

3.  Clothing and footwear

4.  Housing, water, electricity, gas, and other fuels

5.  Furnishings, household equipment, and routine household maintenance

6.  Health

7.  Transport

8.  Communication

9.  Recreation and culture

10.  Education

11.  Restaurants and hotels

12.  Miscellaneous goods and services


A separate forecasting model is developed for each category because expenditure categories may exhibit different trends, seasonal patterns, rates of change, and autocorrelation structures.

Current-price HFCE is used because the financial planning application ultimately operates using **nominal Philippine pesos**. Therefore, the model forecasts the movement of **national nominal expenditure**, rather than real consumption volume or inflation in isolation.

The HFCE forecasting model does **not** forecast the future expenses of an individual user or household. At serving time, the national HFCE forecast is used only as a **macroeconomic trend multiplier** applied to a spending baseline derived separately from the user's historical transactions.

----------

# Time-Series Structure

The HFCE observations are quarterly and are maintained in chronological order throughout model development.

No random shuffling is performed.

Randomly partitioning time-series observations can allow future observations to influence model development or evaluation, creating temporal leakage and producing unrealistic estimates of forecasting performance. Forecast evaluation should preserve temporal ordering and ensure that only observations available before a forecast origin are used for model estimation (Hewamalage et al., 2023).

Because HFCE is observed quarterly, one annual seasonal cycle contains four observations. The seasonal period is therefore fixed at:

$$
s = 4
$$

for all seasonal forecasting models.

----------

# Dataset Partitioning

The complete dataset is divided chronologically into three periods:

Dataset

Period

Observations

Purpose

Initial Training Period

2000-Q1 to 2016-Q4

68

Initial model estimation

Rolling Validation Period

2017-Q1 to 2023-Q4

28

Candidate selection

Final Test Period

2024-Q1 to 2026-Q2

10

Final untouched evaluation

The **2024-Q1 to 2026-Q2** observations are excluded from model and hyperparameter selection.

They are accessed only after the winning model specification for each expenditure category has been selected using the rolling validation period.

This separation maintains a final set of observations that provides an independent estimate of forecasting performance.

Hewamalage et al. (2023) emphasize that forecasting evaluation should maintain strict temporal separation between model development and future evaluation data and recommend time-series cross-validation when sufficient data are available.

----------

# PSA Data Revisions

Philippine national accounts are subject to revisions as newer and more complete information becomes available.

The PSA, for example, released revised annual and quarterly National Accounts estimates for 2024 and 2025 in April 2026 under its approved national accounts revision policy (PSA, 2026b).

Accordingly, this study is considered a **pseudo-out-of-sample forecasting experiment based on a fixed historical data vintage**.

The dataset used corresponds to the PSA HFCE series available as of:

**August 7, 2026**

(PSA, 2026a).

The evaluation therefore does not claim to reproduce the exact data vintages that would have been available at every historical forecast origin.

----------

# Treatment of the COVID-19 Period

Observations during the COVID-19 disruption are retained, including:

$$
2020-Q1 \text{ through } 2021-Q4
$$

These observations represent actual movements in national household consumption and are therefore maintained as part of the historical HFCE series.

They are not deleted, interpolated, replaced, or treated as missing values solely because they represent an unusual economic period.

Forecast evaluation literature identifies structural changes and non-stationarity as important challenges in time-series forecasting and recommends that evaluation procedures expose forecasting models to such changing conditions rather than relying exclusively on stable historical periods (Hewamalage et al., 2023).

No SARIMAX intervention variables are included.

Therefore, the methodology does not attempt to:

-   isolate the causal effect of COVID-19;

-   estimate a separate pandemic intervention coefficient;

-   explicitly model a COVID pulse effect; or

-   explicitly model a post-COVID structural level shift.


Instead, candidate models are required to demonstrate acceptable out-of-sample performance despite the observed disruption.

Because the rolling validation period spans **2017-Q1 to 2023-Q4**, candidate models are evaluated across:

-   pre-COVID observations;

-   the COVID disruption; and

-   post-COVID observations.


Poor forecasting performance during these periods is treated as evidence against the candidate rather than as justification for modifying the historical data.

----------

# Forecasting Model Families

Three model families are evaluated for each expenditure category:

1.  Seasonal Naïve

2.  Exponential Smoothing

3.  Seasonal ARIMA


The purpose of including simpler models is to determine whether the additional complexity of SARIMA produces meaningful improvements in out-of-sample forecasting performance.

Recent forecast-evaluation guidance recommends comparing forecasting methods against appropriate simple benchmarks, including naïve and seasonal-naïve forecasting, because complex models should not automatically be assumed to provide better forecasts (Hewamalage et al., 2023).

Empirical comparisons have likewise demonstrated that the best-performing forecasting family can depend on the characteristics of the individual time series; SARIMA, ETS, and exponential-smoothing methods can outperform one another under different conditions (Xian et al., 2023).

----------

# Seasonal Naïve Benchmark

The **seasonal naïve model** serves as the primary simple benchmark.

For quarterly HFCE, the forecast for a particular quarter is equal to the most recently observed value from the corresponding quarter of the previous year.

The seasonal-naïve forecast can be expressed as:

$$
\hat{Y}_{t+h}=Y_{t+h-4}
$$

where:

-   $\hat{Y}_{t+h}$ is the forecast value;

-   $Y_{t+h-4}$ is the observed value from the corresponding quarter of the previous annual seasonal cycle; and

-   $4$ represents quarterly seasonality.


For example:

$$
\hat{Y}_{2025,Q2}=Y_{2024,Q2}
$$

Hewamalage et al. (2023) recommend the use of appropriate simple benchmarks such as naïve and seasonal-naïve forecasting when evaluating more complex forecasting models.

The seasonal-naïve model therefore establishes the minimum forecasting benchmark that more complex models must compete against.

----------

# Exponential Smoothing

Exponential smoothing or **ETS** is included as a statistical comparison family.

ETS represents a time series through combinations of:

-   error;

-   trend; and

-   seasonal components.


These components may differ depending on the structure detected in a particular expenditure category.

ETS is retained as a comparator because it models level, trend, and seasonality through a different statistical structure from SARIMA. Recent empirical comparisons demonstrate that neither SARIMA nor ETS should be assumed to be universally superior; the more appropriate model depends on the underlying time-series behavior (Xian et al., 2023).

ETS therefore serves as an alternative statistical forecasting method rather than merely as a secondary version of SARIMA.

----------

# Seasonal ARIMA

The primary candidate family is the **Seasonal Autoregressive Integrated Moving Average**, or SARIMA, model.

The general model is represented as:

$$
SARIMA(p,d,q)(P,D,Q)_4
$$

where:

-   $p$ = non-seasonal autoregressive order;

-   $d$ = non-seasonal differencing order;

-   $q$ = non-seasonal moving-average order;

-   $P$ = seasonal autoregressive order;

-   $D$ = seasonal differencing order;

-   $Q$ = seasonal moving-average order; and

-   $4$ = quarterly seasonal period.


The non-seasonal components describe relationships between relatively recent observations and errors, while the seasonal components describe relationships occurring across quarterly seasonal cycles.

SARIMA remains an established statistical method for seasonal univariate time-series forecasting and continues to be evaluated alongside exponential-smoothing approaches in recent forecasting research (Xian et al., 2023).

----------

# SARIMA Candidate Space

Each expenditure category receives its own SARIMA candidate search.

The following candidate orders are considered:

$$
p,q \in {0,1,2,3}
$$

$$
d \in {0,1}
$$

$$
P,Q \in {0,1,2}
$$

$$
D \in {0,1}
$$

with:

$$
s=4
$$

To restrict excessive differencing, candidates where both non-seasonal and seasonal differencing are simultaneously applied are excluded:

$$
d=1 \text{ and } D=1
$$

This constraint limits the search to models with at most one form of differencing and reduces the risk of unnecessary over-differencing in a relatively short quarterly series.

The valid differencing configurations are therefore:

$$
(d,D)\in{(0,0),(1,0),(0,1)}
$$

The order combinations produce:

$$
4\times4\times3\times3\times3=432
$$

SARIMA parameter combinations.

Because each parameter combination is evaluated using both an untransformed and logarithmically transformed target, the maximum candidate count is:

$$
432\times2=864
$$

candidates per expenditure category.

Across twelve categories, models are trained and evaluated independently rather than assuming that one SARIMA order is appropriate for all expenditure purposes.

----------

# Candidate Failure Rules

A candidate is rejected if model estimation or forecasting produces invalid results.

A SARIMA candidate is removed from consideration when it:

-   fails to converge;

-   raises an unrecoverable estimation error;

-   produces non-finite model parameters;

-   produces `NaN` forecasts;

-   produces infinite forecasts; or

-   cannot generate the required forecast horizon.


A failed candidate is not assigned an artificially poor metric value and is not allowed to participate in final candidate ranking.

----------

# Information Criteria

The methodology does **not** use:

-   Akaike Information Criterion (AIC);

-   corrected Akaike Information Criterion (AICc); or

-   Bayesian Information Criterion (BIC)


as model-selection criteria.

Model selection is based on **out-of-sample forecasting performance** generated through rolling-origin validation.

This separates the objective of the study from in-sample model fit. The selected model is the model that performs best when forecasting unseen observations according to the predefined validation procedure.

----------

# Target Transformation

Each eligible SARIMA specification is evaluated using two target representations:

1.  **Raw current-price HFCE**

2.  **Log-transformed current-price HFCE**


For the raw model:

$$
Z_t=Y_t
$$

For the logarithmic model:

$$
Z_t=\log(1+Y_t)
$$

where:

-   $Y_t$ represents the HFCE value at time $t$; and

-   $Z_t$ represents the value used during model estimation.


The `log1p` transformation is retained as a candidate transformation because expenditure series may exhibit increasing variation as their nominal levels increase.

However, the transformation is **not automatically assumed to improve forecasting accuracy**.

It is treated as part of the candidate configuration and must demonstrate better validation performance.

For log-transformed candidates, forecasts are returned to the original scale before evaluation:

$$
\hat{Y}_t=\exp(\hat{Z}_t)-1
$$

All forecast metrics are therefore calculated using the original:

**current-price million-PHP scale**

rather than the transformed scale.

----------

# Rolling-Origin Validation

Candidate selection uses **expanding-window rolling-origin validation**.

Rolling-origin validation repeatedly moves the forecast origin forward while ensuring that each candidate model is trained using only observations that occurred before the forecasted period.

Hewamalage et al. (2023) recommend time-series cross-validation when sufficient historical observations are available and emphasize explicitly preventing leakage during rolling-origin forecast evaluation.

The initial estimation window is:

$$
2000-Q1 \text{ through } 2016-Q4
$$

containing:

$$
68 \text{ quarters}
$$

The first forecast origin is therefore:

$$
2016-Q4
$$

From this origin, the model generates forecasts for:

$$
2017-Q1,\ 2017-Q2,\ 2017-Q3,\ 2017-Q4
$$

The training window is then expanded as actual observations become available.

Conceptually:

$$
2000Q1-2016Q4
\rightarrow
2017Q1-2017Q4
$$

followed by:

$$
2000Q1-2017Q1
\rightarrow
2017Q2-2018Q1
$$

followed by:

$$
2000Q1-2017Q2
\rightarrow
2017Q3-2018Q2
$$

and so forth.

Rolling forecast origins continue until forecasts reach the final validation observation:

$$
2023-Q4
$$

No observation occurring after a forecast origin is used to estimate that model.

----------

# Forecast Horizons

At each eligible validation origin, the candidate generates forecasts for:

$$
h\in{1,2,3,4}
$$

where:

-   $h=1$ = one quarter ahead;

-   $h=2$ = two quarters ahead;

-   $h=3$ = three quarters ahead; and

-   $h=4$ = four quarters ahead.


Four quarters represent one complete annual seasonal cycle.

The use of several forecasting horizons evaluates whether a model performs consistently beyond only the immediate next observation.

Multi-step forecasts are generated using the fitted model's forecasting process.

Future actual observations are **not inserted into the forecast path** for a forecast already generated at a particular origin.

Actual observations are incorporated only after the forecast origin itself moves forward.

----------

# Relationship to Serving Horizon

The HFCE model remains a **quarterly national forecasting model**.

It does not directly generate a monthly personal-expense forecast.

At deployment, the application's personal forecasting component derives a baseline from the user's own historical category spending.

The national HFCE forecast is subsequently used only as a trend-adjustment signal.

Therefore:

$$
\text{Quarterly HFCE Forecast}
\neq
\text{Personal Monthly Expense Forecast}
$$

The quarterly national model provides external macroeconomic information to a separate personal forecasting process.

One-quarter-ahead performance is particularly relevant because the nearest national HFCE forecast is the most immediate macroeconomic signal available to the personal financial planning system.

The longer $h=2$, $h=3$, and $h=4$ forecasts provide additional evidence regarding model stability.

----------

# Forecast Evaluation Metrics

Only four forecast-accuracy metrics are used:

1.  Mean Absolute Error (MAE)

2.  Symmetric Mean Absolute Percentage Error (SMAPE)

3.  Mean Directional Accuracy (MDA)

4.  Root Mean Square Error (RMSE)


Recent forecasting studies continue to use MAE, RMSE, and SMAPE jointly because the metrics provide complementary information regarding absolute error magnitude, large errors, and relative error (Cobbinah & Alnaggar, 2024).

Directional accuracy is evaluated separately because correctly predicting the direction of change represents a distinct forecasting characteristic from minimizing the numerical magnitude of forecasting error (Pincheira et al., 2026).

----------

## 1. Mean Absolute Error

Mean Absolute Error measures the average absolute difference between actual and forecast HFCE values:

$$
MAE=
\frac{1}{n}
\sum_{t=1}^{n}
|Y_t-\hat{Y}_t|
$$

where:

-   $Y_t$ = actual HFCE;

-   $\hat{Y}_t$ = forecast HFCE; and

-   $n$ = number of forecast observations.


Lower MAE indicates better forecasting performance.

MAE retains the same measurement unit as the original target and therefore represents average forecasting error in:

**million Philippine pesos**.

Because absolute errors are not squared, MAE gives each forecast error a linear contribution to the final score. It consequently provides an interpretable measure of typical forecast deviation (Cobbinah & Alnaggar, 2024).

----------

## 2. Symmetric Mean Absolute Percentage Error

Symmetric Mean Absolute Percentage Error measures forecast error relative to the magnitude of both the actual and forecast values.

It is calculated as:

$$
SMAPE=
\frac{100}{n}
\sum_{t=1}^{n}
\frac{2|Y_t-\hat{Y}_t|}
{|Y_t|+|\hat{Y}_t|}
$$

Lower SMAPE indicates better forecasting performance.

Unlike MAE and RMSE, SMAPE is scale-independent and allows the relative forecasting performance of categories with different expenditure magnitudes to be interpreted using a common percentage scale.

Recent time-series forecasting research continues to use SMAPE alongside absolute-error metrics because it provides a relative-error perspective that MAE and RMSE do not provide (Cobbinah & Alnaggar, 2024).

However, SMAPE is not interpreted independently of the absolute-error metrics.

----------

## 3. Mean Directional Accuracy

Mean Directional Accuracy measures whether the forecast correctly predicts the **direction of quarterly change**.

The direction is determined relative to the most recent actual observation available at the corresponding forecast origin.

MDA is calculated as:

\operatorname{sign}(\hat{Y}_t-Y_{t-1})
\right]
\times100
$$

where:

-   $I[\cdot]$ is an indicator function;

-   $Y_t-Y_{t-1}$ represents the actual direction of change; and

-   $\hat{Y}_t-Y_{t-1}$ represents the predicted direction of change.


Higher MDA indicates better performance.

MDA measures a different characteristic from MAE, SMAPE, and RMSE. A forecast may correctly predict whether expenditure rises or falls while still producing a large numerical forecast error.

Recent research demonstrates that directional or sign predictability should be treated separately from conventional measures of numerical forecast association and magnitude (Pincheira et al., 2026).

MDA is therefore treated as a complementary directional metric rather than a replacement for magnitude-based forecast accuracy.

----------

## 4. Root Mean Square Error

Root Mean Square Error is calculated as:

$$
RMSE=
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}
(Y_t-\hat{Y}_t)^2
}
$$

Lower RMSE indicates better forecasting performance.

RMSE retains the original unit of the HFCE target but squares forecast errors before averaging them.

Consequently, larger forecast errors contribute more strongly to RMSE than they do to MAE.

This makes RMSE useful for identifying candidates that may have acceptable average errors but occasionally produce substantially larger forecasting misses (Cobbinah & Alnaggar, 2024).

----------

# Candidate Ranking

Candidates are evaluated **independently for each HFCE category**.

The four metrics are not mathematically averaged into a composite score because they represent different properties of forecasting performance.

Candidate ranking follows the predetermined hierarchy:

1.  **Lowest MAE**

2.  **Lowest SMAPE**

3.  **Lowest RMSE**

4.  **Highest MDA**


MAE serves as the primary ranking criterion because all candidates within a category forecast the same target using the same measurement scale.

SMAPE provides the second comparison based on relative forecast error.

RMSE provides additional preference against models producing large forecast misses.

MDA is used as the final directional comparison.

This ordering prevents a candidate with high directional accuracy but large monetary forecast errors from being selected over a model that more accurately predicts the actual HFCE magnitude.

----------

# Comparison Between Forecasting Families

Selection is not restricted to SARIMA.

For every category, the final SARIMA candidate is compared with the corresponding:

-   seasonal-naïve forecast; and

-   ETS forecast.


Each forecasting family is evaluated using exactly the same:

-   validation observations;

-   forecast origins;

-   forecast horizons;

-   MAE calculation;

-   SMAPE calculation;

-   MDA calculation; and

-   RMSE calculation.


This ensures that differences in performance result from the forecasting methods rather than different evaluation procedures.

Forecasting methods should be compared against relevant and appropriately simple benchmarks before additional complexity is considered justified (Hewamalage et al., 2023).

Recent comparisons of SARIMA and exponential-smoothing approaches also demonstrate that simpler models may outperform SARIMA for particular time series, reinforcing the need for empirical model comparison rather than assuming one forecasting family is universally superior (Xian et al., 2023).

----------

# Category-Specific Model Selection

The twelve HFCE categories are **not required to share the same forecasting model**.

For example, model selection may result conceptually in:

HFCE Category

Selected Family

Food

SARIMA

Housing

ETS

Transport

SARIMA

Communication

Seasonal Naïve

The actual model family is determined entirely from the predefined validation procedure.

This category-specific approach recognizes that the twelve national expenditure series can exhibit different temporal characteristics.

The methodology therefore does not force SARIMA to be deployed when a simpler model produces better out-of-sample forecasts.

----------

# Final Model Locking

After rolling validation has been completed, the winning model specification for each category is **locked**.

For a SARIMA model, the locked configuration includes:

-   $p$;

-   $d$;

-   $q$;

-   $P$;

-   $D$;

-   $Q$;

-   seasonal period $s=4$; and

-   target transformation.


For ETS, the selected ETS specification is locked.

For seasonal naïve, the seasonal period remains fixed at four quarters.

After model locking, information from the final test period cannot be used to:

-   change the model family;

-   change a SARIMA order;

-   change the target transformation;

-   introduce a new candidate;

-   remove a candidate because of poor test performance; or

-   otherwise retune the model.


The test period is used exclusively for final evaluation.

----------

# Final Test Evaluation

The untouched final test period is:

$$
2024-Q1 \text{ through } 2026-Q2
$$

containing:

$$
10 \text{ quarters}
$$

After candidate selection, the locked specification is initially estimated using all observations available through:

$$
2023-Q4
$$

The final test follows an expanding-window procedure intended to approximate production-like updating.

The first test forecast is:

$$
2000Q1-2023Q4
\rightarrow
2024Q1
$$

After the actual 2024-Q1 observation becomes available, it may be added to the estimation data:

$$
2000Q1-2024Q1
\rightarrow
2024Q2
$$

followed by:

$$
2000Q1-2024Q2
\rightarrow
2024Q3
$$

The process continues until:

$$
2026Q2
$$

At every test origin:

-   the selected model family remains unchanged;

-   the selected SARIMA order remains unchanged;

-   the transformation remains unchanged; and

-   only the model parameters are re-estimated using newly available observations.


This preserves the distinction between **model selection** and **model updating**.

The test therefore simulates a deployment environment in which a fixed forecasting methodology may be periodically refitted as new PSA quarterly observations become available.

----------

# Test Metrics

Final test performance is reported using the same four metrics used during validation:

1.  MAE

2.  SMAPE

3.  MDA

4.  RMSE


Validation and test results are reported separately.

Test performance is **not fed back into candidate selection**.

This separation prevents the final test dataset from gradually becoming another validation dataset.

----------

# Cross-Category Reporting

Each of the twelve expenditure categories receives its own forecast-evaluation results.

A recommended reporting structure is:

Category

Model

MAE

SMAPE

MDA

RMSE

Food

—

—

—

—

—

Alcohol and Tobacco

—

—

—

—

—

Clothing and Footwear

—

—

—

—

—

Housing and Utilities

—

—

—

—

—

Furnishings and Household

—

—

—

—

—

Health

—

—

—

—

—

Transport

—

—

—

—

—

Communication

—

—

—

—

—

Recreation and Culture

—

—

—

—

—

Education

—

—

—

—

—

Restaurants and Hotels

—

—

—

—

—

Miscellaneous Goods and Services

—

—

—

—

—

Raw MAE and RMSE values are **not directly averaged across all twelve categories** because the categories differ substantially in their nominal expenditure magnitude.

For example, a category with substantially larger national expenditure may naturally produce larger million-PHP errors even when its relative forecasting performance is comparable to that of a smaller category.

Category-level performance therefore remains the primary reporting unit.

SMAPE and MDA may additionally be summarized across categories because they are not expressed directly in million pesos.

----------

# Model Release Gates

Successful candidate selection does not automatically result in artifact deployment.

Before release, the selected model for each category must satisfy basic operational validity requirements.

A model is rejected if it:

-   fails to estimate successfully;

-   fails to converge consistently;

-   produces `NaN` values;

-   produces infinite values;

-   produces non-finite forecasts;

-   produces invalid negative HFCE forecasts on the original scale; or

-   demonstrates substantial deterioration relative to the simpler benchmark during final evaluation.


A model should not be deployed merely because it belongs to the SARIMA family.

The released model for every category must remain SARIMA. Seasonal-naive and
ETS candidates are evaluation benchmarks only and are never deployed.

For the released implementation, "substantial deterioration" means final-test
MAE more than 10% higher than the strongest seasonal-naive or ETS benchmark
locked during validation. A SARIMA candidate that fails this gate is rejected;
the test period never selects a replacement model.

This approach is consistent with forecast-evaluation guidance emphasizing meaningful comparison against simple benchmarks rather than assuming greater model complexity implies greater forecasting accuracy (Hewamalage et al., 2023).

----------

# Final Full-Data Refit

After:

1.  rolling validation;

2.  model selection;

3.  model locking;

4.  untouched test evaluation; and

5.  release assessment,


the approved model specification is refitted using all available HFCE observations:

$$
2000-Q1 \text{ through } 2026-Q2
$$

This final refit allows the released model to use the maximum available historical information.

The final refit does **not** constitute another accuracy evaluation.

The forecasting accuracy reported for the study remains the performance obtained from the previously untouched:

$$
2024-Q1 \text{ through } 2026-Q2
$$

test period.

----------

# Artifact Generation

After the final full-data refit, the approved forecasting model is stored as a serialized artifact.

The saved artifact should contain, at minimum:

-   expenditure category;

-   selected forecasting family;

-   fitted model;

-   SARIMA order, when applicable;

-   seasonal order, when applicable;

-   transformation method;

-   seasonal frequency;

-   final training endpoint;

-   PSA data vintage;

-   required preprocessing metadata; and

-   model-version information.


The resulting artifact represents the model used during serving.

Conceptually:

$$
\text{PSA HFCE Data}
$$

$$
\downarrow
$$

$$
\text{Candidate Training}
$$

$$
\downarrow
$$

$$
\text{Rolling Validation}
$$

$$
\downarrow
$$

$$
\text{Model Selection}
$$

$$
\downarrow
$$

$$
\text{Locked Test Evaluation}
$$

$$
\downarrow
$$

$$
\text{Release Gate}
$$

$$
\downarrow
$$

$$
\text{Full-Data Refit}
$$

$$
\downarrow
$$

$$
\text{Saved Model Artifact}
$$

----------

# Serving-Time Use of the HFCE Forecast

The released forecasting model produces a quarterly national HFCE forecast for the corresponding expenditure category.

The forecasted national expenditure level is **not directly assigned to the user**.

Instead, the model generates a national trend multiplier.

A general representation is:

\frac{\text{Forecast HFCE}}
{\text{Reference HFCE}}
$$

where the reference HFCE corresponds to the national level used as the comparison baseline.

The user's spending forecast is derived separately from the user's historical transaction data.

Conceptually:

$$
\text{User Transactions}
\rightarrow
\text{Personal Spending Baseline}
$$

while:

$$
\text{PSA HFCE}
\rightarrow
\text{National Category Trend}
$$

The national trend may subsequently adjust the user's independently calculated baseline:

\text{Personal Baseline}
\times
\text{HFCE Trend Multiplier}
$$

The national HFCE component therefore functions as an **external macroeconomic calibration signal**, not as a household-level expenditure model.

----------

# Permitted Interpretation of the Model

The forecasting component may be described as:

> A quarterly national HFCE forecasting model that estimates nominal expenditure trends for Philippine consumption categories and supplies category-level macroeconomic trend factors to the personal financial planning system.

The system may claim that it:

-   forecasts quarterly Philippine national HFCE;

-   forecasts HFCE separately by expenditure category;

-   operates on current-price national expenditure;

-   models quarterly seasonality;

-   evaluates forecasts using chronological out-of-sample observations;

-   compares SARIMA against simpler forecasting approaches;

-   reports MAE, SMAPE, MDA, and RMSE;

-   uses national forecasts as external trend factors; and

-   refits approved models when newer national HFCE observations become available.


----------

# Prohibited Interpretation of the Model

The HFCE forecasting component must **not** claim that it:

-   predicts an individual user's spending;

-   predicts a specific household's expenditure;

-   directly forecasts a user's next-month transactions;

-   represents an individual's personal seasonality;

-   predicts household-level consumption using PSA HFCE alone;

-   isolates inflation from real consumption changes;

-   measures only price changes;

-   estimates the causal effect of COVID-19;

-   guarantees future national expenditure levels; or

-   assumes national consumption movements affect every household equally.


The distinction is:

$$
\boxed{
\text{National HFCE Forecast}
\neq
\text{Individual Household Forecast}
}
$$

----------

# Methodological Rationale

The forecasting methodology follows four principles.

First, **temporal ordering is preserved**. Random splitting is not used, and historical forecasts are produced using only observations available at the corresponding forecast origin. This reduces temporal leakage and more closely represents actual forecasting conditions (Hewamalage et al., 2023).

Second, **multiple rolling forecast origins are used for model selection**. Rather than selecting a model from a single short validation interval, the procedure evaluates performance across 2017-Q1 through 2023-Q4, including pre-COVID, COVID, and post-COVID observations. Time-series cross-validation provides a stronger basis for evaluating forecasting methods when sufficient observations are available (Hewamalage et al., 2023).

Third, **complex models are compared with simpler benchmarks**. Seasonal naïve and ETS models are retained because greater model complexity does not guarantee superior forecasting accuracy. Recent empirical studies comparing SARIMA with exponential-smoothing models demonstrate that performance depends on the properties of the specific series being forecast (Xian et al., 2023).

Fourth, **forecast performance rather than in-sample information criteria determines the selected model**. MAE, SMAPE, MDA, and RMSE evaluate complementary aspects of forecast performance. MAE and RMSE measure magnitude error, SMAPE measures relative error, and MDA evaluates the ability to predict the direction of change (Cobbinah & Alnaggar, 2024; Pincheira et al., 2026).

Given the availability of **106 quarterly observations per category**, the study therefore prioritizes established statistical forecasting approaches and empirical out-of-sample evaluation rather than introducing additional model complexity that cannot be justified by the available time-series length.

----------

# References

Cobbinah, M., & Alnaggar, A. (2024). An attention encoder-decoder RNN model with teacher forcing for predicting consumer price index. _Journal of Data, Information and Management, 6_, 65–83. [https://doi.org/10.1007/s42488-024-00114-3](https://doi.org/10.1007/s42488-024-00114-3)

Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023). Forecast evaluation for data scientists: Common pitfalls and best practices. _Data Mining and Knowledge Discovery, 37_, 788–832. [https://doi.org/10.1007/s10618-022-00894-5](https://doi.org/10.1007/s10618-022-00894-5)

Philippine Statistics Authority. (2026a). _Time series data: National accounts of the Philippines_. [https://psa.gov.ph/statistics/national-accounts/data-series](https://psa.gov.ph/statistics/national-accounts/data-series)

Philippine Statistics Authority. (2026b, April 8). _PSA releases annual revisions of the National Accounts of the Philippines for years 2024 and 2025_. [https://psa.gov.ph/statistics/national-accounts/node/1684083194](https://psa.gov.ph/statistics/national-accounts/node/1684083194)

Pincheira, P., Bentancor, A., & Reus, L. (2026). The paradox between correlations and sign predictability. _Mathematics, 14_(5), 752. [https://doi.org/10.3390/math14050752](https://doi.org/10.3390/math14050752)

Xian, X., Wang, L., Wu, X., Tang, X., Zhai, X., Yu, R., Qu, L., et al. (2023). Comparison of SARIMA model, Holt-Winters model and ETS model in predicting the incidence of foodborne disease. _BMC Infectious Diseases, 23_, 803. [https://doi.org/10.1186/s12879-023-08799-4](https://doi.org/10.1186/s12879-023-08799-4)
