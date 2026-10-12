# Spatial Machine Learning for Italian Electoral Outcomes

## Project Overview

Predicting municipal-level electoral outcomes and voter behavior across Italy is challenging due to deep socio-economic divides, spatial dependencies, and high-dimensional administrative data. This project builds an end-to-end spatial machine learning pipeline that models municipal vote distributions (e.g., party vote shares and abstention rates) by combining past electoral results with granular socio-demographic, economic, fiscal, infrastructural, and geographic indicators for Italian municipalities.

The pipeline solves critical real-world geospatial modeling challenges, including structural non-random missingness (such as regional tax autonomy and statistical disclosure censorship) via Bayesian spatial models (BYM2 in PyMC) and spatial block cross-validation to prevent data leakage during gradient boosted tree training.

## Technical Architecture

```
Raw Multi-Source Data ────► Harmonization & Spatial Joins ────► Bayesian Spatial Imputation
(Elections, ISTAT, MEF)     (String normalization, EPSG:32632)   (BYM2 + ADVI)
                                                                            │
                                                                            ▼
Model Output & Artifacts ◄─── Spatial Block Cross-Validation ◄─── Spatial Feature Engineering
(Trained XGBoost Pipeline)    (50km Block GroupShuffleSplit)     (Queen Lags, Distance to biggest cities)

```

### 1. Data Ingestion & Geospatial Harmonization

The data extraction module merges more than 20 heterogeneous public datasets at the municipal level (`com_istat_code` / `COMUNE`):

* **Electoral Records:** Turnout, total electorate, abstention ratio, and proportional vote shares across all competing national lists.


* **Demographics & Housing:** Population distribution by age cohorts, marital and civil union statuses, homeownership vs. tenancy, and homelessness.


* **Economy & Income:** Ministry of Economy and Finance (MEF) municipal IRPEF declarations by income source (employment, pensions, self-employment, capital/business earnings).


* **Territorial & Labour Dynamics:** Municipal fragility index, educational attainment, commuting matrix flows, agricultural census data, tourism capacity, bank branches, libraries, hospitals, and traffic accidents.


* **Geospatial Boundaries:** Municipal administrative boundaries merged and reprojected to `EPSG:32632`.



### 2. Bayesian Spatial Imputation (BYM2)

Administrative and statistical datasets frequently contain structural missingness that cannot be imputed with naive mean or median replacement:

* **Fiscal Structural Discrepancies:** The Autonomous Province of Bolzano (South Tyrol) exhibits missing/zero property income values due to its distinct local taxation regime.


* **Censored Employment Rates:** Specific specialized employment statistics are officially censored below a 3% threshold to protect privacy.


* **Methodology:** The pipeline utilizes PyMC to formulate spatial hierarchical models combining fixed covariate effects, unstructured random effects and spatial intrinsic autoregressive priors ($u \sim \text{ICAR}(W)$) based on a symmetric $k$-nearest neighbors adjacency graph. The models are estimated using Automatic Differentiation Variational Inference (ADVI) with informative priors to enforce domain constraints (such as bounding censored worker shares below 3%).



### 3. Spatial Feature Engineering

Spatial relationships are engineered using `libpysal` and `geopandas`:

* **Spatial Autoregressive Lags:** Row-standardized Queen contiguity weight matrices compute neighborhood lags for regional fragility indices.


* **Metropolitan Proximity:** Calculation of Euclidean distances from municipal centroids to the 10 primary Italian metropolitan poles.


* **Spatial Blocking:** Division of the national territory into uniform 50 km $\times$ 50 km spatial grid blocks.



### 4. Spatially Independent Cross-Validation & Modeling

Standard random $k$-fold cross-validation results in overly optimistic performance estimates due to spatial autocorrelation between adjacent municipalities. To ensure out-of-region generalization:

* **Spatial Group Splits:** `GroupShuffleSplit` assigns entire 50 km spatial blocks exclusively to train, validation, or test partitions.


* **Gradient Boosting:** Model training uses an `XGBRegressor` pipeline optimized with native categorical handling for regions, early stopping against the spatial validation fold, and evaluation on the unseen spatial test holdout using RMSE and $R^2$.



## Data Sources

* **Ministero dell'Interno (Eligendo):** Municipal election results and turnout metrics.


* **ISTAT (Istituto Nazionale di Statistica):** Demographic structures, labour statistics, agricultural census, territorial fragility indices, road safety, and municipal boundary shapefiles.


* **MEF (Dipartimento delle Finanze):** Municipal-level aggregate IRPEF tax declarations.



All raw data files reside inside the `Elections/` directory and are configured via `configs/pipeline_config.yaml`.


## Installation

### Prerequisites

* Python 3.12
* Poetry package manager

### Setup

Clone the repository and install the locked dependencies:

```
poetry install

```

## Usage

### Run the End-to-End Pipeline

Execute the full execution flow (data assembly, Bayesian imputation, feature engineering, spatial splitting, model training, and metric evaluation):

```
poetry run python run_pipeline.py

```

### Pipeline Configuration

All execution parameters can be adjusted directly in `configs/pipeline_config.yaml` without changing source code:

* `model.target_col`: Choose target outcome (e.g., `M5S_RATIO`, `ASTENSIONE_RATIO`, `PD_RATIO`).


* `features.block_size_meters`: Adjust the spatial blocking grid resolution (default: `50000` meters).


* `imputation.knn_neighbors`: Adjust adjacency parameters for the spatial ICAR prior graph (default: `8`).


* `training.xgboost`: Configure learning rate, maximum tree depth, regularization, and early stopping iterations.



## Tech Stack

* **Language & Core:** Python, NumPy, Pandas
* **Geospatial & Spatial Econometrics:** GeoPandas, PySAL (`libpysal`), Shapely
* **Probabilistic & Bayesian Modeling:** PyMC, ArviZ (ICAR priors, ADVI variational inference)
* **Machine Learning:** XGBoost, Scikit-learn
* **Environment & Packaging:** Poetry, PyYAML
