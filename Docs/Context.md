# Project Context: Ames Housing Exploratory Data Analysis (EDA)

## 1. Project Background & Objective
This project investigates the **Ames Housing Dataset**, a thorough real estate data source containing residential property sales in Ames, Iowa. While standard buying criteria often focus superficially on metrics like bedroom counts or fencing, this exploratory data analysis (EDA) uncovers the complex, multidimensional architectural, structural, and geographic factors that dictate true property valuation.

The explicit focus of this assignment is a rigorous **Exploratory Data Analysis (EDA)**. The core objectives are:
* To evaluate dataset dimensions, data types, and summary statistics across 81 distinct variables.
* To identify and handle missing data structures logically through domain-specific interpretation.
* To discover underlying linear and non-linear correlations between structural features and house sale prices (`SalePrice`).
* To detect data anomalies, skewness, and leveraged outliers that influence real estate valuation.

## 2. Dataset Scope & Dimensions
* **Observations (Rows):** 1,460 residential home sales.
* **Features (Columns):** 81 variables (38 numerical metrics and 43 categorical attributes).
* **Target Variable:** `SalePrice` (measured in CAD/USD), representing final negotiation transaction values.

## 3. Key Analytical Findings
1. **Univariate Skewness:** Both `SalePrice` and above-grade living area (`GrLivArea`) exhibit pronounced right-skewed distributions, with mean values pulled significantly higher than medians due to high-end luxury outliers.
2. **Dominant Price Drivers:** Pearson correlation screening identified **Overall Material and Finish Quality (`OverallQual`, $r = 0.79$)** and **Above-Grade Living Area (`GrLivArea`, $r = 0.71$)** as the primary linear drivers of home valuation.
3. **Missing Value Rationalization:** High-percentage missingness in features such as `PoolQC`, `MiscFeature`, `Alley`, and `Fence` was proven to represent the physical absence of the feature rather than data omission, allowing robust categorical encoding as `"None"`.
4. **Anomalies & Non-Linearity:** Bivariate scatterplots revealed critical anomalies, including extreme high-square-footage partial sales and non-linear valuation drops in 4-car garages, highlighting properties that require preprocessing prior to future predictive modeling.
5. **Geographic Disparities:** Neighborhood analysis demonstrated extreme price stratification across districts in Ames, with median values ranging from under $100,000 (IDOTRR, MeadowV) to over $300,000 (NridgHt, StoneBr).

## 4. Repository Structure
In accordance with course requirements, the project repository is structured as follows:
* `src/`: Contains Python scripts and Jupyter Notebooks implementing data cleaning, visualization, and EDA workflows.
* `data/`: Houses raw data source files (`train.csv`).
* `docs/`: Contains formal assignment documentation, including this `project_context.md` file, the `Executive_Summary.pdf`, and the `data_dictionary.xlsx`.
* `results/`: Stores exported figures, charts, and audit summaries.