# 🏛️ IFRS 9 Expected Credit Loss (ECL) Modeling Engine

# By Rajiv Bandaru

An end-to-end, production-grade **IFRS 9 Expected Credit Loss (ECL)** quantitative risk framework built from scratch in Python. This engine processes credit portfolios, applies automated asset risk staging (SICR), trains an integrated machine learning Probability of Default (PD) pipeline, and subjects provisions to forward-looking macroeconomic stress scenarios.

---

## 📊 Core Framework & Methodology

The engine strictly implements the **IFRS 9 three-stage asset classification** architecture, replacing historical incurred-loss metrics with predictive, point-in-time impairment modeling.

### 🛡️ Asset Staging Architecture (SICR Engine)

Assets are dynamically assigned to risk buckets based on structural markers of a **Significant Increase in Credit Risk (SICR)** derived from account liquidity and facility duration features:

*   **Stage 1 (Performing):** Accounts maintaining stable liquid buffers. Impairment is calculated using a **12-Month ECL** horizon.
*   **Stage 2 (Underperforming / SICR):** Accounts demonstrating liquidity duress or extended terms (`Checking account == 'little'` or `Duration > 24 months`). Evaluated over a **Lifetime ECL** horizon.
*   **Stage 3 (Non-Performing / Impaired):** Extreme risk profiles exhibiting structural cash depletion alongside long-dated maturities. Anchored to a mandatory baseline **100% Probability of Default**.

---

## 🛠️ Integrated Parameters & Mathematical Logic

The application models mathematical provisions using the core regulatory banking equation:

\[\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}\]

### Engineered Factors Mapped
*   **Exposure at Default (EAD):** Factors in outstanding nominal quantities augmented by a dynamic **Credit Conversion Factor (CCF)** on committed undrawn limits to reflect true capital usage at the moment of default.
*   **Loss Given Default (LGD):** Grounded against a conservative baseline regulatory unsecured anchor (e.g., 45%) to evaluate structural hair-cuts on outstanding exposure.
*   **Forward-Looking PD Overlay:** Employs a point-in-time shift framework. Baseline machine learning probabilities are stressed across multiple macroeconomic scenarios (**Base, Upside, Downside**) and statistically aggregated:

\[\text{PIT PD} = (w_{\text{base}} \times \text{PD}_{\text{base}}) + (w_{\text{upside}} \times \text{PD}_{\text{upside}}) + (w_{\text{downside}} \times \text{PD}_{\text{downside}})\]

---

## 🤖 Machine Learning Pipeline Structure

To prevent structural data leakage and manage complex cross-feature patterns, the system utilizes a robust `scikit-learn` orchestration structure:

*   **Numerical Preprocessing:** Standardized scaling (`StandardScaler`) on continuous attributes like `Age`, `Credit amount`, and `Duration`.
*   **Categorical Preprocessing:** Robust isolated one-hot feature expansions (`OneHotEncoder`) managing high-cardinality vectors like loan `Purpose`, `Housing` types, and institutional `Saving accounts` stability indicators.
*   **Predictive Estimator:** A balanced class-weighted `LogisticRegression` pipeline engine calibrated to identify structural hazard boundaries.

---

## 🚀 Setup & Execution Guide

### Prerequisites
Ensure your local Python virtual environment contains the following pre-installed packages:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit
```

### Running the Interactive Dashboard
To interface with the data visualizer, execute the compiled application deployment via Streamlit:
```bash
streamlit run ecl_model_app.py
```

---

## 📈 Portfolio Risk Diagnostics Visualization

The application generates dual-panel diagnostic dashboards to analyze balance-sheet health:
1. **Provision Allocation Profiles:** Aggregates total dollar impairment reserves clustered cleanly across the three discrete IFRS Stages to monitor loss buffer adequacy.
2. **PD Dispersion Analysis:** Visualizes the statistical spread and distribution variations of calculated Forward-Looking PIT PDs across Stage 1 and Stage 2 assets to audit staging boundary stability.

---
*Developed for Quantitative Risk Engineering & IFRS 9 Financial Accounting Compliance.*
