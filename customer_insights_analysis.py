# Customer Insights Statistical Investigation
# Requirements: pandas, numpy, matplotlib, scipy (optional), python-docx (for DOCX report)
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from datetime import datetime

base_dir = "statistics_customer_insights_output"
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(base_dir, exist_ok=True)

CSV_PATH = "YOUR_DATASET_PATH.csv"  # <-- Replace with actual path

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

expected_cols = [
    "CustomerID","Name","State","Education","Gender","Age","Married",
    "NumPets","MonthlySpend","DaysSinceLastInteraction"
]

col_map = {}
for col in expected_cols:
    if col in df.columns:
        col_map[col] = col
    else:
        matches = [c for c in df.columns if c.lower() == col.lower()]
        if matches:
            col_map[col] = matches[0]
df = df[[col_map[c] for c in expected_cols if c in col_map]]

numeric_cols = ["Age","NumPets","MonthlySpend","DaysSinceLastInteraction"]
for nc in numeric_cols:
    if nc in df.columns:
        df[nc] = pd.to_numeric(df[nc], errors="coerce")

if "Married" in df.columns:
    df["Married"] = df["Married"].astype(str).str.strip().str.title()
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
if "Education" in df.columns:
    df["Education"] = df["Education"].astype(str).str.strip().str.title()

# Understanding
info_summary = {
    "rows": len(df),
    "columns": list(df.columns),
    "null_counts": df.isna().sum().to_dict(),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "unique_counts": {c: int(df[c].nunique()) for c in df.columns}
}

# Descriptive stats
desc_numeric = df[numeric_cols].describe().T if set(numeric_cols).issubset(df.columns) else pd.DataFrame()
modes = {}
for c in ["Gender","Education","Married","State"]:
    if c in df.columns and df[c].dropna().shape[0] > 0:
        modes[c] = df[c].mode(dropna=True).iloc[0]

# Visualizations
def save_hist_and_box(column):
    if column not in df.columns: return
    series = df[column].dropna()
    if series.empty: return
    plt.figure(); plt.hist(series, bins=30); plt.title(f"Histogram of {column}"); plt.xlabel(column); plt.ylabel("Frequency")
    plt.savefig(os.path.join(plots_dir, f"{column}_hist.png"), bbox_inches="tight"); plt.close()
    plt.figure(); plt.boxplot(series, vert=True); plt.title(f"Boxplot of {column}"); plt.ylabel(column)
    plt.savefig(os.path.join(plots_dir, f"{column}_box.png"), bbox_inches="tight"); plt.close()

save_hist_and_box("Age")
save_hist_and_box("MonthlySpend")

def save_bar_count(column, top_n=None):
    if column not in df.columns: return
    counts = df[column].value_counts(dropna=False)
    if top_n: counts = counts.head(top_n)
    plt.figure(); counts.plot(kind="bar"); plt.title(f"Count of {column}"); plt.xlabel(column); plt.ylabel("Count")
    plt.savefig(os.path.join(plots_dir, f"{column}_bar.png"), bbox_inches="tight"); plt.close()

save_bar_count("Gender")
save_bar_count("Education")
save_bar_count("State", top_n=15)

if all(c in df.columns for c in ["Age","MonthlySpend"]):
    plt.figure(); plt.scatter(df["Age"], df["MonthlySpend"]); plt.title("Age vs MonthlySpend"); plt.xlabel("Age"); plt.ylabel("MonthlySpend")
    plt.savefig(os.path.join(plots_dir, "Age_vs_MonthlySpend_scatter.png"), bbox_inches="tight"); plt.close()

def save_kde_by_category(cat_col, value_col):
    if (cat_col not in df.columns) or (value_col not in df.columns): return
    data = df[[cat_col, value_col]].dropna()
    if data.empty: return
    categories = data[cat_col].value_counts().index[:4]
    xs = np.linspace(data[value_col].min(), data[value_col].max(), 200)
    import numpy as np
    plt.figure()
    plotted_any = False
    if SCIPY_AVAILABLE:
        from scipy import stats
        for cat in categories:
            subset = data.loc[data[cat_col] == cat, value_col].values
            if len(subset) < 5: continue
            kde = stats.gaussian_kde(subset)
            ys = kde(xs)
            plt.plot(xs, ys, label=str(cat)); plotted_any = True
    if plotted_any:
        plt.title(f"KDE of {value_col} by {cat_col}"); plt.xlabel(value_col); plt.ylabel("Density"); plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{value_col}_by_{cat_col}_kde.png"), bbox_inches="tight")
    plt.close()

save_kde_by_category("Education", "MonthlySpend")
save_kde_by_category("Married", "MonthlySpend")

# Correlation
corr_matrix = df[numeric_cols].corr() if set(numeric_cols).issubset(df.columns) else pd.DataFrame()

# Crosstab
crosstab_gm = pd.crosstab(df["Gender"], df["Married"]) if "Gender" in df.columns and "Married" in df.columns else None

# Grouped stats
grouped_stats = {}
for gc in ["State","Education","Gender"]:
    if gc in df.columns and "MonthlySpend" in df.columns:
        grouped_stats[gc] = df.groupby(gc)["MonthlySpend"].agg(["count","mean","median","std"]).sort_values("mean", ascending=False)

# Hypothesis tests
tests_results = {}
if SCIPY_AVAILABLE:
    if "Gender" in df.columns and "MonthlySpend" in df.columns:
        male = df.loc[df["Gender"].str.lower() == "male", "MonthlySpend"].dropna()
        female = df.loc[df["Gender"].str.lower() == "female", "MonthlySpend"].dropna()
        if len(male) > 2 and len(female) > 2:
            t_stat, p_val = stats.ttest_ind(male, female, equal_var=False, nan_policy="omit")
            tests_results["t_test_gender_spend"] = {"t_stat": float(t_stat), "p_value": float(p_val)}

    if "Education" in df.columns and "MonthlySpend" in df.columns:
        groups = [g["MonthlySpend"].dropna().values for _, g in df.groupby("Education")]
        groups = [g for g in groups if len(g) > 2]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            tests_results["anova_education_spend"] = {"f_stat": float(f_stat), "p_value": float(p_val)}

    if "Age" in df.columns and "DaysSinceLastInteraction" in df.columns:
        clean = df[["Age","DaysSinceLastInteraction"]].dropna()
        if len(clean) > 3:
            r, p_val = stats.pearsonr(clean["Age"], clean["DaysSinceLastInteraction"])
            tests_results["corr_age_inactivity"] = {"pearson_r": float(r), "p_value": float(p_val)}

    if "State" in df.columns and "MonthlySpend" in df.columns:
        top_states = df["State"].value_counts().head(10).index
        groups_state = [df.loc[df["State"] == st, "MonthlySpend"].dropna().values for st in top_states]
        groups_state = [g for g in groups_state if len(g) > 2]
        if len(groups_state) >= 2:
            f_stat, p_val = stats.f_oneway(*groups_state)
            tests_results["anova_state_spend_top10"] = {"f_stat": float(f_stat), "p_value": float(p_val), "states": list(map(str, top_states))}

# Save outputs
desc_numeric.to_csv(os.path.join(base_dir, "02_descriptive_numeric.csv"))
if corr_matrix is not None and not corr_matrix.empty:
    corr_matrix.to_csv(os.path.join(base_dir, "04_correlation_matrix.csv"))
if crosstab_gm is not None:
    crosstab_gm.to_csv(os.path.join(base_dir, "05_crosstab_gender_married.csv"))
for k,v in grouped_stats.items():
    v.to_csv(os.path.join(base_dir, f"06_grouped_stats_{k}.csv"))
with open(os.path.join(base_dir, "07_hypothesis_tests.json"), "w") as f:
    json.dump(tests_results, f, indent=2)

print("Analysis complete. See the 'statistics_customer_insights_output' folder for results.")
