#!/usr/bin/env python3
"""Reproduce all tables from the paper.

Usage:
    pip install pandas numpy pyarrow
    python reproduce.py

All data is included in the data/ directory.
No external dependencies beyond pandas, numpy, and pyarrow.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RUNS = ["baseline", "+gated_lags", "+transitions"]
RUN_LABELS = {"baseline": "Baseline", "+gated_lags": "+Gated Lags",
              "+transitions": "+Transitions"}

# SIA 380/1 breakpoints for fuzzy temperature partition
# Format: (a, b, c, d) for trapezoidal membership functions
TEMP_BREAKPOINTS = {
    "mu_cold": (-50, -50, 5, 12),
    "mu_cool": (5, 12, 12, 16),
    "mu_mild": (12, 16, 16, 22),
    "mu_warm": (16, 22, 22, 28),
    "mu_hot":  (22, 28, 50, 50),
}


def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal membership function."""
    return np.maximum(0.0, np.minimum(
        np.minimum((x - a) / (b - a + 1e-12), 1.0),
        (d - x) / (d - c + 1e-12)))


def compute_fuzzy_memberships(temp: pd.Series) -> pd.DataFrame:
    """Compute Ruspini-normalised fuzzy membership degrees from temperature.

    Uses SIA 380/1 breakpoints. Each row sums to 1.
    """
    x = temp.values.astype(float)
    mu = {name: trapmf(x, *params) for name, params in TEMP_BREAKPOINTS.items()}
    mu_df = pd.DataFrame(mu, index=temp.index)
    row_sums = mu_df.sum(axis=1).clip(lower=1e-12)
    return mu_df.div(row_sums, axis=0)



def load_predictions() -> dict[str, pd.DataFrame]:
    """Load prediction files and compute fuzzy memberships from temperature."""
    models = {}
    for name in RUNS:
        p = pd.read_parquet(DATA_DIR / f"predictions_{name}.parquet")
        p["ds"] = pd.to_datetime(p["ds"], utc=True)
        p["hour"] = p["ds"].dt.hour
        p["dow"] = p["ds"].dt.dayofweek
        p["is_weekend"] = p["dow"] >= 5
        p["date"] = p["ds"].dt.date

        # Compute fuzzy memberships from temperature (SIA 380/1)
        fuzzy = compute_fuzzy_memberships(p["temperature_2m_day2"])
        for col in fuzzy.columns:
            p[col] = fuzzy[col].values

        p["delta_mu_cold"] = p["mu_cold"].diff().fillna(0.0)

        # Defuzzification (maximum membership)
        regime_cols = {"Cold": "mu_cold", "Cool": "mu_cool", "Mild": "mu_mild",
                       "Warm": "mu_warm", "Hot": "mu_hot"}
        p["regime"] = p[list(regime_cols.values())].idxmax(axis=1).map(
            {v: k for k, v in regime_cols.items()})

        # Time of day
        def tod(h):
            if h <= 5: return "Night"
            elif h <= 9: return "Morning"
            elif h <= 14: return "Midday"
            elif h <= 18: return "Afternoon"
            return "Evening"
        p["tod"] = p["hour"].apply(tod)

        models[name] = p
    return models


def mu_few(p): return min(1.0, max(0.0, (0.3 - p) / 0.2))
def mu_about_half(p): return max(0.0, 1.0 - abs(p - 0.5) / 0.2)
def mu_most(p): return min(1.0, max(0.0, (p - 0.5) / 0.3))

QUANTIFIERS = {"Few": mu_few, "About half": mu_about_half, "Most": mu_most}

def best_quantifier(p):
    scores = {name: fn(p) for name, fn in QUANTIFIERS.items()}
    best = max(scores, key=scores.get)
    return best, scores[best]


def fuzzy_proportion(mu_r, mu_s):
    """Hudec et al. eq. 9: fuzzy-weighted proportion."""
    return float(np.minimum(mu_r, mu_s).sum() / max(mu_r.sum(), 1e-12))


def table_model_performance(models):
    print("\n" + "=" * 60)
    print("TABLE: Model Performance")
    print("=" * 60)
    print(f"{'Model':<15s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
    print("-" * 42)
    for name in RUNS:
        p = models[name]
        mae = p["abs_residual"].mean()
        rmse = np.sqrt((p["residual"] ** 2).mean())
        ss_res = (p["residual"] ** 2).sum()
        ss_tot = ((p["y"] - p["y"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f"{RUN_LABELS[name]:<15s} {mae:8.0f} {rmse:8.0f} {r2:8.4f}")


def table_regime_gradient(bl):
    print("\n" + "=" * 60)
    print("TABLE: Error Gradient Across Regimes")
    print("=" * 60)
    eq25, eq75 = bl["abs_residual"].quantile([0.25, 0.75])
    bl = bl.copy()
    bl["err_cat"] = pd.cut(bl["abs_residual"], bins=[-1, eq25, eq75, np.inf],
                           labels=["Low", "Medium", "High"])

    regime_cols = {"Cold": "mu_cold", "Cool": "mu_cool", "Mild": "mu_mild",
                   "Warm": "mu_warm", "Hot": "mu_hot"}

    print(f"{'Regime':<8s} {'n':>6s} {'MAE':>6s} {'%Low':>6s} {'%Med':>6s} {'%High':>6s} {'Bias':>8s}")
    print("-" * 48)
    for regime in ["Cold", "Cool", "Mild", "Warm", "Hot"]:
        sub = bl[bl["regime"] == regime]
        n = len(sub)
        if n < 10: continue
        print(f"{regime:<8s} {n:6d} {sub['abs_residual'].mean():6.0f}"
              f" {(sub['err_cat']=='Low').mean():6.0%}"
              f" {(sub['err_cat']=='Medium').mean():6.0%}"
              f" {(sub['err_cat']=='High').mean():6.0%}"
              f" {sub['residual'].mean():+8.0f}")

    # Fuzzy LS (Hudec eq. 9)
    print("\n  Linguistic summaries (Hudec eq. 9, fuzzy-weighted validity):")
    for regime, mu_col in regime_cols.items():
        mu_r = bl[mu_col].values
        for err in ["Low", "Medium", "High"]:
            mu_s = (bl["err_cat"] == err).astype(float).values
            p = fuzzy_proportion(mu_r, mu_s)
            q, v = best_quantifier(p)
            if v >= 0.4:
                print(f"    \"{q} {regime} hours have {err} error\""
                      f" (p={p:.0%}, v={v:.2f})")


def table_regime_tod(bl):
    print("\n" + "=" * 60)
    print("TABLE: Regime x Time-of-Day MAE")
    print("=" * 60)
    regimes = ["Cold", "Cool", "Mild", "Warm", "Hot"]
    tods = ["Night", "Morning", "Midday", "Afternoon", "Evening"]
    header = f"{'ToD':<12s}" + "".join(f"{r:>10s}" for r in regimes)
    print(header)
    print("-" * len(header))
    for td in tods:
        row = f"{td:<12s}"
        for regime in regimes:
            sub = bl[(bl["regime"] == regime) & (bl["tod"] == td)]
            if len(sub) >= 15:
                row += f"{sub['abs_residual'].mean():10.0f}"
            else:
                row += f"{'--':>10s}"
        print(row)


def table_ls_central(bl):
    print("\n" + "=" * 60)
    print("TABLE: Central Linguistic Summaries (Hudec eq. 9)")
    print("=" * 60)
    eq25, eq75 = bl["abs_residual"].quantile([0.25, 0.75])
    bl = bl.copy()
    bl["err_Low"] = (bl["abs_residual"] < eq25).astype(float)
    bl["err_Med"] = ((bl["abs_residual"] >= eq25) & (bl["abs_residual"] <= eq75)).astype(float)
    bl["err_High"] = (bl["abs_residual"] > eq75).astype(float)

    regime_cols = {"Cold": "mu_cold", "Cool": "mu_cool", "Mild": "mu_mild",
                   "Warm": "mu_warm", "Hot": "mu_hot"}

    print(f"\n{'Summary':<50s} {'p':>6s} {'v':>6s} {'n_eff':>8s}")
    print("-" * 74)

    # Regime -> Error
    print("  Regime to Error magnitude")
    for reg, mu_col in regime_cols.items():
        mu_r = bl[mu_col].values
        n_eff = mu_r.sum()
        for err in ["Low", "Med", "High"]:
            mu_s = bl[f"err_{err}"].values
            p = fuzzy_proportion(mu_r, mu_s)
            q, v = best_quantifier(p)
            if v >= 0.4:
                print(f"  {q} {reg} hours have {err} error"
                      f"{'':<20s} {p:6.1%} {v:6.2f} {n_eff:8.0f}")

    # Regime x ToD -> Error
    print("\n  Regime x ToD to Error")
    for reg, mu_col in regime_cols.items():
        for td in ["Night", "Morning", "Midday", "Afternoon", "Evening"]:
            mask = bl["tod"] == td
            mu_r = bl.loc[mask, mu_col].values
            n_eff = mu_r.sum()
            if n_eff < 15: continue
            for err in ["Low", "Med", "High"]:
                mu_s = bl.loc[mask, f"err_{err}"].values
                p = fuzzy_proportion(mu_r, mu_s)
                q, v = best_quantifier(p)
                if v >= 0.9:
                    print(f"  {q} {reg}/{td} hours have {err} error"
                          f"{'':<10s} {p:6.1%} {v:6.2f} {n_eff:8.0f}")

    # Calendar x Regime -> Bias
    print("\n  Calendar x Regime to Bias direction")
    for reg, mu_col in regime_cols.items():
        for cal, col, val in [("weekend", "is_weekend", True),
                              ("weekday", "is_weekend", False)]:
            mask = bl[col] == val
            mu_r = bl.loc[mask, mu_col].values
            n_eff = mu_r.sum()
            if n_eff < 30: continue
            for direction, sign in [("over-predict", 1), ("under-predict", -1)]:
                mu_s = ((bl.loc[mask, "residual"] * sign) > 0).astype(float).values
                p = fuzzy_proportion(mu_r, mu_s)
                q, v = best_quantifier(p)
                if v >= 0.5:
                    print(f"  {q} {reg} {cal} hours {direction}"
                          f"{'':<10s} {p:6.1%} {v:6.2f} {n_eff:8.0f}")


def table_cross_model(models):
    print("\n" + "=" * 60)
    print("TABLE: Cross-Model Comparison (Baseline vs +Transitions)")
    print("=" * 60)
    bl, tr = models["baseline"], models["+transitions"]
    print(f"{'Subdomain':<25s} {'BL MAE':>8s} {'+T MAE':>8s} {'Improv.':>8s}")
    print("-" * 52)
    results = []
    for regime in ["Cold", "Cool", "Mild", "Warm", "Hot"]:
        for td in ["Night", "Morning", "Midday", "Afternoon", "Evening"]:
            bl_sub = bl[(bl["regime"] == regime) & (bl["tod"] == td)]
            tr_sub = tr[(tr["regime"] == regime) & (tr["tod"] == td)]
            if len(bl_sub) >= 30:
                bl_mae = bl_sub["abs_residual"].mean()
                tr_mae = tr_sub["abs_residual"].mean()
                imp = (bl_mae - tr_mae) / bl_mae * 100
                if abs(imp) > 3:
                    results.append((f"{regime} / {td}", bl_mae, tr_mae, imp))
    results.sort(key=lambda x: -x[3])
    for label, bl_mae, tr_mae, imp in results:
        print(f"{label:<25s} {bl_mae:8.0f} {tr_mae:8.0f} {imp:+7.1f}%")


def table_persistence(bl):
    print("\n" + "=" * 60)
    print("TABLE: Error Persistence")
    print("=" * 60)
    eq75 = bl["abs_residual"].quantile(0.75)
    bl = bl.copy()
    bl["high_error"] = bl["abs_residual"] > eq75

    print(f"{'Regime':<12s} {'Base':>6s} {'t+1h':>10s} {'t+3h':>10s} {'t+6h':>10s}")
    print("-" * 42)
    for regime in ["Cold", "Cool", "Mild", "Warm"]:
        reg = bl[bl["regime"] == regime].copy().reset_index(drop=True)
        base = reg["high_error"].mean()
        row = f"{regime:<12s} {base:5.0%} "
        for lag in [1, 3, 6]:
            reg["he_prev"] = reg["high_error"].shift(lag)
            cond = reg[reg["he_prev"] == True]
            if len(cond) > 20:
                p = cond["high_error"].mean()
                row += f" {p:4.0%} ({p/max(base,0.01):.1f}x)"
            else:
                row += f" {'--':>9s}"
        print(row)

    # Morning -> Midday
    print()
    daily = bl.groupby("date").apply(lambda g: pd.Series({
        "regime_dom": g["regime"].mode().iloc[0],
        "morn_high": g.loc[g["tod"] == "Morning", "high_error"].mean()
            if (g["tod"] == "Morning").any() else np.nan,
        "mid_high": g.loc[g["tod"] == "Midday", "high_error"].mean()
            if (g["tod"] == "Midday").any() else np.nan,
    })).reset_index().dropna()

    bad_m = daily[daily["morn_high"] > 0.5]
    good_m = daily[daily["morn_high"] <= 0.5]
    print(f"Morning -> Midday (all regimes):")
    print(f"  P(bad midday | bad morning)  = {(bad_m['mid_high'] > 0.5).mean():.0%}"
          f"  (n={len(bad_m)} days)")
    print(f"  P(bad midday | good morning) = {(good_m['mid_high'] > 0.5).mean():.0%}"
          f"  (n={len(good_m)} days)")

    cold_bad = daily[(daily["regime_dom"] == "Cold") & (daily["morn_high"] > 0.5)]
    if len(cold_bad) >= 5:
        p = (cold_bad["mid_high"] > 0.5).mean()
        v = mu_most(p)
        print(f"\n  LS: \"Most cold days with bad mornings also have bad middays\""
              f" (p={p:.0%}, v={v:.2f})")


def table_calendar(bl):
    print("\n" + "=" * 60)
    print("TABLE: Calendar Effects")
    print("=" * 60)
    print(f"{'Calendar':<12s} {'Regime':<12s} {'MAE':>8s} {'Bias':>8s} {'n':>6s}")
    print("-" * 48)
    for cal_name, cal_col, cal_val in [
        ("Holiday", "is_holiday", True), ("Normal", "is_holiday", False),
        ("Weekend", "is_weekend", True), ("Weekday", "is_weekend", False),
    ]:
        for regime in ["Cold", "Cool", "Mild", "Warm", "Hot"]:
            sub = bl[(bl[cal_col] == cal_val) & (bl["regime"] == regime)]
            if len(sub) >= 20:
                print(f"{cal_name:<12s} {regime:<12s}"
                      f" {sub['abs_residual'].mean():8.0f}"
                      f" {sub['residual'].mean():+8.0f}"
                      f" {len(sub):6d}")
        if cal_name in ("Normal", "Weekday"):
            print()



def main():
    models = load_predictions()
    bl = models["baseline"]

    table_model_performance(models)
    table_regime_gradient(bl)
    table_regime_tod(bl)
    table_ls_central(bl)
    table_cross_model(models)
    table_persistence(bl)
    table_calendar(bl)


if __name__ == "__main__":
    main()
