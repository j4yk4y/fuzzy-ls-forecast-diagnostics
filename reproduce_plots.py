#!/usr/bin/env python3
"""Reproduce all figures from the paper.

Usage:
    pip install pandas numpy pyarrow matplotlib
    python reproduce_plots.py

Outputs PDF (vector) and PNG (300 dpi) to plots/.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)
RUNS = ["baseline", "+gated_lags", "+transitions"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

BLUE, ORANGE, GREEN, RED, PURPLE, GREY = (
    "#3C78B5", "#DD8452", "#55A868", "#C44E52", "#8B6DAF", "#888888")
MODEL_COLORS = {"baseline": BLUE, "+gated_lags": GREEN, "+transitions": ORANGE}
MODEL_LABELS = {"baseline": "Baseline", "+gated_lags": "+Gated Lags",
                "+transitions": "+Transitions"}
REGIME_COLORS = {"Cold": BLUE, "Cool": PURPLE, "Mild": "#E8A838",
                 "Warm": RED, "Hot": "#8B1A1A"}

TEMP_BP = {"cold": (-50,-50,5,12), "cool": (5,12,12,16),
           "mild": (12,16,16,22), "warm": (16,22,22,28),
           "hot": (22,28,50,50)}  # display names for Ruspini plot


def save(fig, name, width=3.5, height=2.5):
    fig.set_size_inches(width, height)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def trapmf(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a+1e-12), 1), (d-x)/(d-c+1e-12)))


TEMP_BREAKPOINTS = {"mu_cold":(-50,-50,5,12), "mu_cool":(5,12,12,16),
                    "mu_mild":(12,16,16,22), "mu_warm":(16,22,22,28),
                    "mu_hot":(22,28,50,50)}

def compute_fuzzy(temp):
    """Compute Ruspini-normalised fuzzy memberships from temperature."""
    x = temp.values.astype(float)
    mu = {k: trapmf(x,*v) for k,v in TEMP_BREAKPOINTS.items()}
    df = pd.DataFrame(mu, index=temp.index)
    return df.div(df.sum(axis=1).clip(lower=1e-12), axis=0)

def load():
    models = {}
    for name in RUNS:
        p = pd.read_parquet(DATA_DIR / f"predictions_{name}.parquet")
        p["ds"] = pd.to_datetime(p["ds"], utc=True)
        p["hour"] = p["ds"].dt.hour
        p["is_weekend"] = p["ds"].dt.dayofweek >= 5
        # Compute fuzzy memberships from temperature
        fuzzy = compute_fuzzy(p["temperature_2m_day2"])
        for col in fuzzy.columns:
            p[col] = fuzzy[col].values
        rc = {"Cold":"mu_cold","Cool":"mu_cool","Mild":"mu_mild",
              "Warm":"mu_warm","Hot":"mu_hot"}
        p["regime"] = p[list(rc.values())].idxmax(axis=1).map(
            {v:k for k,v in rc.items()})
        def tod(h):
            if h<=5: return "Night"
            elif h<=9: return "Morning"
            elif h<=14: return "Midday"
            elif h<=18: return "Afternoon"
            return "Evening"
        p["tod"] = p["hour"].apply(tod)
        models[name] = p
    return models


models = load()
bl = models["baseline"]

t_grid = np.linspace(-12, 36, 500)
display = {"cold":"Cold","cool":"Cool","mild":"Mild","warm":"Warm","hot":"Hot"}
cols = {"cold":BLUE,"cool":PURPLE,"mild":"#E8A838","warm":RED,"hot":"#8B1A1A"}

mu_grid = {k: trapmf(t_grid,*v) for k,v in TEMP_BP.items()}
total = sum(mu_grid.values())
for k in mu_grid: mu_grid[k] /= np.maximum(total, 1e-12)

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.hist(bl["temperature_2m_day2"].dropna(), bins=np.arange(-12,37,1),
         color="grey", alpha=0.25, edgecolor="white", linewidth=0.3)
ax2.set_ylabel("Count", color="grey"); ax2.tick_params(axis="y", colors="grey")
for k in TEMP_BP:
    ax.plot(t_grid, mu_grid[k], color=cols[k], linewidth=1.5, label=display[k])
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Membership degree")
ax.set_xlim(-12,36); ax.set_ylim(0,1.05)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
ax.set_title("Ruspini fuzzy partition (SIA 380/1)")
save(fig, "fig1_ruspini", width=3.5, height=2.3)

regimes = ["Cold","Cool","Mild","Warm","Hot"]
biases = [bl[bl["regime"]==r]["residual"].mean() for r in regimes]
fig, ax = plt.subplots()
bars = ax.bar(regimes, biases, color=[REGIME_COLORS[r] for r in regimes],
              width=0.6, edgecolor="grey", linewidth=0.3)
ax.axhline(0, color="grey", linewidth=0.8)
for bar, val in zip(bars, biases):
    ax.text(bar.get_x()+bar.get_width()/2, val/2, f"{val:+.0f}",
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
ax.set_ylabel("Mean bias (kWh)"); ax.set_title("Forecast bias by regime")
save(fig, "fig2_bias_by_regime", width=3.5, height=2.5)

tod_order = ["Night","Morning","Midday","Afternoon","Evening"]
grid = np.full((5, 5), np.nan)
for i, r in enumerate(regimes):
    for j, t in enumerate(tod_order):
        sub = bl[(bl["regime"]==r)&(bl["tod"]==t)]
        if len(sub) >= 15: grid[i,j] = sub["abs_residual"].mean()
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=700, vmax=5500)
ax.set_xticks(range(5)); ax.set_xticklabels(tod_order, rotation=30, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(regimes)
for i in range(5):
    for j in range(5):
        if not np.isnan(grid[i,j]):
            c = "white" if grid[i,j]>4000 else "black"
            ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center",
                    fontsize=7, color=c, fontweight="bold")
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("MAE (kWh)", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_title("Forecast error by regime and time of day")
save(fig, "fig3_regime_tod_heatmap", width=3.5, height=2.8)

tr = models["+transitions"]
improvements = []
for r in regimes:
    for t in tod_order:
        bs = bl[(bl["regime"]==r)&(bl["tod"]==t)]
        ts = tr[(tr["regime"]==r)&(tr["tod"]==t)]
        if len(bs)>=30:
            bm, tm = bs["abs_residual"].mean(), ts["abs_residual"].mean()
            imp = (bm-tm)/bm*100
            if abs(imp)>2: improvements.append((f"{r} / {t}", imp, r))
improvements.sort(key=lambda x:-x[1])
top = [x for x in improvements if x[1]>0][:10]
bot = [x for x in improvements if x[1]<=0]
improvements = top + bot

fig, ax = plt.subplots()
labels = [x[0] for x in improvements]
values = [x[1] for x in improvements]
colors = [REGIME_COLORS.get(x[2], GREY) for x in improvements]
bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="grey",
               linewidth=0.3, height=0.65)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("MAE improvement (%)"); ax.set_title("+Transitions vs Baseline")
ax.invert_yaxis()
for bar, val in zip(bars, values):
    if val >= 0:
        ax.text(val+0.4, bar.get_y()+bar.get_height()/2,
                f"+{val:.1f}%", va="center", ha="left", fontsize=6.5)
    else:
        ax.text(0.4, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left", fontsize=6.5, color="grey")
save(fig, "fig4_cross_model", width=4.0, height=4.0)

eq75 = bl["abs_residual"].quantile(0.75)
blp = bl.copy(); blp["high_error"] = blp["abs_residual"] > eq75
fig, ax = plt.subplots()
for regime, color in [("Cold",BLUE),("Cool",PURPLE),("Mild","#E8A838"),("Warm",RED)]:
    reg = blp[blp["regime"]==regime].copy().reset_index(drop=True)
    base = reg["high_error"].mean()
    lags = [1,2,3,6,12,24]; probs = []
    for lag in lags:
        reg["he_prev"] = reg["high_error"].shift(lag)
        cond = reg[reg["he_prev"]==True]
        probs.append(cond["high_error"].mean() if len(cond)>20 else np.nan)
    ax.plot(lags, probs, "o-", color=color, linewidth=1.3, markersize=4,
            label=f"{regime} (base {base:.0%})")
    ax.axhline(base, color=color, linewidth=0.6, linestyle=":", alpha=0.5)
ax.set_xlabel("Lag (hours)"); ax.set_ylabel("P(high error | prev high)")
ax.set_xticks(lags); ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(framealpha=0.9, edgecolor="none", fontsize=7, loc="upper right")
ax.set_title("Error persistence by regime")
save(fig, "fig5_persistence", width=3.5, height=2.3)

cold = bl[bl["regime"]=="Cold"]
cats = ["Holiday","Normal","Weekend","Weekday"]
biases_cal = [cold[cold["is_holiday"]==True]["residual"].mean(),
              cold[cold["is_holiday"]==False]["residual"].mean(),
              cold[cold["is_weekend"]==True]["residual"].mean(),
              cold[cold["is_weekend"]==False]["residual"].mean()]
fig, ax = plt.subplots()
bars = ax.bar(cats, biases_cal, color=[RED,GREY,ORANGE,GREY], width=0.6,
              edgecolor="grey", linewidth=0.3)
ax.axhline(0, color="grey", linewidth=0.8)
for bar, val in zip(bars, biases_cal):
    ax.text(bar.get_x()+bar.get_width()/2, val/2, f"{val:+.0f}",
            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
ax.set_ylabel("Mean bias (kWh)"); ax.set_title("Cold regime: calendar bias")
ax.axvline(1.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.4)
save(fig, "fig6_calendar_bias", width=3.5, height=2.5)

fig, ax = plt.subplots()
for name in RUNS:
    p = models[name]
    monthly = p.groupby(p["ds"].dt.to_period("M"))["abs_residual"].mean()
    ax.plot([str(m) for m in monthly.index], monthly.values, "o-",
            color=MODEL_COLORS[name], linewidth=1.2, markersize=3,
            label=MODEL_LABELS[name])
ax.set_ylabel("MAE (kWh)")
ax.legend(framealpha=0.9, edgecolor="none", loc="upper left")
plt.xticks(rotation=45, ha="right"); ax.set_title("Monthly forecast error")
save(fig, "fig7_monthly_mae", width=3.5, height=2.5)
