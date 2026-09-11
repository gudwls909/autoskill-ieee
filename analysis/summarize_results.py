"""Reproduce the headline numbers of the paper from results/per_run_n10.csv
(no Minecraft logs needed).

    python analysis/summarize_results.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

rows = list(csv.DictReader(open(Path(__file__).resolve().parent.parent / "results/per_run_n10.csv")))
grp = {m: [r for r in rows if r["method"] == m] for m in ("Voyager", "AutoSkill")}

def col(m, k):
    return np.array([float(r[k]) for r in grp[m]])

print(f"{'metric':16s} {'Voyager':>14s} {'AutoSkill':>14s} {'delta':>8s}")
for k in ("distinct_items", "tts", "library_size", "t3_items", "t4_items", "t5_items"):
    v, a = col("Voyager", k), col("AutoSkill", k)
    print(f"{k:16s} {v.mean():7.1f} ±{v.std():4.1f} {a.mean():7.1f} ±{a.std():4.1f} {100*(a.mean()/v.mean()-1):+7.1f}%")
v, a = col("Voyager", "distinct_items") / col("Voyager", "library_size"), \
       col("AutoSkill", "distinct_items") / col("AutoSkill", "library_size")
print(f"items/skill (ratio of means) {col('Voyager','distinct_items').mean()/col('Voyager','library_size').mean():.2f} vs "
      f"{col('AutoSkill','distinct_items').mean()/col('AutoSkill','library_size').mean():.2f}")

for k in ("tts", "distinct_items"):
    u, p = mannwhitneyu(col("AutoSkill", k), col("Voyager", k), alternative="greater")
    print(f"Mann-Whitney U ({k}, one-sided): U={u:.1f}, p={p:.2g}")
print("complete separation on TTS:", col("AutoSkill", "tts").min() > col("Voyager", "tts").max())
