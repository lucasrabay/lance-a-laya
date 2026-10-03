**Window level, human gold set** (141 windows of the test match; 9 marked unsure are excluded). F1 at thresholds chosen on validation, with threshold-free average precision (AP) in parentheses.

| method | goal | big chance | controversy | card | intensity acc. (MAE) |
|---|---|---|---|---|---|
| Keyword rules | 0.17 (0.27) | 0.40 (0.59) | 0.47 (0.43) | 0.29 (0.40) | 0.38 (0.87) |
| Zero-shot Laya | 0.29 (0.31) | 0.08 (0.10) | 0.24 (0.26) | 0.06 (0.08) | 0.42 (0.76) |
| **Fine-tuned Laya** | 0.70 (0.93) | 0.62 (0.66) | 0.67 (0.78) | 0.57 (0.64) | 0.72 (0.30) |
| *Claude teacher (LLM reference)* | 0.87 (0.85) | 0.77 (0.84) | 0.57 (0.46) | 0.80 (0.52) | 0.79 (0.21) |

Gold positives: goal 12, big chance 16, controversy 20, card 4.

**Event level, whole test match.** Positive windows merge into triggers; an event is caught if a trigger lies within its tolerance (goal ±30 s, card −10 s…+120 s after the foul, big chance and controversy ±30 s of a human-confirmed moment). A highlight lasts at most 2 min. Cells: recall (caught / events) · precision (true triggers / triggers) · total flagged time (in-play match ≈ 142 min). Triggers that only match a moment the reviewer marked unsure are left out.

| method | goal | big chance | controversy | card |
|---|---|---|---|---|
| Keyword rules | 3/6 · 3/17 · 12 min | 5/12 · 5/7 · 5 min | 11/16 · 11/18 · 13 min | 4/7 · 4/5 · 4 min |
| Zero-shot Laya | 5/6 · 5/33 · 16 min | 12/12 · 16/67 · 108 min | 16/16 · 26/75 · 162 min | 7/7 · 11/71 · 84 min |
| **Fine-tuned Laya** | 6/6 · 6/6 · 7 min | 9/12 · 10/14 · 12 min | 12/16 · 12/20 · 12 min | 4/7 · 4/5 · 4 min |
| *Claude teacher (LLM reference)* | 6/6 · 6/6 · 7 min | 9/12 · 9/10 · 11 min | 9/16 · 9/14 · 15 min | 5/7 · 5/5 · 4 min |

**Calibration of the fine-tuned model on the gold set** (4 yes/no questions pooled, 10 bins): ECE 0.037 with temperatures fitted on validation vs 0.056 at T = 1.

**Latency** (fine-tuned, all 5 questions in one forward pass per window):

| GPU | p50 / p90 per window (batch 1) | batched throughput |
|---|---|---|
| Tesla T4 | 31.6 / 34.4 ms | 46.2 windows/s |
| NVIDIA L4 | 37.8 / 39.5 ms | 58.7 windows/s |

End-to-end in the streaming demo (this Mac in Brazil -> deployed Modal T4 -> back, 1698 sequential windows): p50 262 ms, p90 300 ms, against a 5 s budget per window.
