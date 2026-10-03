# Error analysis: fine-tuned laya-multilingual on the human gold set (WC22 final)

**Scope.** Gold set of 150 windows, 9 marked unsure and excluded. Thresholds were picked on the validation
matches, never on the test match. At those thresholds the fine-tuned model makes **6 false positives and
24 false negatives** across the four yes/no questions.

The cases below were picked to cover every cause. Excerpts are short and lightly trimmed
(PT-BR ASR output).

## The dominant cause: thresholds at the probability ceiling

The val-fitted temperature (noul 4.65; the training script's own fit was almost identical) maps the head's
large logits into a narrow band. **Every test probability lies in [0.018, 0.982].**

Laya returns probabilities rounded to 4 decimals. Near the top, many windows therefore tie, and a threshold
chosen on val (goal 0.9808, card 0.9794) sits right at the ceiling.

| question | gold | p | threshold | excerpt |
|---|---|---|---|---|
| goal | ✓ | 0.980 | 0.981 | "…autorizado, partiu para a bola, vem Lionel Messi, gol, é…" (Messi's penalty) |
| goal | ✓ | 0.979 | 0.981 | "Messi, e aí? Messi, gol! É da Argentina" (Messi's ET goal) |
| goal | ✓ | 0.975 | 0.981 | "…Thuram devolveu, Mbappé… Gol" (Mbappé's volley) |
| card | ✓ | 0.979 | 0.979 | "Marca o pênalti para a França. Amarelo para Montiel" |

- **Event level is unaffected:** each of these goals is still caught by a neighbouring window (6/6 goals,
  6/6 precision).
- **Post-hoc sensitivity**, *not* used for the headline: a goal threshold of 0.97 would give window F1 0.83
  instead of 0.70.
- **Lesson:** with Laya, pick thresholds with a margin below the ceiling, threshold on rank or logits, or
  keep unrounded scores.

## False positives (all 6)

| cause | n | question | excerpt / note |
|---|---|---|---|
| Penalty **awarded** read as a big chance | 2 | big_chance | "Di María foi para o chão, é pênalti!"; "invadiu, área! E é pênalti!" (gold: controversy) |
| Borderline chance (labeler strictness) | 2 | big_chance | "Acuña na área, bate para o gol… direto para fora, incrível" (gold: not a clear chance) |
| Card **anticipated** before it is shown | 1 | card | "Pode subir a cor que quiser… Amarelo? Vai dar amarelo?" |
| Post-goal recap with score and "golaço" | 1 | goal | "2 para a Argentina, 0 para a França, que golaço…" (a recap after the call; gold: not goal) |

## False negatives (10 of 24 shown)

| cause | n (of 24) | question | excerpt / note |
|---|---|---|---|
| Threshold at the ceiling (see above) | 4 | goal ×3, card | p within 0.006 of the threshold |
| **Real controversy scored at the floor** | 12 | controversy | All scored p 0.019–0.026 against a threshold of 0.040. Examples: "Calma que não vai valer… seria o Lautaro impedido? Vamos ver na repetição"; "O juizão apitou. Não é possível que tu apitou". Lowering the threshold *post hoc* to 0.019 only gives F1 0.36, so these sit among ordinary windows. The model learned the loud cases (VAR, "é pênalti") and misses quieter disputes. |
| Goal call **split across a window boundary** | 2 | goal | "…Alexis cruza, Di María, gol" ends one window; "da Argentina! Contra-ataque de manual" opens the next |
| Chance described through the **save**, not the shot | 3 | big_chance | "Martínez espetacular! Absurdo, absurdo" (the 120+3' one-on-one); "Quase passou da bola o Lloris" |
| Chance talk in **replay/analysis** form | 2 | big_chance | "Olha o gol do Montiel… a travada do Upamecano é histórica" (the human counted it, the model didn't) |
| Partly garbled ASR | 1 | big_chance | "…ele soltou-lhe um petáculo…" |

## What this suggests trying next (not done here)

- **Previous-window context**, the ablation already supported by `state_for(prev_text=...)`. It fixes split
  goal calls and gives the context a recap needs.
- **Threshold policy:** thresholds with a margin below the ceiling, or event-level F1 tuning on val.
- **More controversy training signal:** quieter disputes are rare in the 7 training matches. Most teacher
  controversy spans were marked uncertain and therefore ignored.
