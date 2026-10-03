import random

from lal.align import CUES, cue_hits, fit_offset, residual


def _words_at(pairs):
    out = []
    for t, text in pairs:
        for j, tok in enumerate(text.split()):
            out.append({"start": t + 0.3 * j, "end": t + 0.3 * j + 0.25, "word": " " + tok})
    return sorted(out, key=lambda w: w["start"])


def test_recovers_synthetic_offset():
    rng = random.Random(0)
    true_offset = 3725.0
    anchors = [{"kind": "period_start", "t": 0.0}, {"kind": "period_end", "t": 2900.0}]
    anchors += [{"kind": "goal", "t": t} for t in (610.0, 1800.0)]
    anchors += [{"kind": "corner", "t": t} for t in (300.0, 900.0, 1500.0, 2200.0, 2600.0)]
    pairs = [(true_offset + 0.5, "rola a bola"), (true_offset + 2905.0, "termina o primeiro tempo")]
    pairs += [(true_offset + a["t"] + 2.5, "gooool do brasil") for a in anchors if a["kind"] == "goal"]
    pairs += [(true_offset + a["t"] - 12 + rng.uniform(-8, 8), "escanteio para o brasil")
              for a in anchors if a["kind"] == "corner"]
    # distractors: commentary noise and an unrelated "gol" far away
    pairs += [(true_offset + t, "a bola fica com o zagueiro") for t in range(0, 2900, 37)]
    pairs += [(true_offset - 900, "lembra daquele gol")]
    hits = cue_hits(_words_at(pairs))
    est = fit_offset(anchors, hits, center=true_offset + 400, radius=1200)
    assert abs(est - true_offset) < 3.0
    goal = next(a for a in anchors if a["kind"] == "goal")
    assert abs(residual(goal, est, hits)) < 3.0


def test_goal_regex_ignores_goleiro():
    hits = cue_hits(_words_at([(10.0, "o goleiro defende"), (20.0, "goooool")]))
    assert hits["goal_call"] == [20.0]
    assert CUES["corner"].pattern.match("escanteio para")
