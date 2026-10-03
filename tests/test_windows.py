from lal.windows import make_windows


def test_windows_cover_period_with_margin_and_stride():
    words = [{"start": float(t), "end": t + 0.5, "word": f" w{t}"} for t in range(0, 400)]
    align = {"match": "m", "periods": {"1": {"offset": 100.0, "audio_start": 100.0, "audio_end": 200.0}}}
    wins = make_windows(words, align, length=20, stride=5, margin=10)
    assert wins[0]["start"] == 90.0 and wins[-1]["end"] <= 210.0 + 1e-9
    assert all(b["start"] - a["start"] == 5 for a, b in zip(wins, wins[1:]))
    assert wins[0]["n_words"] == 20 and wins[0]["t_start"] == -10.0
    assert wins[0]["text"].startswith("w90") and "w110" not in wins[0]["text"]
