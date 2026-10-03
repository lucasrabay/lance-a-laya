from lal.transcripts import clean_words, text_between


def _w(word, t):
    return {"start": t, "end": t + 0.3, "word": f" {word}", "p": 0.9}


def test_loop_cut_keeps_short_repeats():
    words = [_w("gol", i) for i in range(4)]
    seg = {"text": "gol gol gol gol", "words": words}
    out, stats = clean_words([seg])
    assert len(out) == 4 and stats["loop_cut"] == 0


def test_loop_cut_trims_runaway_loops():
    words = [_w("vai", 2 * i) if i % 2 == 0 else _w("Brasil", 2 * i) for i in range(30)]
    out, stats = clean_words([{"text": "vai Brasil " * 15, "words": words}])
    assert len(out) <= 10 and stats["loop_cut"] >= 20


def test_stock_hallucination_dropped():
    segs = [
        {"text": "Legendas pela comunidade Amara.org", "words": [_w("Legendas", 0)]},
        {"text": "bola rolando", "words": [_w("bola", 5), _w("rolando", 5.5)]},
    ]
    out, stats = clean_words(segs)
    assert stats["dropped_stock"] == 1
    assert text_between(out, 0, 10) == "bola rolando"
