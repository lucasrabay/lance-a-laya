from lal.baselines.keywords import hits, predict, tune


def test_hits_and_predict():
    h = hits("GOOOOOL do Brasil! Que golaço! O goleiro nada pôde fazer")
    assert h["goal"] == 2  # goleiro is not a goal hit
    th = {"goal": 1, "big_chance": 1, "controversy": 1, "card": 1, "intensity": (1, 3)}
    p = predict("cartão amarelo para o zagueiro", th)
    assert p["card"]["pred"] == 1 and p["goal"]["pred"] == 0


def test_tune_prefers_threshold_with_best_f1():
    texts = ["gol gol", "gol", "nada", "gol gol gol"]
    labels = [{"goal": 1, "big_chance": None, "controversy": None, "card": None, "intensity": 2},
              {"goal": 0, "big_chance": None, "controversy": None, "card": None, "intensity": 0},
              {"goal": 0, "big_chance": None, "controversy": None, "card": None, "intensity": 0},
              {"goal": 1, "big_chance": None, "controversy": None, "card": None, "intensity": 2}]
    assert tune(texts, labels)["goal"] == 2
