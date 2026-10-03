from lal.schema import QUESTION_IDS, QUESTIONS, validate


def test_schema_valid():
    validate()
    assert set(QUESTIONS) == set(QUESTION_IDS)
    for q in QUESTIONS.values():
        assert q["instructions"] and q["criteria"]
