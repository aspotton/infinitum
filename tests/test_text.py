from infinitum.text import lexical_similarity


def test_lexical_similarity_prefers_related_text():
    related = lexical_similarity("PostgreSQL database standard", "PostgreSQL 17 is our database standard")
    unrelated = lexical_similarity("PostgreSQL database standard", "The user prefers concise answers")
    assert related > unrelated
    assert related > 0.25
