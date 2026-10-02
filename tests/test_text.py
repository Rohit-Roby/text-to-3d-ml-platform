from src.preprocessing.text import clean_description


def test_clean_description():
    assert clean_description("  Modern, RED chair!! ") == "modern red chair"
