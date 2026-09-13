from sklearn.feature_extraction.text import TfidfVectorizer

from agentmem_os.benchmarks.uncapped_lexical_retrieval import rank_turns


def test_rare_query_entity_survives_large_history_vocabulary():
    noise = " ".join(f"frequent{i}" for i in range(600))
    target = "The zephyrium device arrived yesterday."
    contents = [noise] * 4 + [target]
    legacy = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
    legacy.fit(contents)
    assert legacy.transform(["zephyrium"]).nnz == 0
    assert rank_turns(contents, "zephyrium", 1) == [target]


def test_scope_and_provenance_are_preserved_without_answer_metadata():
    allowed = ["The blue device arrived Monday.", "The red device arrived Friday."]
    assert rank_turns(allowed, "red", 1) == [allowed[1]]
    assert rank_turns(allowed, "unrelated", 5) == []


def test_equal_scores_are_repeatable_and_respect_limit():
    turns = ["device arrived one", "device arrived two", "device arrived six"]
    selected = rank_turns(turns, "device arrived", 2)
    assert len(selected) == 2
    assert selected == rank_turns(turns, "device arrived", 2)
    assert rank_turns(turns, "device", 0) == []


def test_short_empty_and_nonlexical_histories():
    assert rank_turns([], "device") == []
    assert rank_turns(["!!!", "😀"], "device") == []
    assert rank_turns(["device arrived"], " ") == []
    assert rank_turns(["device arrived"], "device") == ["device arrived"]
