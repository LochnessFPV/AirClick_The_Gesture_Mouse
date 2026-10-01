import pytest

Proton = pytest.importorskip("Proton")


@pytest.mark.parametrize(
    "spoken, phrase",
    [
        ("proton search github", "search"),
        ("what is the time", "time"),
        ("proton bye", "bye"),
    ],
)
def test_whole_words_are_matched(spoken, phrase):
    assert Proton.said(spoken, phrase)


@pytest.mark.parametrize(
    "spoken, phrase",
    [
        ("proton open my research folder", "search"),
        ("sort by name", "bye"),
        ("update the candidate list", "date"),
    ],
)
def test_substrings_are_not_matched(spoken, phrase):
    """The old substring matching fired 'search' on 'research'."""
    assert not Proton.said(spoken, phrase)


def test_any_of_several_phrases_matches():
    assert Proton.said("proton quit now", "exit", "quit")


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("open 12", 12),
        ("proton open item 3 ", 3),
        ("open the first one", None),
        ("", None),
    ],
)
def test_trailing_index(spoken, expected):
    assert Proton.trailing_index(spoken) == expected


def test_out_of_range_choice_is_reported(monkeypatch):
    replies = []
    monkeypatch.setattr(Proton, "reply", replies.append)
    monkeypatch.setattr(Proton, "files", ["a.txt", "b.txt"])

    assert Proton.resolve_choice("open 9") is None
    assert Proton.resolve_choice("open everything") is None
    assert len(replies) == 2


def test_valid_choice_resolves_inside_the_browse_root(monkeypatch, tmp_path):
    monkeypatch.setattr(Proton, "files", ["a.txt", "b.txt"])
    monkeypatch.setattr(Proton, "path", tmp_path)
    assert Proton.resolve_choice("open 2") == tmp_path / "b.txt"
