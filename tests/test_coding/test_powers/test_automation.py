from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from diaad.coding.powers import automation


def test_powers_text_helpers(monkeypatch):
    monkeypatch.setattr(automation, "process_utterances", lambda text: text.lower().replace("_", " "))

    assert automation.count_fillers("um &-uh hello") == 2
    assert "cannot" in automation.expand_contractions("can't")
    assert automation.expand_and_process_utterances("&=code Hello-There") == "hello there"


def test_content_word_counting_and_be_fallback():
    token = lambda text, pos, tag="", like_num=False: SimpleNamespace(
        text=text,
        pos_=pos,
        tag_=tag,
        like_num=like_num,
        __str__=lambda self=None, t=text: t,
    )
    doc = [
        token("cat", "NOUN"),
        token("quickly", "ADV"),
        token("is", "AUX"),
    ]

    total_cw, total_nouns, tagged = automation.count_content_words_from_doc(doc)
    adjusted_tagged, adjusted_total = automation.check_main_verb("is ", 0)

    assert total_cw == 3
    assert total_nouns == 1
    assert "_NOUN_CW_N" in tagged
    assert adjusted_total == 1
    assert adjusted_tagged.endswith("_BE_FORM_MAIN")


def test_run_automation_adds_columns(monkeypatch):
    df = pd.DataFrame({"utterance": ["Cat runs"]})
    calls = {}

    def fake_get_powers_nlp(model_name="en_core_web_sm"):
        calls["model_name"] = model_name
        return SimpleNamespace(pipe=lambda utterances, **kwargs: [["doc"]])

    monkeypatch.setattr(automation, "get_powers_nlp", fake_get_powers_nlp)
    monkeypatch.setattr(automation, "expand_and_process_utterances", lambda utt: utt.lower())
    monkeypatch.setattr(automation, "compute_speech_units", lambda utt: 2)
    monkeypatch.setattr(automation, "count_fillers", lambda utt: 0)
    monkeypatch.setattr(automation, "_automate_content_measures", lambda utterances, nlp: ([3], [1], ["tagged"]))

    out = automation.run_automation(df.copy(), spacy_model_name="en_core_web_trf")

    assert list(out.columns) == ["utterance", "tagged_utterance", "speech_units", "filled_pauses", "content_words", "num_nouns"]
    assert out.loc[0, "content_words"] == 3
    assert calls["model_name"] == "en_core_web_trf"
