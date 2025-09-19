from bs4 import BeautifulSoup

from news_mvp.etl.load.enhancer_by_source import (
    extract_selector_list,
    try_selector_for_texts,
)


def test_extract_selector_list_bracketed():
    s = "[css:div.a],[css:span.b]"
    out = extract_selector_list(s)
    assert out == ["css:div.a", "css:span.b"]


def test_try_selector_for_texts_css_and_attr():
    html = '<div class="a"><a href="/x">Link</a></div><span id="s">SpanText</span>'
    soup = BeautifulSoup(html, "html.parser")
    # css selector
    texts = try_selector_for_texts(soup, "css:div.a")
    assert texts == ["Link"]
    # attr selector (id as attr match)
    texts2 = try_selector_for_texts(soup, 'id="s"')
    assert texts2 == ["SpanText"]
