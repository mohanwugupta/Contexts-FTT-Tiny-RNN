from research_search.common import PACKAGE,read
def test_untriggered_historical_measure_is_missing_not_false():
    rows=read(PACKAGE/'history/SCREEN_HYPOTHESES_v2.json')
    a=next(r for r in rows if r['candidate']=='A07')
    assert all(x is None for x in a['original_early'])
