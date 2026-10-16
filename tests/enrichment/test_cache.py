import threading

from amazon_recsys.enrichment.cache import ResponseCache, make_input_hash

KEY = ("extractor", "r1", "ph", "gpt", "ih")


def test_put_get_round_trip(tmp_path):
    with ResponseCache(tmp_path / "c.sqlite") as cache:
        assert cache.get(*KEY) is None
        cache.put(*KEY, '{"a": 1}')
        assert cache.get(*KEY) == '{"a": 1}'
        assert cache.count() == 1 and cache.count("extractor") == 1 and cache.count("judge") == 0


def test_miss_on_any_key_component_change(tmp_path):
    with ResponseCache(tmp_path / "c.sqlite") as cache:
        cache.put(*KEY, "x")
        for position, other in enumerate(["judge", "r2", "ph2", "gpt2", "ih2"]):
            changed = list(KEY)
            changed[position] = other
            assert cache.get(*changed) is None, position


def test_judge_entry_invalidated_when_extraction_hash_changes(tmp_path):
    review_input = "review text"
    first = make_input_hash(review_input, "extraction-hash-A")
    second = make_input_hash(review_input, "extraction-hash-B")
    assert first != second
    assert first == make_input_hash(review_input, "extraction-hash-A")
    with ResponseCache(tmp_path / "c.sqlite") as cache:
        cache.put("judge", "r1", "qh", "jev-1", first, "answers")
        assert cache.get("judge", "r1", "qh", "jev-1", first) == "answers"
        assert cache.get("judge", "r1", "qh", "jev-1", second) is None


def test_survives_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    with ResponseCache(path) as cache:
        cache.put(*KEY, "persisted")
    with ResponseCache(path) as reopened:
        assert reopened.get(*KEY) == "persisted"


def test_concurrent_writes_do_not_corrupt(tmp_path):
    path = tmp_path / "c.sqlite"
    cache = ResponseCache(path)

    def worker(offset: int) -> None:
        for n in range(100):
            cache.put("extractor", f"r{offset}_{n}", "ph", "m", "ih", f"resp{offset}_{n}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.count() == 200
    assert cache.get("extractor", "r1_99", "ph", "m", "ih") == "resp1_99"
    cache.close()
    with ResponseCache(path) as reopened:
        assert reopened.count() == 200


def test_overwrite_same_key(tmp_path):
    with ResponseCache(tmp_path / "c.sqlite") as cache:
        cache.put(*KEY, "old")
        cache.put(*KEY, "new")
        assert cache.get(*KEY) == "new" and cache.count() == 1
