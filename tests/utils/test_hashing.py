from amazon_recsys.utils.hashing import hash_file, hash_obj, hash_text


def test_hash_obj_independent_of_key_order():
    assert hash_obj({"a": 1, "b": {"x": 1, "y": 2}}) == hash_obj({"b": {"y": 2, "x": 1}, "a": 1})


def test_hash_obj_changes_with_any_value():
    base = {"a": 1, "b": [1, 2]}
    assert hash_obj(base) != hash_obj({"a": 1, "b": [1, 3]})
    assert hash_obj(base) != hash_obj({"a": 2, "b": [1, 2]})


def test_hash_text_known_value():
    assert hash_text("abc") == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_hash_file_matches_known_value(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"abc")
    assert hash_file(f) == hash_text("abc")
