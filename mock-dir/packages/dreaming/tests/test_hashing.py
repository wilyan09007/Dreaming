from dreaming.hashing import canonical, env_sha, sha, short


def test_canonical_is_order_independent():
    assert canonical({"b": 1, "a": [1, 2]}) == canonical({"a": [1, 2], "b": 1}) == b'{"a":[1,2],"b":1}'
    assert sha({"x": 1}) == sha({"x": 1}) and sha({"x": 1}) != sha({"x": 2})


def test_env_hash_changes_with_threads():
    base = {"image": "sha256:abc", "numpy": "2.1", "threads": 1, "batch": 1, "device": "cpu"}
    assert env_sha(base) != env_sha({**base, "threads": 8})
    assert len(short(sha("x"))) == 6
