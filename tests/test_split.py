import pytest

from src.data.split import check_disjoint, episode_offsets, split_task_episodes


def test_split_50_episodes_40_5_5():
    s = split_task_episodes(50, [], n_val=5, n_test=5, n_train=40, seed=0)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (40, 5, 5)
    check_disjoint(s)
    assert sorted(s["train"] + s["val"] + s["test"]) == list(range(50))


def test_excluded_never_used_and_val_test_keep_size():
    s = split_task_episodes(51, [12, 28, 47], n_val=5, n_test=5, n_train=40, seed=4)
    used = set(s["train"]) | set(s["val"]) | set(s["test"])
    assert not used & {12, 28, 47}
    assert (len(s["val"]), len(s["test"])) == (5, 5)
    assert len(s["train"]) == 38


def test_train_cap_records_unused():
    s = split_task_episodes(55, [], n_val=5, n_test=5, n_train=40, seed=1)
    assert len(s["train"]) == 40 and len(s["unused"]) == 5
    assert not set(s["unused"]) & (set(s["train"]) | set(s["val"]) | set(s["test"]))


def test_split_is_deterministic():
    assert split_task_episodes(50, [3], 5, 5, 40, seed=7) == split_task_episodes(50, [3], 5, 5, 40, seed=7)
    assert split_task_episodes(50, [3], 5, 5, 40, seed=7) != split_task_episodes(50, [3], 5, 5, 40, seed=8)


def test_invalid_exclusion_raises():
    with pytest.raises(ValueError):
        split_task_episodes(10, [10], 2, 2, None, 0)


def test_episode_offsets_match_aggregation_order():
    assert episode_offsets([50, 50, 51, 50]) == [0, 50, 100, 151]


def test_check_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError):
        check_disjoint({"train": [1, 2], "val": [2], "test": [3]})
