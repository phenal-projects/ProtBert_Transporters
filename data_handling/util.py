from typing import List, Tuple
from sklearn.model_selection import train_test_split


def split_list(
    lst: List, train_size: float, val_size: float, random_state: int = 42
) -> Tuple[List, List, List]:
    assert (train_size + val_size) < 1, "Test size must be > 0"
    train, val_test = train_test_split(
        lst, test_size=(1 - train_size), random_state=random_state
    )
    val, test = train_test_split(
        val_test,
        test_size=(1 - train_size - val_size) / (1 - train_size),
        random_state=random_state,
    )
    return train, val, test
