def pytest_collection_modifyitems(items):
    "Don't type-check source files which only contain tests."

    from pytest_mypy import MypyFileItem

    def is_not_mypy_testfile_item(item):
        return not (
            isinstance(item, MypyFileItem) and item.fspath.basename.startswith("test_")
        )

    filter_in_place(is_not_mypy_testfile_item, items)


def filter_in_place(pred, items):
    """
    Like list(filter(pred, items)) but modifies the original list instead of
    returning a new one.
    """

    dst = src = 0
    while src < len(items):
        item = items[src]
        if pred(item):
            items[dst] = item
            dst += 1
        src += 1
    del items[dst:]
