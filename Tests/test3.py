from Utils import check_arguments


@check_arguments
def f(a, b: int, c: int | None = None, *, d: int) -> int:
    print(a, b)
    return a + b


if __name__ == "__main__":
    f(1, 2, d=4.0)
