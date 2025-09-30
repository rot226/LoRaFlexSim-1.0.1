from scipy.stats import kstest, expon


class ShiftedExponential:
    """Callable object whose ``__call__`` expects an extra ``loc`` argument."""

    def __call__(self, x, loc):
        return expon(loc=loc).cdf(x)


def test_kstest_callable_object_args():
    data = [1.5, 2.0, 2.5, 3.0]
    stat, p = kstest(data, ShiftedExponential(), args=(1.0,))
    # The important part is that ``kstest`` can invoke the callable without
    # raising ``TypeError`` and returns valid statistics.
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= p <= 1.0
