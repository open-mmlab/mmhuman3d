import time

from mmhuman3d.utils.demo_utils import StopWatch


def test_stopwatch():
    window_size = 5
    test_loop = 10
    outer_time = 100
    inner_time = 100

    stop_watch = StopWatch(window=window_size)
    for _ in range(test_loop):
        with stop_watch.timeit():
            time.sleep(outer_time / 1000.)
            with stop_watch.timeit('inner'):
                time.sleep(inner_time / 1000.)

    _ = stop_watch.report()
    _ = stop_watch.report_strings()
