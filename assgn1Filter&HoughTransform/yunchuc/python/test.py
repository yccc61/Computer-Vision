import time
from contextlib import contextmanager
from collections import OrderedDict

class SimpleTimer:
    save_dict = OrderedDict()

    @classmethod
    @contextmanager
    def time(cls, key):
        start_time = time.perf_counter()
        cls.save_dict[key] = None
        yield None
        time_elapsed = time.perf_counter() - start_time
        cls.save_dict[key] = time_elapsed

    @classmethod
    def print(cls):
        for k, v in cls.save_dict.items():
            print("{}: {:.2f}ms".format(k, v * 1000))
if __name__ == "__main__":
    with SimpleTimer.time("Sleeping for 1 second..."):
        time.sleep(1)
    with SimpleTimer.time("Outer Nest"):
        with SimpleTimer.time("----Inner Nest"):
            time.sleep(1)
        time.sleep(1)
    SimpleTimer.print()