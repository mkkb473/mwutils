import threading
import time
import os
import requests
import warnings
import json
import atexit
import jwt


def mili_time():
    return int(time.time() * 1000)


def nano_time():
    return int(time.time() * 1000 * 1000)


def append_logs_buf_to_local(logs_buf, filename):
    with open(filename, 'a+') as f:
        for entry in logs_buf:
            f.write(json.dumps(entry))


def append_logs_buf_to_remote(logs_buf, name, metadata, post_addr):
    json_struct = {"logs": logs_buf, "phase": name, "metadata": metadata}
    # try 3 times
    for _ in range(3):
        r = requests.post(post_addr, json=json_struct, headers={"Authorization": jwt.encode(
            {"whatever": "1"}, "857851b2-c28c-4d94-83c8-f607b50ccd03")})
        if r.status_code >= 400:
            # something wrong
            jb = ''
            try:
                jb = r.json()
            except:
                pass
            print("resp:", r)
            warnings.warn("code: {}, resp.json: {}, resp.text: {}".format(
                r.status_code, jb, r.text))
        else:
            return True
    return False


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class Logger():
    # Logger for metrics
    def __init__(self, name, sample_time_interval_seconds=10, metadata={}, local_path="", post_addr="", buffer_all=False):
        self._logs_buf = []
        self._sample_time_interval_seconds = max(sample_time_interval_seconds, 2)
        self.name = name
        self.metadata = metadata
        self.mutex = threading.Lock()
        self._thread = RepeatedTimer(
            self._sample_time_interval_seconds, self.flush)
        self.post_addr = post_addr
        self.remote = True if post_addr else False
        self.local_path = local_path
        self._user_buf = list()
        self._buffer_all = buffer_all
        self.memoize = dict()
        self.memoize_funcs = []

        # flush before exit
        atexit.register(self.flush)

    def append(self, val):
        with self.mutex:
            q = {"val": val, "timestamp": mili_time()}
            self._logs_buf.append(q)
            if self._buffer_all:
                self._user_buf.append(q)

    def start(self):
        # do not delete existing file,
        if self.local_path:
            with open(self.local_path, 'w+'):
                pass
        self._thread.start()

    def cancel(self):
        self.flush()
        self._thread.stop()

    def flush(self):
        with self.mutex:
            if self._logs_buf:
                if self.remote:
                    good = append_logs_buf_to_remote(
                        self._logs_buf, self.name, self.metadata, self.post_addr)
                    if not good:
                        # failed to send logs, retry next time.
                        return
                if self.local_path:
                    append_logs_buf_to_local(self._logs_buf, self.local_path)
                self._logs_buf = list()

    def log(self, val):
        for f in self.memoize_funcs:
            if callable(f):
                f(self.memoize, val)
        self.append(val)

    def add_memoize_funcs(self, memoize_funcs):
        for f in memoize_funcs:
            self.memoize_funcs.append(f)

    def show_buffer(self):
        print(self._logs_buf)

    def show_user_buffer(self):
        print(self._user_buf)


if __name__ == "__main__":
    lg = Logger("fake", sample_time_interval_seconds=1,
                metadata={}, local_path="./what.json")
    lg.start()
    lg.log({"v1": 1, "v2": 2})
    for i in range(30):
        time.sleep(0.3)
        lg.log({"v1": i, "v2": 100+2*i})
        if i % 10 == 0:
            lg.log([{"fucker": 1*i}, {"fucker2": 2*i+100}])
        if i % 5 == 0:
            lg.show_buffer()

    lg.cancel()
