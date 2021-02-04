import mwutils
from mwutils.logs import Logger, mili_time
from mwutils.sys_stat import SystemStats
import os
import requests
import warnings
import jwt
import time
import traceback
import signal
import numpy as np
from os import path, getpid

_STEP = 'step'
_EPOCH = 'epoch'
_BATCH = 'batch'
_LOSS = 'loss'
_ACC = 'acc'
_TIMESTAMP = 'timestamp'

_MAX_ACC = "max_accuracy"
_MIN_LOSS = "min_loss"
_BEST = "best"

MODEL_TYPE_TF = "tf"
MODEL_TYPE_KERAS = "keras"
MODEL_TYPE_TORCH = "torch"
MODEL_TYPE_CUSTOM = "custom"


class MLLoger(Logger):
    def log(self, step=None, epoch=None, batch=None, loss=None, acc=None):
        val = dict()
        val[_TIMESTAMP] = int(time.time())
        if step is not None:
            val[_STEP] = step + 1
        if epoch is not None:
            val[_EPOCH] = epoch + 1
        if batch is not None:
            val[_BATCH] = batch + 1
        if loss is not None:
            val[_LOSS] = loss
        if acc is not None:
            val[_ACC] = acc

        if acc:
            if _MAX_ACC not in self.memoize and acc:
                self.memoize[_MAX_ACC] = val
            elif self.memoize[_MAX_ACC][_ACC] < val[_ACC]:
                self.memoize[_MAX_ACC] = val

        if loss:
            best = False
            if _MIN_LOSS not in self.memoize and loss:
                self.memoize[_MIN_LOSS] = val
                best = True
            elif self.memoize[_MIN_LOSS][_LOSS] > val[_LOSS]:
                self.memoize[_MIN_LOSS] = val
                best = True
            if best:
                if step is not None:
                    self.memoize["{}_{}".format(_BEST, _STEP)] = step+1
                elif epoch is not None:
                    self.memoize["{}_{}".format(_BEST, _EPOCH)] = epoch+1
                elif batch is not None:
                    self.memoize["{}_{}".format(_BEST, _BATCH)] = batch+1

        super().log(val)


def example_memoize_func(memoize_buf, val):
    if "cost" in val:
        if "min_cost" in memoize_buf and memoize_buf["min_cost"] > val["cost"]:
            memoize_buf["min_cost"] = val["cost"]


class CustomLogger(Logger):
    pass


class Run():
    def __init__(self, name="lab_run", user_id="user1", lab_id="lab1", org_id="", flush_interval_seconds=5,
                 sys_stat_sample_size=1, sys_stat_sample_interval=2, local_path='', write_logs_to_local=False,
                 remote_path='https://www.kesci.com/api/runs', buffer_all_logs=False):
        self._loggers = {}
        self.custom_loggers = {}
        env_user_id = os.getenv("KLAB_USER_ID")
        env_lab_id = os.getenv("KLAB_LAB_ID")
        env_org_id = os.getenv("KLAB_ORG_ID")
        timestr = str(mili_time())
        self.user_id = env_user_id if env_user_id else user_id
        self.lab_id = env_lab_id if env_lab_id else lab_id
        self.org_id = env_org_id if env_org_id else org_id
        self.run_id = name + '_' + timestr
        self.flush_interval_seconds = max(5, flush_interval_seconds)
        self._sys_stat_sample_size = sys_stat_sample_size
        self._sys_stat_sample_interval_seconds = sys_stat_sample_interval
        self.local_path = local_path
        self.write_logs_to_local = write_logs_to_local
        self.logs_remote_path = remote_path + '/logs' if remote_path else ''
        self.conclude_remote_path = remote_path + '/conclude'
        self.remote_path = remote_path
        self.abort_remote_path = remote_path + "/abort"
        self.buffer_all_logs = buffer_all_logs
        self.model_path = ""
        self.metadata = {"name": name, "user_id": user_id,
                         "lab_id": lab_id, "run_id": self.run_id, "org_id": org_id}
        self.pid = None
        self.started = False

    def init_ml(self):
        if self.pid:
            return
        self.pid = getpid()
        train_path = path.join(
            self.local_path, "train.json") if self.write_logs_to_local else ''
        test_path = path.join(
            self.local_path, "test.json") if self.write_logs_to_local else ''
        val_path = path.join(
            self.local_path, "val.json") if self.write_logs_to_local else ''
        sys_path = path.join(
            self.local_path, "sys.json") if self.write_logs_to_local else ''
        self._loggers['train'] = MLLoger("train", sample_time_interval_seconds=self.flush_interval_seconds,
                                          metadata=self.metadata, local_path=train_path, post_addr=self.logs_remote_path,
                                          buffer_all=self.buffer_all_logs)
        self._loggers['test'] = MLLoger("test", sample_time_interval_seconds=self.flush_interval_seconds,
                                         metadata=self.metadata, local_path=test_path, post_addr=self.logs_remote_path,
                                         buffer_all=self.buffer_all_logs)
        self._loggers['val'] = MLLoger("val", sample_time_interval_seconds=self.flush_interval_seconds,
                                        metadata=self.metadata, local_path=val_path, post_addr=self.logs_remote_path,
                                        buffer_all=self.buffer_all_logs)
        self._loggers['system'] = CustomLogger("system", sample_time_interval_seconds=self.flush_interval_seconds,
                                                metadata=self.metadata, local_path=sys_path, post_addr=self.logs_remote_path,
                                                buffer_all=self.buffer_all_logs)

    def start_ml(self):
        if self.started:
            return
        self.started = True
        self.__register_signal_handlers()
        for _, logger in self._loggers.items():
            logger.start()
        for _, clogger in self.custom_loggers.items():
            clogger.start()
        self.sys_stat = SystemStats(self)
        self.sys_stat.start()

    def log_ml(self, step=None, epoch=None, batch=None, loss=None, acc=None, phase="train"):
        # phase is the same thing with namea
        if isinstance(loss, np.float32):
            loss = float(loss)
        if isinstance(acc, np.float32):
            acc = float(acc)
        self._loggers[phase].log(step=step, epoch=epoch,
                                  batch=batch, loss=loss, acc=acc)

    def new_custom_logger(self, name, local_path=''):
        self.custom_loggers[name] = CustomLogger(name, sample_time_interval_seconds=self.flush_interval_seconds,
                                                 metadata=self.metadata, local_path=local_path, post_addr=self.logs_remote_path,
                                                 buffer_all=self.buffer_all_logs)

    def add_memoize_funcs_to_logger(self, name, funcs):
        self._loggers[name].add_memoize_funcs(funcs)

    def set_tf_model(self, model):
        self.model = model
        self.model_type = MODEL_TYPE_TF

    def _save_tf_model(self, model_path):
        # SavedModel
        # tf2
        import tensorflow as tf
        tf.saved_model.save(self.model, model_path)
        pass

    def set_keras_model(self, model):
        self.model = model
        self.model_type = MODEL_TYPE_KERAS

    def _save_keras_model(self, model_path):
        # SavedModel
        # tf2
        import tensorflow as tf
        tf.keras.models.save_model(self.model, model_path)
        self.model_path = model_path

    def set_torch_model(self, model):
        self.model = model
        self.model_type = MODEL_TYPE_TORCH

    def _save_torch_model(self, model_path):
        # torch version >= 1.6
        import torch
        torch.save(self.model, model_path)
        self.model_path = model_path
        pass

    def set_custom_model(self, path):
        self.model_type = MODEL_TYPE_CUSTOM
        self.model_path = path

    def _save_model(self, model_path):
        if hasattr(self, "model_type"):
            if self.model_type == MODEL_TYPE_TORCH:
                self._save_torch_model(model_path)
            elif self.model_type == MODEL_TYPE_KERAS:
                self._save_keras_model(model_path)
            elif self.model_type == MODEL_TYPE_TF:
                self._save_tf_model(model_path)

    def __upload_model(self):
        pass

    def __register_signal_handlers(self):
        signal.signal(signal.SIGINT, self.__sigint_handler)
        signal.signal(signal.SIGTERM, self.__sigterm_handler)

    def __sigint_handler(self, signum, frame):
        self.__abort_run("SIGINT", "[SIGINT]Terminated by system")
        traceback.print_stack(f=frame)
        raise RuntimeError("terminated by system")

    def __sigterm_handler(self, signum, frame):
        self.__abort_run("SIGTERM", "[SIGTERM]Terminated by user")
        traceback.print_stack(f=frame)
        raise KeyboardInterrupt("termniated by user")

    def __abort_run(self, sig, reason):
        if self.remote_path:
            tp = int(time.time())
            json_struct = {"metadata": self.metadata, "timestamp":tp, "signal": sig, "reason": reason}
            for _ in range(3):
                r = requests.post(self.abort_remote_path, json=json_struct, headers={"Authorization": jwt.encode(
                    {"whatever": "1"}, "79eb9467-8348-4b29-a997-7a9685e1a820")})
                if r.status_code >= 400:
                    # something wrong
                    jb = ''
                    try:
                        jb = r.json()
                    except:
                        pass
                    print("resp:", r)
                    msg = "code: {}, resp.json: {}, resp.text: {}".format(
                        r.status_code, jb, r.text)
                    print(msg)
                    warnings.warn(msg)
                else:
                    print("abort remote call succeed. resp:", r)
                    break
        self.started = False
        self.run_id = "aborted"

    def conclude(self, show_memoize=True, upload_model=False, model_path="./saved_model"):
        if not self.started:
            pass
        for _, logger in self._loggers.items():
            logger.cancel()
            if show_memoize and logger.memoize:
                print(logger.name, logger.memoize)
        for _, clogger in self.custom_loggers.items():
            clogger.cancel()
            if show_memoize and clogger.memoize:
                print(clogger.name, clogger.memoize)
        self._save_model(model_path)

        if self.remote_path:
            tp = int(time.time())
            json_struct = {"metadata": self.metadata, "best": [{"phase": name, "val": logger.memoize, _TIMESTAMP: tp} for name, logger in self._loggers.items()]}
            for _ in range(3):
                r = requests.post(self.conclude_remote_path, json=json_struct, headers={"Authorization": jwt.encode(
                    {"whatever": "1"}, "79eb9467-8348-4b29-a997-7a9685e1a820")})
                if r.status_code >= 400:
                    # something wrong
                    jb = ''
                    try:
                        jb = r.json()
                    except:
                        pass
                    print("resp:", r)
                    msg = "code: {}, resp.json: {}, resp.text: {}".format(
                        r.status_code, jb, r.text)
                    print(msg)
                    warnings.warn(msg)
                else:
                    print("conclude remote call succeed. resp:", r)
                    break
        if upload_model:
            self.__upload_model()
        self.started = False
        self.run_id = "concluded"


if __name__ == "__main__":
    import time

    def sys_memoize_func_maxcpu(memoize_buf, val):
        if "cpu" in val:
            if "max_cpu" in memoize_buf and memoize_buf["max_cpu"] < val["cpu"]:
                memoize_buf["max_cpu"] = val["cpu"]
            if "max_cpu" not in memoize_buf:
                memoize_buf["max_cpu"] = val["cpu"]

    def sys_memoize_func_mincpu(memoize_buf, val):
        if "cpu" in val:
            if "min_cpu" in memoize_buf and memoize_buf["min_cpu"] > val["cpu"]:
                memoize_buf["min_cpu"] = val["cpu"]
            if "min_cpu" not in memoize_buf:
                memoize_buf["min_cpu"] = val["cpu"]

    r = Run("test88", "testuser123", "proj123", "job123", flush_interval_seconds=5,
            local_path="/Users/mk/heyw/github/mwutils/mwutils", sys_stat_sample_interval=5, sys_stat_sample_size=21, buffer_all_logs=True)
    r.init_ml()
    r.add_memoize_funcs_to_logger(
        "system", [sys_memoize_func_maxcpu, sys_memoize_func_mincpu])

    r.start_ml()

    for i in range(150):
        r.log_ml(step=i, acc=i*(1/150), loss=149-i)
        time.sleep(0.2)
    for i in range(20):
        r.log_ml(epoch=i, acc=i*(1/40)+0.5, loss=1, phase='test')
        time.sleep(0.2)
    r.conclude()
