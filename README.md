## Introduction

**mwutils** is a package developed to be used in the [ModelWhale](https://modelwhale.com/) platform for machine learning experiments tracking. It is now under active development.

At the moment, mwutils supports *keras* and *PyTorch*
mwutils has experimental support for *Tensorflow*


## Usage


class Run:
methods:
- __init__(self, name="lab_run", user_id="user1", lab_id="lab1", org_id="",
                 flush_interval_seconds=5,
                 sys_stat_sample_size=5, sys_stat_sample_interval=5, local_path='', write_logs_to_local=False,
                 remote_path='', buffer_all_logs=False):
  - name: name for current experiment
  - user_id, lab_id, org_id: user shall get these information from the **notebook tools** panel in ModelWhale Notebook interface.
  - flush_interval_seconds: time interval to flush buffered logs, default is **5**
  - sys_stat_sample_size: maximum data size for system metrics, default is **5**
  - sys_stat_sample_interval: sampling rate for system metrics, default is **5**
  - write_logs_to_local: if logs should be stored locally
  - local_path: local path to store logs
  - remote_path: kesci api
  - buffer_all_logs: if all logs should be buffered in RAM (use with caution)

- init_ml():

- start_ml():

- log_ml(step=None, epoch=None, batch=None, loss=None, acc=None, phase="train"):
  - step, epoch, batch: starting from **1**;
  - loss: numpy.float32 woulde be converted to **float** type;
  - acc: numpy.float32 woulde be converted to **float** type;
  - phase: default is **train**, **test**, **val**, **system**


- conclude(show_memoize=True):
  call function after training end
  - show_memoize:

- add_memoize_funcs_to_logger:
  testing;

### Keras:

provide MWCustomCallback method

**example**

```
from mwutils.keras import MWCustomCallback
from mwutils.run import Run

r = Run(name=RUN_NAME,
        user_id = $user_id,
        lab_id = $lab_id,
        org_id = $org_id,
        flush_interval_seconds=30,
        sys_stat_sample_size=5,
        sys_stat_sample_interval=5,
        local_path='',
        write_logs_to_local=False,
        remote_path= $remote_path,
        buffer_all_logs=True)
callBack = MWCustomCallback()
callBack.set_run(r)        
history = model.fit(X_train, y_train,
                    epochs = 40,
                    batch_size = 32,
                    validation_data=(X_test,y_test),
                    shuffle=True,
                    callbacks=[callBack])
r.conclude()
```

### PyTorch:
provide ```LoggerHook``` method

**example**

```
from mwutils.torch_utils import LoggerHook
from mwutils.run import Run

r = Run(name=RUN_NAME,
        user_id = $user_id,
        lab_id = $lab_id,
        org_id = $org_id,
        flush_interval_seconds=30,
        sys_stat_sample_size=5,
        sys_stat_sample_interval=5,
        local_path='',
        write_logs_to_local=False,
        remote_path= $remote_path,
        buffer_all_logs=True)
loggerHook = LoggerHook()
loggerHook.set_run(r)
criterion.register_forward_hook(loggerHook.torch_loss_hook)

r.conclude()
```
