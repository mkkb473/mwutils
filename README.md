## 简介

mwuitls 是 ModelWhale 平台中进行训练监控的工具包


## 使用方法

class Run:
methods:
- __init__(self, name="lab_run", user_id="user1", lab_id="lab1", org_id="", 
                 flush_interval_seconds=5,
                 sys_stat_sample_size=5, sys_stat_sample_interval=5, local_path='', write_logs_to_local=False,
                 remote_path='', buffer_all_logs=False):
  会根据name生成一个唯一的runid
  - name: 当前一次Run的名字
  - user_id, lab_id, org_id: 自动从环境变量中获取的参数
  - flush_interval_seconds: 隔几秒清空一下log buffer并上传到remote_path+'/logs'下
  - sys_stat_sample_size: 最多攒几个system的metrics一起传
  - sys_stat_sample_interval: system的metrics取样间隔
  - local_path: logs本地根目录存储
  - write_logs_to_local: 是否写到本地
  - remote_path: logs上传到remote_path+'/logs'下，conclude时会post metadata到remote_path+'/conclude'
  - buffer_all_logs: 是否于内存中缓存所有的Logs（不影响上传的logs）（慎用）

- init_ml():
    无参数，需要在start_ml()前调用。未来考虑并到__init__()里。

- start_ml():
    无参数，令每个log thread开始运行

- log_ml(step=None, epoch=None, batch=None, loss=None, acc=None, phase="train"):
  存一个metric到对应phase的log buffer
  - step, epoch, batch: 从1开始, None则忽略;
  - loss: numpy.float32会被转成python float以jsonify，None则忽略;
  - acc: 同上
  - phase: 可以自己填，目前支持train, test, val, system


- conclude(show_memoize=True):
  结束，run_id报废，metadata删掉。
  - show_memoize: 显示自定义的记忆值

- add_memoize_funcs_to_logger:
  可以自己添加memoize function，先不建议用;

keras:
class MWCustomCallback:
  嵌入run之后就当一个普通的keras callback用，用完了记得conclude一下
- set_run(run): 嵌入一个run class，帮你init_ml() + start_ml()
- conclude_run(): 帮你conclude()

torch:
class LoggerHook:
  能给一个torch_loss_hook method当hook用
- set_run(run): 嵌入一个run class，帮你init_ml() + start_ml()
- conclude_run(): 帮你conclude()
