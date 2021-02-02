import time
import tensorflow as tf
import warnings

if tf.__version__[0] == 1:
    SessionRunArgs = tf.train.SessionRunArgs
    SessionRunHook = tf.train.SessionRunHook
else:
    SessionRunArgs = tf.compat.v1.train.SessionRunArgs
    SessionRunHook = tf.estimator.SessionRunHook

NONE_INDEX = -333


class LoggerHook(SessionRunHook):
    """Logs loss and runtime."""

    def set_run(self, run, loss, phase="train"):
        warnings.warn("Native TF plugins are not supported yet.")
        return
        self.run = run
        self.phase = phase
        self.loss = loss
        self._epoch = -1
        run.init_ml()
        run.start_ml()

    def begin(self):
        self._epoch = -1

    def before_run(self, run_context):
        self._epoch += 1
        # Asks for loss value.
        return SessionRunArgs(self.loss)

    def after_run(self, run_context, run_values):
        loss_value = run_values.results
        self.run.log_ml(epoch=self._epoch, loss=loss_value.astype(
            float), phase=self.phase)
        #print("DEBUG: EPOCH {}, LOSS VALUE: {}".format(self._epoch, loss_value))

    def end(self, session):
        self.run.conclude()
