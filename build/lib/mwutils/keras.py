from tensorflow import keras
import os


LOGS_VAL = "val_"


class MWCustomCallback(keras.callbacks.Callback):
    def set_run(self, run):
        self.run = run
        run.init_ml()
        run.start_ml()

    def conclude_run(self):
        self.run.conclude()

    def on_test_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('acc')
        acc = acc if acc else logs.get('accuracy')
        self.run.log_ml(batch=batch, loss=loss,
                        acc=acc, phase="test")

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('acc')
        acc = acc if acc else logs.get('accuracy')
        self.run.log_ml(epoch=epoch, loss=loss,
                        acc=acc, phase="train")
        self._log_val_if_exists(epoch, logs=logs)

    def _log_val_if_exists(self, epoch, logs=None):
        loss = logs.get(LOGS_VAL+'loss')
        acc = logs.get(LOGS_VAL+'acc')
        acc = acc if acc else logs.get(LOGS_VAL+'accuracy')
        if loss and acc:
            self.run.log_ml(epoch=epoch, loss=loss,
                            acc=acc, phase="val")

    # def on_train_begin(self, logs):

    def on_test_begin(self, logs=None):
        # todo
        self.run.start_ml()

    def on_test_end(self, logs=None):
        pass
        # self.run.conclude()

    def on_train_end(self, logs):
        pass
        # upload/save to local
        # self.run.conclude()
