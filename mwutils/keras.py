from tensorflow import keras
import os


LOGS_VAL = "val_"


class MWCustomCallback(keras.callbacks.Callback):
    def set_run(self, run):
        self.run = run
        self.test_epoch = 0
        run.init_ml()
        run.start_ml()

    def conclude_run(self):
        self.run.conclude()

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('acc')
        acc = acc if acc else logs.get('accuracy')
        self.run.log_ml(epoch=epoch, loss=loss,
                        acc=acc, phase="train", custom_logs=logs)
        self._log_val_if_exists(epoch, logs=logs)

    def _log_val_if_exists(self, epoch, logs=None):
        loss = logs.get(LOGS_VAL+'loss')
        acc = logs.get(LOGS_VAL+'acc')
        acc = acc if acc else logs.get(LOGS_VAL+'accuracy')
        if loss and acc:
            self.run.log_ml(epoch=epoch, loss=loss,
                            acc=acc, phase="val", custom_logs=logs)

    # def on_train_begin(self, logs):

    def on_test_begin(self, logs=None):
        # todo
        self.run.start_ml()

    def on_test_end(self, logs=None):
        self.test_epoch += 1
        loss = logs.get('loss')
        acc = logs.get('acc')
        acc = acc if acc else logs.get('accuracy')
        self.run.log_ml(epoch=self.test_epoch, loss=loss,
                        acc=acc, phase="test")
        # self.run.conclude()

    def on_train_end(self, logs):
        pass
        # upload/save to local
        # self.run.conclude()
