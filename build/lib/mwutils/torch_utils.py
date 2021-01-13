class LoggerHook:
    def set_run(self, run, phase="train"):
        self.run = run
        self.phase = phase
        run.init_ml()
        run.start_ml()
        self._step = -1

    def conclude_run(self):
        self.run.conclude()
        self._step = -1

    def torch_loss_hook(self, model, input, loss):
        loss_val = -1
        if isinstance(loss.item(), list):
            loss_val = loss.item()[0]
        else:
            loss_val = loss.item()
        self._step += 1
        self.run.log_ml(step=self._step, loss=loss_val, phase=self.phase)
