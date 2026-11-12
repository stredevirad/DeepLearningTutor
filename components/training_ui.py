"""
Live Training Display
=====================
A Keras callback that streams training progress into Streamlit placeholders:
a progress bar (updated within long epochs too) and live loss / accuracy line
charts that grow as epochs finish.

Redraws are throttled: a small table dataset such as Iris finishes an epoch
every few milliseconds, and redrawing two charts after every one of them would
cost more than the training itself. The final state is always drawn.
"""

import time

import pandas as pd
import tensorflow as tf

SERIES_COLORS = ["#4F46E5", "#F97316"]   # training, validation (same as the result charts)


class LiveTrainingCallback(tf.keras.callbacks.Callback):
    """Updates `progress_bar`, `loss_chart` and `acc_chart` (st.empty() placeholders) during fit()."""

    def __init__(self, total_epochs, progress_bar, loss_chart, acc_chart, labels,
                 min_batch_interval=0.5, min_epoch_interval=0.3):
        super().__init__()
        self.total_epochs = max(1, int(total_epochs))
        self.progress_bar = progress_bar
        self.loss_chart = loss_chart
        self.acc_chart = acc_chart
        self.labels = labels          # dict of translated strings
        self.rows = []
        self.steps_per_epoch = None
        self._epoch = 0
        self._last_batch_update = 0.0
        self._last_draw = 0.0
        self._drawn_rows = 0
        self.min_batch_interval = min_batch_interval
        self.min_epoch_interval = min_epoch_interval

    def on_train_begin(self, logs=None):
        self.steps_per_epoch = (self.params or {}).get("steps")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        # Keep the bar moving during long CNN epochs (throttled to avoid flooding the browser).
        now = time.monotonic()
        if not self.steps_per_epoch or now - self._last_batch_update < self.min_batch_interval:
            return
        self._last_batch_update = now
        frac = (self._epoch + (batch + 1) / self.steps_per_epoch) / self.total_epochs
        self.progress_bar.progress(min(frac, 1.0), text=self.labels["epoch_running"].format(
            epoch=self._epoch + 1, total=self.total_epochs))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.rows.append({
            "epoch": epoch + 1,
            self.labels["train_loss"]: float(logs.get("loss", float("nan"))),
            self.labels["val_loss"]: float(logs.get("val_loss", float("nan"))),
            self.labels["train_acc"]: float(logs.get("accuracy", float("nan"))),
            self.labels["val_acc"]: float(logs.get("val_accuracy", float("nan"))),
        })
        now = time.monotonic()
        if epoch + 1 >= self.total_epochs or now - self._last_draw >= self.min_epoch_interval:
            self._draw()

    def on_train_end(self, logs=None):
        if len(self.rows) != self._drawn_rows:      # e.g. early stopping ended between redraws
            self._draw()

    def _draw(self):
        self._last_draw = time.monotonic()
        self._drawn_rows = len(self.rows)
        self.render()
        last = self.rows[-1]
        self.progress_bar.progress(min(last["epoch"] / self.total_epochs, 1.0), text=self.labels["epoch_done"].format(
            epoch=last["epoch"], total=self.total_epochs, loss=last[self.labels["train_loss"]],
            acc=last[self.labels["val_acc"]]))

    def render(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows).set_index("epoch")
        self.loss_chart.line_chart(df[[self.labels["train_loss"], self.labels["val_loss"]]], color=SERIES_COLORS,
                                   x_label=self.labels["epoch_axis"], y_label=self.labels["loss_axis"], height=260)
        self.acc_chart.line_chart(df[[self.labels["train_acc"], self.labels["val_acc"]]], color=SERIES_COLORS,
                                  x_label=self.labels["epoch_axis"], y_label=self.labels["acc_axis"], height=260)

    def load_history(self, history):
        """Draw a complete history at once (used for remote MCP training progress)."""
        n = len(history.get("loss", []))
        self.rows = [{
            "epoch": i + 1,
            self.labels["train_loss"]: history.get("loss", [None] * n)[i],
            self.labels["val_loss"]: history.get("val_loss", [None] * n)[i],
            self.labels["train_acc"]: history.get("accuracy", [None] * n)[i],
            self.labels["val_acc"]: history.get("val_accuracy", [None] * n)[i],
        } for i in range(n)]
        self._drawn_rows = n
        self.render()
