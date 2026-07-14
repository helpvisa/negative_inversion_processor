import sys
import traceback
from PySide6.QtCore import QRunnable, Slot, QObject, Signal


# custom worker class for handling multithreading
class WorkerSignals(QObject):
    """
    Signals from a running Worker thread.

    finished: int thread_id
    error: tuple (exctype, value, traceback.format_exc())
    result: object data returned from thread processing
    progress: tuple (thread_id, progress_value)
    """
    finished = Signal(int)
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(tuple)


class Worker(QRunnable):
    """
    Worker thread for processing images in parallel to UI actions.
    """
    def __init__(self, function, thread_id, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.thread_id = thread_id

    @Slot()
    def run(self):
        """
        Initialize the function passed in on thread creation, with args.
        """
        print(f"Thread started: {self.thread_id}", file=sys.stderr)
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            traceback.print_exc()
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit(self.thread_id)
            print(f"Thread ended: {self.thread_id}", file=sys.stderr)



# custom class for handling the image processing pipeline
# should tackle image processing in steps, saving intermediate images for each step
# source_image > pre-inversion wb > wb_image > shift+scale > wb_image > user wb > output
