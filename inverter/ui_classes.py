import sys
import traceback
from PySide6.QtCore import (QRunnable, Slot, QObject, Signal,
                            Qt, QPoint)
from PySide6.QtWidgets import QGraphicsView
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QColor, QBrush


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


# custom QGraphicsView class to implement new controls
class ImageView(QGraphicsView):
    """
    Custom QGraphicsView that incorporates color picker and zooming.

    When "pick_mode" is enabled, the pointPicked signal is emitted if the user
    left-clicks anywhere within the view. Pressing any other button exits
    "pick_mode".
    """
    pointPicked = Signal(float, float)

    def __init__(self, parent=None):
        super(ImageView, self).__init__(parent)
        self.last_mouse_pos = (0, 0)
        # configure view properties
        self.setViewport(QOpenGLWidget())
        self.setBackgroundBrush(QBrush(QColor(128,128,128)))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)

    def set_pick_mode(self, enabled: bool):
        self._pick_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def mousePressEvent(self, event):
        # intercept clicks if we're in pick mode
        if self._pick_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.mapToScene(event.pos())
                self.set_pick_mode(False)
                self.pointPicked.emit(scene_pos.x(), scene_pos.y())
            else:
                self.set_pick_mode(False)
        event.accept()
        return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        # zoom QGraphicsView in and out
        angle = event.angleDelta().y()
        scale_factor = 1.1
        if (angle < 0):
            scale_factor = 1.0 / scale_factor
        self.scale(scale_factor, scale_factor)

# custom class for handling the image processing pipeline
# should tackle image processing in steps, saving intermediate images for each step
# source_image > pre-inversion wb > wb_image > shift+scale > wb_image > user wb > output
