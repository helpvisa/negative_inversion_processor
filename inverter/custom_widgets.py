from PySide6.QtCore import (Qt, Slot, Signal, QPoint, QTimer,
                            QPointF, QRectF)
from PySide6.QtGui import QColor, QBrush, QPainter, QCursor
from PySide6.QtWidgets import (QApplication, QWidget, QGraphicsView,
                               QHBoxLayout, QVBoxLayout,
                               QLabel, QInputDialog, QPushButton,
                               QSpacerItem, QSizePolicy, QFrame)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from globals import GLOBAL_FLAGS


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
        self._pick_mode = False
        # configure view properties
        self.setViewport(QOpenGLWidget())
        self.setBackgroundBrush(QBrush(QColor(128,128,128)))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)

    @Slot()
    def enable_pick_mode(self):
        self.set_pick_mode(True)

    @Slot()
    def disable_pick_mode(self):
        self.set_pick_mode(False)
    
    def set_pick_mode(self, enabled: bool):
        self._pick_mode = enabled
        if self._pick_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def mousePressEvent(self, event):
        # intercept clicks if we're in pick mode
        if self._pick_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self.set_pick_mode(False)
                scene_pos = self.mapToScene(event.pos())
                self.pointPicked.emit(scene_pos.x(), scene_pos.y())
            else:
                self.set_pick_mode(False)
                self.pointPicked.emit(None, None)
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


class Scrubber(QWidget):
    """
    Fine-grained scrubber-style slider control for careful colour grading.
    """
    valueChanged = Signal(int)
    rightClicked = Signal()

    def __init__(self, minimum, maximum, value,
                 fill="#C0C0C0", bg="#404040", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(16)
        self.setMaximumHeight(32)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self._min = minimum
        self._max = maximum
        self._value = value
        self._default = value
        self._last_mouse_x = None
        # used to slightly delay position updates to allow mouse to fully move
        self._timer = QTimer(self)
        # color values
        self._fill_color = fill
        self._background_color = bg

    def value(self):
        return self._value

    def get_min_value(self):
        return self._min

    def get_max_value(self):
        return self._max

    def setValue(self, value):
        self._value = max(self._min, min(self._max, value))
        self.update()

    def set_last_mouse_x(self, x):
        self._last_mouse_x = x

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # hide and teleport mouse to center of widget
            if GLOBAL_FLAGS['platform'] != "wayland":
                self.setCursor(Qt.CursorShape.BlankCursor)
                center_x = self.width() // 2
                center_y = self.height() // 2
                local_center = QPoint(center_x, center_y)
                global_center = self.mapToGlobal(local_center)
                QCursor.setPos(global_center)
                QApplication.processEvents()
                self._timer.timeout.connect(lambda: self.set_last_mouse_x(global_center.x()))
                self._timer.start(1)
            else:
                self.set_last_mouse_x(event.globalPosition().x())
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit()

    def mouseMoveEvent(self, event):
        if self._last_mouse_x is not None:
            delta_x = event.globalPosition().x() - self._last_mouse_x
            # teleport mouse to center of widget
            if GLOBAL_FLAGS['platform'] != 'wayland':
                center_x = self.width() // 2
                center_y = self.height() // 2
                local_center = QPoint(center_x, center_y)
                global_center = self.mapToGlobal(local_center)
                QCursor.setPos(global_center)
                x_pos = event.globalPosition().x()
                self._timer.timeout.connect(lambda: self.set_last_mouse_x(x_pos))
                self._timer.start(50)
            else:
                self.set_last_mouse_x(event.globalPosition().x())

            speed = 1.0
            modifiers = QApplication.keyboardModifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                speed = 5.0
            elif modifiers & Qt.KeyboardModifier.ControlModifier:
                speed = 0.2
            new_value = self._value + delta_x * speed
            new_value = max(self._min, min(self._max, new_value))
            if new_value != self._value:
                self._value = new_value
                self.valueChanged.emit(int(self._value))
                self.update()

    def wheelEvent(self, event):
        speed = 10.0
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            speed = 50.0
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            speed = 1.0
        if event.angleDelta().y() < 0:
            speed *= -1
        new_value = self._value + speed
        new_value = max(self._min, min(self._max, new_value))
        if new_value != self._value:
            self._value = new_value
            self.valueChanged.emit(int(self._value))
            self.update()

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        # delay also required here, to avoid overwrite from move event
        if GLOBAL_FLAGS['platform'] != 'wayland':
            self._timer.timeout.connect(lambda: self.set_last_mouse_x(None))
            self._timer.start(1)
        else:
            self.set_last_mouse_x(None)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._value != self._default:
                self._value = self._default
                self.valueChanged.emit(int(self._value))
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self._background_color))
        fill = 0
        if self._max > self._min:
            fill = (self._value - self._min) / (self._max - self._min)
        painter.fillRect(0, 0,
                         int(self.width() * fill), self.height(),
                         QColor(self._fill_color))


class LabeledSlider(QWidget):
    """
    Slider with text label and numeric readout.
    """
    valueChanged = Signal(float)

    def __init__(self, label,
                 minimum, maximum, value, precision=100,
                 fill="#C0C0C0", bg="#404040", parent=None):
        super().__init__(parent)
        self.precision = precision

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 8)
        header = QHBoxLayout()
        self.label = QLabel(label)
        self.readout = QLabel(f"{value:.2f}")
        header.addWidget(self.label)
        # adding a stretch ensures label <-> readout are hard left + right
        header.addStretch()
        header.addWidget(self.readout)

        self.slider = Scrubber(int(minimum * precision),
                               int(maximum * precision),
                               int(value * precision),
                               fill, bg)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.rightClicked.connect(self._on_right_click)

        layout.addLayout(header)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def _on_right_click(self):
        current_float = self.value()
        min_float = self.get_min_value()
        max_float = self.get_max_value()
        value, ok = QInputDialog.getDouble(self, "Set Value", "New value:",
                                           current_float, min_float, max_float,
                                           decimals=4)
        if ok:
            self.setValue(value)
            self.valueChanged.emit(value)

    def _on_slider_changed(self, raw_value):
        value = raw_value / self.precision
        self.readout.setText(f"{value:.2f}")
        self.valueChanged.emit(value)

    def value(self):
        return self.slider.value() / self.precision

    def get_min_value(self):
        return self.slider.get_min_value() / self.precision

    def get_max_value(self):
        return self.slider.get_max_value() / self.precision

    def setValue(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self.precision))
        self.readout.setText(f"{value:.2f}")
        self.slider.blockSignals(False)


class ColorPicker(QWidget):
    """
    Pick a color from the preview image.
    """
    pickRequested = Signal()

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._point: QPoint = QPoint(0, 0)
        self._color: tuple[float, float, float] = None

        main_layout = QVBoxLayout()
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        v_layout = QVBoxLayout(frame)
        v_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_layout = QHBoxLayout()
        h_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button = QPushButton("Pick")
        self.button.setCheckable(True)
        self.button.clicked.connect(self._on_clicked)
        self.label = QLabel(f"<i>{label}</i>")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_display = QLabel("None")
        h_layout.addWidget(self.button)
        h_layout.addItem(QSpacerItem(8, 32,
                                   QSizePolicy.Policy.Minimum,
                                   QSizePolicy.Policy.Minimum))
        h_layout.addWidget(self.value_display)
        v_layout.addWidget(self.label)
        v_layout.addLayout(h_layout)
        main_layout.addWidget(frame)
        self.setLayout(main_layout)

    def _on_clicked(self):
        if self.button.isChecked():
            self.button.setText("...")
            self.pickRequested.emit()
        else:
            self.button.setText("Pick")

    @Slot()
    def finish_pick(self, point_x: float = None, point_y: float = None):
        # should only receive a value if it's waiting for one
        if self.button.isChecked():
            self.button.setChecked(False)
            self.button.setText("Pick")
            if point_x and point_y:
                self._point = QPoint(point_x, point_y)
                self.value_display.setText(f"({self._point.x()}, {self._point.y()})")
