# This file is part of Negative Inversion Processor.
#
# Negative Inversion Processor  is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# Negative Inversion Processor is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# Negative Inversion Processor. If not, see <https://www.gnu.org/licenses/>. 

import sys
import numpy as np
from enum import IntEnum
from scipy import ndimage
from PySide6.QtCore import Qt, QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
                               QGraphicsScene, QMainWindow, QApplication,
                               QFileDialog, QPushButton, QLabel, QSplitter,
                               QStatusBar, QListView)
from custom_widgets import ImageView, ColorPicker
from processing import (load_raw_image, apply_ffc, average_sample_point,
                        invert_to_density, to_density, density_to_luminance)
from threads import WorkerThreadPool
from colour_management import convert_to_sRGB
from global_vars import RAW_EXTENSIONS


class DensityMode(IntEnum):
    ALL   = 0
    RED   = 1
    GREEN = 2
    BLUE  = 3


class DensityStage(IntEnum):
    ANALYSIS = 0
    IMAGE    = 1


class DensitometryView(QWidget):
    sendMessage = Signal(str)

    def __init__(self, threadpool: WorkerThreadPool, parent=None):
        super(DensitometryView, self).__init__(parent)
        self.threadpool = threadpool
        self.layout = QHBoxLayout()
        self.scene = QGraphicsScene(0, 0, 1200, 800)
        self.view = ImageView(self.scene)
        self.d_pixmap = self.scene.addPixmap(QPixmap(1200, 800))
        self.d_pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.d_pixmap.setPos(0, 0)
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

    @Slot()
    def update_image(self, image_data):
        def current():
            print("Preparing preview...", file=sys.stderr)
            self.sendMessage.emit("Loading visual preview...")
            display_image = image_data.copy()
            if display_image.ndim < 3:
                display_image = np.stack((display_image,
                                          display_image,
                                          display_image), axis=-1)
            display_image = convert_to_sRGB(image_data)
            display_image = np.clip(image_data, a_min=0, a_max=1)
            q_image = QImage(np.multiply(display_image, 255).astype(np.uint8),
                             display_image.shape[1],
                             display_image.shape[0],
                             3 * display_image.shape[1],
                             QImage.Format.Format_RGB888)
            return q_image

        def finish(q_image):
            new_pixmap = QPixmap.fromImage(q_image)
            print("Preview prepared!", file=sys.stderr)
            self.sendMessage.emit("Visual preview prepared.")
            self.d_pixmap.setPixmap(new_pixmap)
            self.scene.setSceneRect(self.d_pixmap.boundingRect())
            self.view.centerOn(self.d_pixmap)

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(finish)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)


class Densitometer(QObject):
    imageChanged = Signal()
    referenceChanged = Signal()
    baseDensityChanged = Signal()
    analysisUpdated = Signal(np.ndarray)
    sendMessage = Signal(str)

    def __init__(self, threadpool: WorkerThreadPool):
        super().__init__()
        self.threadpool = threadpool
        self.preview_size = 256
        self.resize_scale = 5
        self.reference = None
        self.image = None
        self.base_density = None
        self.density_analysis = None
        self.density_mode = 

    def density_to_zone(self, density: np.ndarray):
        # zones from zone system, in order (0 - 10)
        zones = np.array([0.0, 0.08, 0.18, 0.35, 0.5, 0.65, 0.85, 1.05, 1.25, 1.45, 1.6])
        zone_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        zone_indices = np.clip(np.digitize(density, zones), a_min=0, a_max=10)
        return zone_vals[zone_indices]

    def analyse(self):
        def current():
            if self.image is not None:
                print("Analyzing densities...", file=sys.stderr)
                self.sendMessage.emit("Analyzing image density...")
                image_density = invert_to_density(self.image)
                ref_density = np.array([0, 0, 0]).reshape((1, 1, 3))
                if self.reference is not None:
                    ref_density = invert_to_density(self.reference)
                if self.base_density is not None:
                    ref_density += self.base_density
                net_density = image_density - ref_density
                self.density_analysis = net_density
                # self.density_visual = net_density / np.percentile(net_density, 98)
                self.density_visual = ndimage.zoom(self.density_to_zone(net_density),
                                                   (self.resize_scale,
                                                    self.resize_scale, 1),
                                                   order=0)
                self.analysisUpdated.emit(self.density_visual)
                print("Analysis updated!", file=sys.stderr)
                self.sendMessage.emit("Density analysis updated.")

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    @Slot()
    def set_reference(self, reference_data):
        self.reference = reference_data
        # scale down for faster and more purposeful analysis
        height, width, _ = self.reference.shape
        scale = self.preview_size / height
        if width > height:
            scale = self.preview_size / width
        self.reference = ndimage.zoom(self.reference,
                                      (scale, scale, 1),
                                      order=1)
        self.referenceChanged.emit()
        self.sendMessage.emit("Updated illuminant reference image.")

    @Slot()
    def clear_reference(self):
        self.reference = None
        self.referenceChanged.emit()
        self.sendMessage.emit("Updated illuminant reference image.")

    @Slot()
    def set_image(self, image_data):
        self.image = image_data
        # scale down for faster and more purposeful analysis
        height, width, _ = self.image.shape
        scale = self.preview_size / height
        if width > height:
            scale = self.preview_size / width
        self.image = ndimage.zoom(self.image,
                                      (scale, scale, 1),
                                      order=1)
        self.imageChanged.emit()
        self.sendMessage.emit("Updated analysis image.")

    @Slot()
    def pick_density(self, point_x, point_y, stage, picker: ColorPicker):
        # we need to reincorporate stage here
        # Stage = ANALYSIS for checking density
        # Stage = SET for setting reference base density
        value = average_sample_point(self.density_analysis,
                                     int(point_x / self.resize_scale),
                                     int(point_y / self.resize_scale),
                                     16)
        picker.update_color(value)

    @Slot()
    def set_base_density(self, color):
        # reshape (H, W, nDim) to subtract single val against RGB array
        self.base_density = np.array([color[0],
                                      color[1],
                                      color[2]]).reshape((1, 1, 3))
        self.baseDensityChanged.emit()
        self.sendMessage.emit("Update base density.")

    @Slot()
    def clear_base_density(self):
        self.base_density = None
        self.baseDensityChanged.emit()
        self.sendMessage.emit("Cleared base density.")


class AnalysisDisplay(QWidget):
    def __init__(self, threadpool: WorkerThreadPool, parent=None):
        super().__init__(parent)
        # initialize processer
        self.densitometer = Densitometer(threadpool)
        self.densitometer.imageChanged.connect(self.densitometer.analyse)
        self.densitometer.referenceChanged.connect(self.densitometer.analyse)
        self.densitometer.baseDensityChanged.connect(self.densitometer.analyse)
        # setup layouts
        self.view = DensitometryView(threadpool)
        self.load_reference_button = QPushButton("Load Reference Image")
        self.load_image_button = QPushButton("Load Image to Analyse")
        self.base_density_picker = ColorPicker("Set Base Density")
        self.clear_base_density_button = QPushButton("Clear Base Density")
        self.density_analysis_picker = ColorPicker("Check Density at Point")
        self.current_density = QLabel("No density selected.")
        self.layout = QVBoxLayout()
        self.control_panel = QWidget()
        self.control_panel_layout = QVBoxLayout()
        self.control_panel_layout.addWidget(self.load_reference_button)
        self.control_panel_layout.addWidget(self.load_image_button)
        self.control_panel_layout.addWidget(self.base_density_picker)
        self.control_panel_layout.addWidget(self.clear_base_density_button)
        self.control_panel_layout.addWidget(self.density_analysis_picker)
        self.control_panel_layout.addWidget(self.current_density)
        self.control_panel_layout.addStretch()
        self.control_panel.setLayout(self.control_panel_layout)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(16)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(self.control_panel)
        self.layout.addWidget(self.splitter)
        self.setLayout(self.layout)
        # wire up functionality
        # self.view.view... why did i do this
        self.load_reference_button.clicked.connect(self.load_reference)
        self.clear_base_density_button.clicked.connect(self.densitometer.clear_base_density)
        self.load_image_button.clicked.connect(self.load_analysis_target)
        self.density_analysis_picker.pickRequested.connect(self.view.view.enable_pick_mode)
        self.base_density_picker.pickRequested.connect(self.view.view.enable_pick_mode)
        self.density_analysis_picker.valueChanged.connect(self.densitometer.pick_density)
        self.base_density_picker.valueChanged.connect(self.densitometer.pick_density)
        self.density_analysis_picker.colorChanged.connect(self.update_current_density)
        self.base_density_picker.colorChanged.connect(self.densitometer.set_base_density)
        self.view.view.pointPicked.connect(self.density_analysis_picker.finish_pick)
        self.view.view.pointPicked.connect(self.base_density_picker.finish_pick)
        self.densitometer.analysisUpdated.connect(self.view.update_image)

    def open_load_dialog(self):
        filter_string = "Camera RAW ("
        filter_string += " ".join(f"*.{e}" for e in RAW_EXTENSIONS)
        filter_string += ");; All Files (*)"
        image_path, _ = QFileDialog.getOpenFileName(
            None,
            caption="Select RAW File",
            filter=filter_string
        )
        return image_path

    def load_reference(self):
        path = self.open_load_dialog()
        if path:
            reference = load_raw_image(path)
            self.densitometer.set_reference(reference)
        else:
            self.densitometer.clear_reference()

    def load_analysis_target(self):
        path = self.open_load_dialog()
        if path:
            image = load_raw_image(path)
            self.densitometer.set_image(image)

    def update_current_density(self, color):
        string = f"Density at point: R{color[0]:.3f} | G{color[1]:.3f} | B{color[2]:.3f}"
        self.current_density.setText(string)


class MainWindow(QMainWindow):
    def __init__(self, threadpool: WorkerThreadPool, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("NIP Densitometer")
        self.analysis_display = AnalysisDisplay(threadpool)
        self.status_bar = self.statusBar()
        self.analysis_display.view.sendMessage.connect(self.push_message)
        self.analysis_display.densitometer.sendMessage.connect(self.push_message)
        self.push_message("Ready.")
        self.setCentralWidget(self.analysis_display)
        self.resize(1200, 600)

    @Slot()
    def push_message(self, message):
        self.status_bar.showMessage(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open("inverter/styles.qss", "r") as qss:
        _styles = qss.read()
        app.setStyleSheet(_styles)
    palette = QPalette(QColor(4,   4,   4  ),  # windowText
                       QColor(128, 128, 128),  # window
                       QColor(222, 222, 222),  # light
                       QColor(8,   8,   8  ),  # dark
                       QColor(128, 128, 128),  # mid
                       QColor(4,   4,   4  ),  # text
                       QColor(192, 192, 192))  # base
    app.setPalette(palette)
    threadpool = WorkerThreadPool()
    window = MainWindow(threadpool)
    window.show()
    print("Waiting for all threads to finish!", file=sys.stderr)
    threadpool.waitForDone()
    print("All threads finished; exiting!", file=sys.stderr)
    exit_code = app.exec()
    sys.exit(exit_code)
