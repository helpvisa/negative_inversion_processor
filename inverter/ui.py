import sys
import numpy as np
from PySide6.QtCore import Qt, Slot, QThreadPool
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog,
                               QGraphicsScene, QMessageBox, QSplitter,
                               QCheckBox, QGroupBox)
from scipy import ndimage
import parse_cli_arguments
from ui_classes import Worker
from custom_widgets import ImageView, LabeledSlider, ColorPicker
from inverter import load_raw_image, process_negative
from colour_management import convert_to_sRGB
from processing import rotate_image, process_all_adjustments
from globals import GLOBAL_FLAGS


# some global variables for tracking information about the current session
MAX_PREVIEW_SIZE = 1280
LOADED_RAW_PATH = None
CURRENT_IMAGE_SOURCE_DATA = []


#--- create a thread pool for offloading image processing
THREADPOOL = QThreadPool()
THREAD_COUNT = THREADPOOL.maxThreadCount()
print(f"Multithreading active with {THREAD_COUNT} threads.",
      file=sys.stderr)
# hold threads to prevent garbage collection
# and initialize a thread_id iterator
ACTIVE_THREADS = {}
THREAD_ID = 0


def instantiate_thread(func, *args, **kwargs):
    global THREAD_ID
    THREAD_ID += 1
    thread = Worker(func, thread_id=THREAD_ID)
    return thread


def remove_thread(thread_id):
    global ACTIVE_THREADS
    ACTIVE_THREADS[thread_id] = None


#--- UI Definitions
# derive from QWidget to create a custom updateable image class
class PrimaryImageView(QWidget):
    def __init__(self, parent=None):
        super(PrimaryImageView, self).__init__(parent)
        self.setMinimumSize(600, 400)

        # configure QGraphicsScene and ImageView
        self.layout = QHBoxLayout()
        self.scene = QGraphicsScene(0, 0, 1200, 800)
        self.view = ImageView(self.scene)
        # configure preview image itself
        self.preview_pixmap = self.scene.addPixmap(QPixmap(1200, 800))
        self.preview_pixmap.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.preview_pixmap.setPos(0, 0)
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)

    @Slot()
    def update_image(self):
        """
        Modify numpy pixel data array and update the GUI to display the new
        image
        """
        display_image, sRGB_profile = convert_to_sRGB(self.image_array)
        display_image = np.clip(display_image, a_min=0, a_max=1)
        q_image = QImage(np.multiply(display_image, 255).astype(np.uint8),
                         display_image.shape[1],
                         display_image.shape[0],
                         3 * display_image.shape[1],
                         QImage.Format.Format_RGB888)
        # convert to QPixmap for display
        new_pixmap = QPixmap.fromImage(q_image)
        self.preview_pixmap.setPixmap(new_pixmap)
        # resize view to match image
        self.scene.setSceneRect(self.preview_pixmap.boundingRect())
        self.view.centerOn(self.preview_pixmap)


class ToolPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #--- top-level tools layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        #--- pre-inversion layout (orientation and initial white balance)
        self.pre_inversion_groupbox = QGroupBox("Pre-Inversion")
        self.pre_inversion_layout = QVBoxLayout()
        self.pre_inv_orientation_layout = QHBoxLayout()
        self.pre_inv_rotate_left = QPushButton("Rotate Left")
        self.pre_inv_rotate_right = QPushButton("Rotate Right")
        self.pre_inv_orientation_layout.addWidget(self.pre_inv_rotate_left)
        self.pre_inv_orientation_layout.addWidget(self.pre_inv_rotate_right)
        self.pre_inv_wb_picker = ColorPicker("Dmin - Base Color")
        self.pre_inversion_layout.addLayout(self.pre_inv_orientation_layout)
        self.pre_inversion_layout.addWidget(self.pre_inv_wb_picker)
        self.pre_inversion_groupbox.setLayout(self.pre_inversion_layout)
        #--- inversion tools layout
        self.inversion_groupbox = QGroupBox("Inversion")
        self.inversion_layout = QVBoxLayout()
        self.skip_inversion_checkbox = QCheckBox("Skip Inversion")
        self.skip_inversion_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.red_ratio_slider = LabeledSlider("Red Ratio",
                                              0.0, 3.0, 1.36, 300,
                                              "#ffcccc")
        self.blue_ratio_slider = LabeledSlider("Blue Ratio",
                                               0.0, 3.0, 0.86, 300,
                                               "#ccccff")
        self.contrast_slider = LabeledSlider("Contrast",
                                             0.0, 5.0, 1.5, 500)
        self.lo_neutral_picker = ColorPicker("Low Density")
        self.hi_neutral_picker = ColorPicker("High Density")
        self.inversion_layout.addWidget(self.skip_inversion_checkbox)
        self.inversion_layout.addWidget(self.red_ratio_slider)
        self.inversion_layout.addWidget(self.blue_ratio_slider)
        self.inversion_layout.addWidget(self.contrast_slider)
        density_picker_layout = QHBoxLayout()
        density_picker_layout.addWidget(self.lo_neutral_picker)
        density_picker_layout.addWidget(self.hi_neutral_picker)
        self.inversion_layout.addLayout(density_picker_layout)
        self.inversion_groupbox.setLayout(self.inversion_layout)
        #--- user grading
        self.custom_grading_groupbox = QGroupBox("Grading")
        self.custom_grading_layout = QVBoxLayout()
        self.grading_gain_layout = QHBoxLayout()
        self.red_gain_slider = LabeledSlider("Red Gain",
                                             0.0, 2.0, 1.0, 1000,
                                             "#ff0000")
        self.green_gain_slider = LabeledSlider("Green Gain",
                                               0.0, 2.0, 1.0, 1000,
                                               "#00ff00")
        self.blue_gain_slider = LabeledSlider("Blue Gain",
                                              0.0, 2.0, 1.0, 1000,
                                              "#0000ff")
        self.grading_gain_layout.addWidget(self.red_gain_slider)
        self.grading_gain_layout.addWidget(self.green_gain_slider)
        self.grading_gain_layout.addWidget(self.blue_gain_slider)
        self.grading_tune_layout = QHBoxLayout()
        self.red_tune_slider = LabeledSlider("Red Tune",
                                             -1.0, 1.0, 0.0, 1000,
                                             "#ffeeee")
        self.green_tune_slider = LabeledSlider("Green Tune",
                                               -1.0, 1.0, 0.0, 1000,
                                               "#eeffee")
        self.blue_tune_slider = LabeledSlider("Blue Tune",
                                              -1.0, 1.0, 0.0, 1000,
                                              "#eeeeff")
        self.grading_tune_layout.addWidget(self.red_tune_slider)
        self.grading_tune_layout.addWidget(self.green_tune_slider)
        self.grading_tune_layout.addWidget(self.blue_tune_slider)
        self.grading_wb_picker = ColorPicker("White Balance Point")
        self.custom_grading_layout.addLayout(self.grading_gain_layout)
        self.custom_grading_layout.addLayout(self.grading_tune_layout)
        self.custom_grading_layout.addWidget(self.grading_wb_picker)
        self.custom_grading_groupbox.setLayout(self.custom_grading_layout)
        #--- add all layouts
        self.layout.addWidget(self.pre_inversion_groupbox)
        self.layout.addWidget(self.inversion_groupbox)
        self.layout.addWidget(self.custom_grading_groupbox)
        self.setLayout(self.layout)


class EditingDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # internal tracking vars
        self.preview_image = []

        # image preview / canvas
        self.image_preview = PrimaryImageView(self)
        # controls
        # demo controls
        self.current_file_label = QLabel("NO FILE LOADED")
        self.load_button = QPushButton("Load Image")
        # actual side panel
        self.tool_panel = ToolPanel()

        # define layouts
        # top-level layout
        self.layout = QVBoxLayout()
        # file layout
        self.load_layout = QHBoxLayout()
        self.load_layout.addWidget(self.current_file_label)
        self.load_layout.addWidget(self.load_button)
        # editing layout
        self.editing_layout = QSplitter(Qt.Horizontal)
        self.editing_layout.setHandleWidth(16)
        self.editing_layout.addWidget(self.image_preview)
        self.editing_layout.addWidget(self.tool_panel)
        # add all layouts and wrap it all up
        self.layout.addLayout(self.load_layout)
        self.layout.addWidget(self.editing_layout)
        self.setLayout(self.layout)

        # wiring up functions
        # include some quick shorthand variables for readability
        tp = self.tool_panel
        ip = self.image_preview
        self.load_button.clicked.connect(self.load_raw_file)
        tp.pre_inv_rotate_left.clicked.connect(lambda: self.rotate_image(1))
        tp.pre_inv_rotate_right.clicked.connect(lambda: self.rotate_image(-1))
        # this is super weird and fragile with many edge cases
        # maybe instead: enable pick mode with signal, but track active picker
        #                by letting picker pass reference to itself with signal?
        #                could `self.sender` be a solution?
        # allow pickers to trigger picker mode
        tp.pre_inv_wb_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.lo_neutral_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.hi_neutral_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.grading_wb_picker.pickRequested.connect(ip.view.enable_pick_mode)
        # allow view to send values back
        ip.view.pointPicked.connect(tp.pre_inv_wb_picker.finish_pick)
        ip.view.pointPicked.connect(tp.lo_neutral_picker.finish_pick)
        ip.view.pointPicked.connect(tp.hi_neutral_picker.finish_pick)
        ip.view.pointPicked.connect(tp.grading_wb_picker.finish_pick)

    def load_raw_file(self):
        # load raw image and resize it for the preview pane
        image_path, discard_text = QFileDialog.getOpenFileName()
        if image_path:
            global THREADPOOL
            global ACTIVE_THREADS
            global LOADED_RAW_PATH
            LOADED_RAW_PATH = image_path
            # disable button while we load the new image
            self.load_button.setEnabled(False)
            self.current_file_label.setText("Loading your image...")
            # the function to be executed within a separate thread
            def init_func():
                raw = load_raw_image(image_path).astype(np.float32) / 65535.0
                height, width, _ = raw.shape
                # preview size maxes out at 1920 for easier processing
                preview_scale = MAX_PREVIEW_SIZE / width
                raw = ndimage.zoom(raw, (preview_scale, preview_scale, 1), order=0)
                return raw
            # the function to be executed upon thread completion
            def post_func(image_data):
                global CURRENT_IMAGE_SOURCE_DATA
                CURRENT_IMAGE_SOURCE_DATA = image_data
                self.preview_image = image_data
                self.image_preview.image_array = self.preview_image
                self.image_preview.update_image()
                self.current_file_label.setText(LOADED_RAW_PATH)
                self.load_button.setEnabled(True)
            def error_func(e):
                exctype, value, error = e
                # tell user there's an issue
                QMessageBox.critical(self, "Error processing image!", error)
                # and re-enable the buttons
                self.current_file_label.setText("NO FILE LOADED")
                self.load_button.setEnabled(True)
            # instantiate and run a thread
            # should split this out into its own function, surely
            thread = instantiate_thread(init_func)
            ACTIVE_THREADS[thread.thread_id] = thread
            thread.signals.result.connect(post_func)
            thread.signals.error.connect(error_func)
            thread.signals.finished.connect(remove_thread)
            THREADPOOL.start(thread)

    def rotate_image(self, direction):
        global ACTIVE_THREADS
        global THREADPOOL
        def init_func():
            self.preview_image = rotate_image(self.preview_image, direction)
        def post_func():
            self.image_preview.image_array = self.preview_image
            self.image_preview.update_image()
        thread = instantiate_thread(init_func)
        ACTIVE_THREADS[thread.thread_id] = thread
        thread.signals.result.connect(post_func)
        THREADPOOL.start(thread)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Negative Inversion Processor")
        self.editing_display = EditingDisplay()
        self.setCentralWidget(self.editing_display)
        self.resize(1600, 800)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # what display server? do we support mouse warp?
    GLOBAL_FLAGS["platform"] = app.platformName()
    print(f"Running on {GLOBAL_FLAGS['platform']} platform",
          file=sys.stderr)
    # load custom QSS stylesheet
    with open("inverter/styles.qss", "r") as qss:
        _styles = qss.read()
        app.setStyleSheet(_styles)
    # set a global palette (should do this in stylesheet? enh)
    palette = QPalette(QColor(4,   4,   4  ),  # windowText
                       QColor(128, 128, 128),  # window
                       QColor(222, 222, 222),  # light
                       QColor(8,   8,   8  ),  # dark
                       QColor(128, 128, 128),  # mid
                       QColor(4,   4,   4  ),  # text
                       QColor(192, 192, 192))  # base
    app.setPalette(palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
