import sys
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog,
                               QGraphicsScene, QSplitter,
                               QCheckBox, QGroupBox, QScrollArea)
from custom_widgets import ImageView, LabeledSlider, ColorPicker
from colour_management import convert_to_sRGB
from edit_params import EditParams, Stage
from pipeline import ProcessingPipeline
from global_vars import GLOBAL_FLAGS


# some global variables for tracking information about the current session
LOADED_RAW_PATH = None
CURRENT_IMAGE_SOURCE_DATA = []
PIPELINE: ProcessingPipeline = ProcessingPipeline()


# --- UI Definitions
# derive from QWidget to create a custom updateable image class
class PrimaryImageView(QWidget):
    def __init__(self, parent=None):
        global PIPELINE
        super(PrimaryImageView, self).__init__(parent)
        self.setMinimumSize(300, 200)
        PIPELINE.previewUpdated.connect(self.update_image)

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
    def update_image(self, center=False):
        """
        Modify numpy pixel data array and update the GUI to display the new
        image
        """
        global PIPELINE

        def process():
            preview_image = PIPELINE.final_preview.copy()
            if preview_image.ndim < 3:
                preview_image = np.stack((preview_image,
                                          preview_image,
                                          preview_image), axis=-1)
            display_image, _ = convert_to_sRGB(preview_image)
            display_image = np.clip(display_image, a_min=0, a_max=1)
            q_image = QImage(np.multiply(display_image, 255).astype(np.uint8),
                             display_image.shape[1],
                             display_image.shape[0],
                             3 * display_image.shape[1],
                             QImage.Format.Format_RGB888)
            return q_image

        def finish(q_image):
            # convert to QPixmap for display
            new_pixmap = QPixmap.fromImage(q_image)
            self.preview_pixmap.setPixmap(new_pixmap)
            # resize view to match image
            self.scene.setSceneRect(self.preview_pixmap.boundingRect())
            if center:
                self.view.centerOn(self.preview_pixmap)
        thread = PIPELINE.threadpool.instantiate_thread(process)
        PIPELINE.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(finish)
        thread.signals.finished.connect(PIPELINE.threadpool.remove_thread)
        PIPELINE.threadpool.start(thread)


class ToolPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        # track all the active tools
        self.tools = []
        # --- top-level tools layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # --- pre-inversion layout (orientation and initial white balance)
        pre_inversion_groupbox = QGroupBox("Pre-Inversion")
        pre_inversion_layout = QVBoxLayout()
        self.pre_inv_orientation_layout = QHBoxLayout()
        self.pre_inv_rotate_left = QPushButton("Rotate Left")
        self.pre_inv_rotate_right = QPushButton("Rotate Right")
        self.pre_inv_orientation_layout.addWidget(self.pre_inv_rotate_left)
        self.pre_inv_orientation_layout.addWidget(self.pre_inv_rotate_right)
        self.pre_inv_wb_picker = ColorPicker("Dmin - Base Color", Stage.PRE_INV)
        pre_inversion_layout.addLayout(self.pre_inv_orientation_layout)
        pre_inversion_layout.addWidget(self.pre_inv_wb_picker)
        pre_inversion_groupbox.setLayout(pre_inversion_layout)
        self.tools.extend([self.pre_inv_rotate_left,
                           self.pre_inv_rotate_right,
                           self.pre_inv_wb_picker])
        # --- inversion tools layout
        inversion_groupbox = QGroupBox("Inversion")
        inversion_layout = QVBoxLayout()
        checkbox_layout = QHBoxLayout()
        self.skip_inversion_checkbox = QCheckBox("Skip Inversion")
        self.skip_inversion_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.bw_checkbox = QCheckBox("Black and White")
        self.bw_checkbox.setCheckState(Qt.CheckState.Unchecked)
        checkbox_layout.addWidget(self.skip_inversion_checkbox)
        checkbox_layout.addWidget(self.bw_checkbox)
        self.red_ratio_slider = LabeledSlider("Red Ratio",
                                              0.0, 3.0, 1.36, 300,
                                              "#ffcccc")
        self.blue_ratio_slider = LabeledSlider("Blue Ratio",
                                               0.0, 3.0, 0.86, 300,
                                               "#ccccff")
        self.contrast_slider = LabeledSlider("Contrast",
                                             0.0, 5.0, 1.5, 500)
        self.lo_gray_picker = ColorPicker("Low Gray", Stage.INV)
        self.hi_gray_picker = ColorPicker("High Gray", Stage.INV)
        inversion_layout.addLayout(checkbox_layout)
        inversion_layout.addWidget(self.red_ratio_slider)
        inversion_layout.addWidget(self.blue_ratio_slider)
        inversion_layout.addWidget(self.contrast_slider)
        density_picker_layout = QHBoxLayout()
        density_picker_layout.addWidget(self.lo_gray_picker)
        density_picker_layout.addWidget(self.hi_gray_picker)
        inversion_layout.addLayout(density_picker_layout)
        inversion_groupbox.setLayout(inversion_layout)
        self.tools.extend([self.skip_inversion_checkbox,
                           self.bw_checkbox,
                           self.red_ratio_slider,
                           self.blue_ratio_slider,
                           self.contrast_slider,
                           self.lo_gray_picker,
                           self.hi_gray_picker])
        # --- user grading
        custom_grading_groupbox = QGroupBox("Grading")
        custom_grading_layout = QVBoxLayout()
        grading_gain_layout = QHBoxLayout()
        self.red_gain_slider = LabeledSlider("R Mult",
                                             0.0, 2.0, 1.0, 1000,
                                             "#ff0000")
        self.green_gain_slider = LabeledSlider("G Mult",
                                               0.0, 2.0, 1.0, 1000,
                                               "#00ff00")
        self.blue_gain_slider = LabeledSlider("B Mult",
                                              0.0, 2.0, 1.0, 1000,
                                              "#0000ff")
        grading_gain_layout.addWidget(self.red_gain_slider)
        grading_gain_layout.addWidget(self.green_gain_slider)
        grading_gain_layout.addWidget(self.blue_gain_slider)
        grading_tune_layout = QHBoxLayout()
        self.red_tune_slider = LabeledSlider("R Add",
                                             -1.0, 1.0, 0.0, 1000,
                                             "#ffeeee")
        self.green_tune_slider = LabeledSlider("G Add",
                                               -1.0, 1.0, 0.0, 1000,
                                               "#eeffee")
        self.blue_tune_slider = LabeledSlider("B Add",
                                              -1.0, 1.0, 0.0, 1000,
                                              "#eeeeff")
        grading_tune_layout.addWidget(self.red_tune_slider)
        grading_tune_layout.addWidget(self.green_tune_slider)
        grading_tune_layout.addWidget(self.blue_tune_slider)
        self.grading_wb_picker = ColorPicker("Pick White Balance", Stage.GRADE,
                                             hide_value=True)
        self.exposure_slider = LabeledSlider("Exposure Compensation",
                                             0.0, 10.0, 1.0, 1000)
        self.tonemap_checkbox = QCheckBox("Apply Tonemapping")
        custom_grading_layout.addLayout(grading_gain_layout)
        custom_grading_layout.addLayout(grading_tune_layout)
        custom_grading_layout.addWidget(self.grading_wb_picker)
        custom_grading_layout.addWidget(self.exposure_slider)
        custom_grading_layout.addWidget(self.tonemap_checkbox)
        custom_grading_groupbox.setLayout(custom_grading_layout)
        self.tools.extend([self.red_gain_slider,
                           self.green_gain_slider,
                           self.blue_gain_slider,
                           self.red_tune_slider,
                           self.green_tune_slider,
                           self.blue_tune_slider,
                           self.grading_wb_picker,
                           self.exposure_slider,
                           self.tonemap_checkbox])
        # --- add all layouts
        self.layout.addWidget(pre_inversion_groupbox)
        self.layout.addWidget(inversion_groupbox)
        self.layout.addWidget(custom_grading_groupbox)
        self.setLayout(self.layout)


class EditingDisplay(QWidget):
    def __init__(self, parent=None):
        global PIPELINE
        super().__init__(parent)
        # internal tracking vars
        self.edit_params = EditParams()
        PIPELINE.editParamsUpdated.connect(self.set_edit_params_from_pipeline)

        # image preview / canvas
        self.image_preview = PrimaryImageView(self)

        # controls
        # demo controls
        self.current_file_label = QLabel("NO FILE LOADED")
        self.load_button = QPushButton("Load Image")
        # actual side panel
        self.scrollable_sidebar = QScrollArea()
        self.scrollable_sidebar.setWidgetResizable(True)
        self.scrollable_sidebar.setMinimumWidth(350)
        self.scrollable_sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tool_panel = ToolPanel()
        self.scrollable_sidebar.setWidget(self.tool_panel)

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
        self.editing_layout.addWidget(self.scrollable_sidebar)
        # add all layouts and wrap it all up
        self.layout.addLayout(self.load_layout)
        self.layout.addWidget(self.editing_layout)
        self.setLayout(self.layout)

        # wiring up functions
        # include some quick shorthand variables for readability
        tp = self.tool_panel
        ip = self.image_preview
        # ep = self.edit_params
        self.load_button.clicked.connect(self.open_load_dialog)
        PIPELINE.rawLoaded.connect(self.update_current_filename_display)
        tp.pre_inv_rotate_left.clicked.connect(lambda: self.update_rotation(1))
        tp.pre_inv_rotate_right.clicked.connect(lambda: self.update_rotation(-1))
        # this is super weird and fragile with many edge cases
        # i.e. two pickers can be activated at once
        # but I think I actually kinda like that?
        # allow pickers to trigger picker mode
        tp.pre_inv_wb_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.pre_inv_wb_picker.valueChanged.connect(PIPELINE.pick_color_from_image)
        tp.lo_gray_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.lo_gray_picker.valueChanged.connect(PIPELINE.pick_color_from_image)
        tp.hi_gray_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.hi_gray_picker.valueChanged.connect(PIPELINE.pick_color_from_image)
        tp.grading_wb_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.grading_wb_picker.valueChanged.connect(PIPELINE.pick_color_from_image)
        # allow view to send values back
        ip.view.pointPicked.connect(tp.pre_inv_wb_picker.finish_pick)
        ip.view.pointPicked.connect(tp.lo_gray_picker.finish_pick)
        ip.view.pointPicked.connect(tp.hi_gray_picker.finish_pick)
        ip.view.pointPicked.connect(tp.grading_wb_picker.finish_pick)
        # update_edit_params on change of any subvalue of ToolPanel
        for tool in tp.tools:
            if hasattr(tool, "clicked"):
                tool.clicked.connect(self.update_edit_params)
            elif hasattr(tool, "valueChanged"):
                tool.valueChanged.connect(self.update_edit_params)

    def update_rotation(self, direction):
        global PIPELINE
        ep = self.edit_params
        ep.rotation = ep.rotation + direction
        if PIPELINE:
            PIPELINE.process_image(ep.copy())

    def update_edit_params(self):
        global PIPELINE
        ep = self.edit_params
        tp = self.tool_panel
        ep.skip_inversion = tp.skip_inversion_checkbox.isChecked()
        ep.bw_mode = tp.bw_checkbox.isChecked()
        ep.base_color_xy = tp.pre_inv_wb_picker.value()
        ep.base_color = tp.pre_inv_wb_picker.colorValue()
        ep.red_ratio = tp.red_ratio_slider.value()
        ep.blue_ratio = tp.blue_ratio_slider.value()
        ep.green_exponent = tp.contrast_slider.value()
        ep.lo_gray_xy = tp.lo_gray_picker.value()
        ep.lo_gray = tp.lo_gray_picker.colorValue()
        ep.hi_gray_xy = tp.hi_gray_picker.value()
        ep.hi_gray = tp.hi_gray_picker.colorValue()
        ep.red_gain = tp.red_gain_slider.value()
        ep.green_gain = tp.green_gain_slider.value()
        ep.blue_gain = tp.blue_gain_slider.value()
        ep.wb_xy = tp.grading_wb_picker.value()
        ep.wb_reference = tp.grading_wb_picker.colorValue()
        ep.wb_red = tp.red_tune_slider.value()
        ep.wb_green = tp.green_tune_slider.value()
        ep.wb_blue = tp.blue_tune_slider.value()
        ep.exposure_comp = tp.exposure_slider.value()
        ep.tonemap = tp.tonemap_checkbox.isChecked()
        if PIPELINE:
            PIPELINE.process_image(ep.copy())

    def set_edit_params_from_pipeline(self):
        self.edit_params = PIPELINE.edit_params.copy()
        ep = self.edit_params
        tp = self.tool_panel
        tp.skip_inversion_checkbox.setChecked(ep.skip_inversion)
        tp.bw_checkbox.setChecked(ep.bw_mode)
        tp.pre_inv_wb_picker.update_color(ep.base_color)
        tp.red_ratio_slider.setValue(ep.red_ratio)
        tp.blue_ratio_slider.setValue(ep.blue_ratio)
        tp.contrast_slider.setValue(ep.green_exponent)
        tp.lo_gray_picker.update_color(ep.lo_gray)
        tp.hi_gray_picker.update_color(ep.hi_gray)
        tp.red_gain_slider.setValue(ep.red_gain)
        tp.green_gain_slider.setValue(ep.green_gain)
        tp.blue_gain_slider.setValue(ep.blue_gain)
        tp.grading_wb_picker.update_color(ep.wb_reference)
        tp.red_tune_slider.setValue(ep.wb_red)
        tp.green_tune_slider.setValue(ep.wb_green)
        tp.blue_tune_slider.setValue(ep.wb_blue)
        tp.exposure_slider.setValue(ep.exposure_comp)
        tp.tonemap_checkbox.setChecked(ep.tonemap)

    def open_load_dialog(self):
        global PIPELINE, LOADED_RAW_PATH
        image_path, _ = QFileDialog.getOpenFileName()
        PIPELINE.load_raw_file(image_path)
        LOADED_RAW_PATH = image_path

    def update_current_filename_display(self, filename: str):
        self.current_file_label.setText(filename)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Negative Inversion Processor")
        self.editing_display = EditingDisplay()
        self.setCentralWidget(self.editing_display)
        self.resize(1200, 600)


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
