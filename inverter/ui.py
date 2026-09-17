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
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QLabel, QPushButton, QFileDialog, QDialog,
                               QGraphicsScene, QSplitter,
                               QCheckBox, QGroupBox, QScrollArea, QComboBox,
                               QSizePolicy, QListWidget, QMessageBox)
from custom_widgets import ImageView, LabeledSlider, ColorPicker
from colour_management import convert_to_sRGB
from edit_params import EditParams, Stage
from pipeline import ProcessingPipeline
from processing import estimate_inversion_ratios
from sidecars import load_params_from_sidecar, update_sidecar
from global_vars import GLOBAL_FLAGS, PHOTO_INDEX, SAVE_FORMATS, RAW_EXTENSIONS


# some global variables for tracking information about the current session
CLIPBOARD = None
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
        # -- no associated layout (generics)
        self.copy_params_button = QPushButton("Copy Parameters")
        self.paste_params_button = QPushButton("Paste Parameters")
        self.layout.addWidget(self.copy_params_button)
        self.layout.addWidget(self.paste_params_button)
        # --- pre-inversion layout (orientation and initial white balance)
        pre_inversion_groupbox = QGroupBox("Pre-Inversion")
        pre_inversion_layout = QVBoxLayout()
        self.ffc_layout = QHBoxLayout()
        self.ffc_label = QLabel("No FFC image loaded.")
        self.ffc_button = QPushButton("Load FFC")
        self.ffc_layout.addWidget(self.ffc_label)
        self.ffc_layout.addWidget(self.ffc_button)
        self.crop_inset_slider = LabeledSlider("Crop Inset",
                                               0.0, 1.0, 1.0, 300)
        self.crop_shift_layout = QHBoxLayout()
        self.crop_shift_h_slider = LabeledSlider("Horizontal Shift",
                                                 -0.5, 0.5, 0.0, 1000)
        self.crop_shift_v_slider = LabeledSlider("Vertical Shift",
                                                 -0.5, 0.5, 0.0, 1000)
        self.crop_shift_layout.addWidget(self.crop_shift_h_slider)
        self.crop_shift_layout.addWidget(self.crop_shift_v_slider)
        self.pre_inv_orientation_layout = QHBoxLayout()
        self.pre_inv_rotate_left = QPushButton("Rotate Left")
        self.pre_inv_rotate_right = QPushButton("Rotate Right")
        self.pre_inv_orientation_layout.addWidget(self.pre_inv_rotate_left)
        self.pre_inv_orientation_layout.addWidget(self.pre_inv_rotate_right)
        self.pre_inv_wb_picker = ColorPicker("Dmin - Base Color", Stage.PRE_INV)
        pre_inversion_layout.addLayout(self.ffc_layout)
        pre_inversion_layout.addWidget(self.crop_inset_slider)
        pre_inversion_layout.addLayout(self.crop_shift_layout)
        pre_inversion_layout.addLayout(self.pre_inv_orientation_layout)
        pre_inversion_layout.addWidget(self.pre_inv_wb_picker)
        pre_inversion_groupbox.setLayout(pre_inversion_layout)
        self.tools.extend([self.crop_inset_slider,
                           self.crop_shift_h_slider,
                           self.crop_shift_v_slider,
                           self.pre_inv_rotate_left,
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
        self.pivot_slider = LabeledSlider("Pivot",
                                          0.0, 2.0, 0.745, 300)
        # self.pivot_picker = ColorPicker("Pick Pivot from Film Base", Stage.RATIO,
        #                                 hide_value=True)
        self.red_ratio_slider = LabeledSlider("Red Ratio",
                                              0.0, 3.0, 1.36, 300,
                                              "#ffcccc")
        self.blue_ratio_slider = LabeledSlider("Blue Ratio",
                                               0.0, 3.0, 0.86, 300,
                                               "#ccccff")
        self.contrast_slider = LabeledSlider("Contrast",
                                             0.0, 5.0, 1.0, 300)
        self.estimate_ratios_button = QPushButton("Estimate Ratios (Crop-Dependent)")
        inversion_layout.addLayout(checkbox_layout)
        # inversion_layout.addWidget(self.pivot_picker)
        inversion_layout.addWidget(self.pivot_slider)
        inversion_layout.addWidget(self.red_ratio_slider)
        inversion_layout.addWidget(self.blue_ratio_slider)
        inversion_layout.addWidget(self.estimate_ratios_button)
        inversion_layout.addWidget(self.contrast_slider)
        inversion_groupbox.setLayout(inversion_layout)
        self.tools.extend([self.skip_inversion_checkbox,
                           self.bw_checkbox,
                           # self.pivot_picker,
                           self.pivot_slider,
                           self.red_ratio_slider,
                           self.blue_ratio_slider,
                           self.estimate_ratios_button,
                           self.contrast_slider])
        # --- user grading
        custom_grading_groupbox = QGroupBox("Grading")
        custom_grading_layout = QVBoxLayout()
        grading_gain_layout = QHBoxLayout()
        self.red_gain_slider = LabeledSlider("R Gain",
                                             0.0, 2.0, 1.0, 1000,
                                             "#ff0000")
        self.green_gain_slider = LabeledSlider("G Gain",
                                               0.0, 2.0, 1.0, 1000,
                                               "#00ff00")
        self.blue_gain_slider = LabeledSlider("B Gain",
                                              0.0, 2.0, 1.0, 1000,
                                              "#0000ff")
        grading_gain_layout.addWidget(self.red_gain_slider)
        grading_gain_layout.addWidget(self.green_gain_slider)
        grading_gain_layout.addWidget(self.blue_gain_slider)
        grading_tune_layout = QHBoxLayout()
        self.red_tune_slider = LabeledSlider("R Offset",
                                             -1.0, 1.0, 0.0, 1000,
                                             "#ffeeee")
        self.green_tune_slider = LabeledSlider("G Offset",
                                               -1.0, 1.0, 0.0, 1000,
                                               "#eeffee")
        self.blue_tune_slider = LabeledSlider("B Offset",
                                              -1.0, 1.0, 0.0, 1000,
                                              "#eeeeff")
        grading_tune_layout.addWidget(self.red_tune_slider)
        grading_tune_layout.addWidget(self.green_tune_slider)
        grading_tune_layout.addWidget(self.blue_tune_slider)
        self.grading_wb_picker = ColorPicker("Pick White Balance", Stage.GAIN_GRADE,
                                             hide_value=True)
        custom_grading_layout.addLayout(grading_gain_layout)
        custom_grading_layout.addLayout(grading_tune_layout)
        custom_grading_layout.addWidget(self.grading_wb_picker)
        custom_grading_groupbox.setLayout(custom_grading_layout)
        self.tools.extend([self.red_gain_slider,
                           self.green_gain_slider,
                           self.blue_gain_slider,
                           self.red_tune_slider,
                           self.green_tune_slider,
                           self.blue_tune_slider,
                           self.grading_wb_picker])
        # --- tonemapping and final output
        tonemap_groupbox = QGroupBox("Tonemapping and Output")
        tonemap_layout = QVBoxLayout()
        self.exposure_slider = LabeledSlider("Final Exposure",
                                             0.0, 20.0, 1.0, 400)
        self.tonemap_checkbox = QCheckBox("Apply Tonemapping")
        self.toe_slider = LabeledSlider("Toe",
                                        0.0, 1.0, 0.20, 400)
        tonemap_layout.addWidget(self.exposure_slider)
        tonemap_layout.addWidget(self.tonemap_checkbox)
        tonemap_layout.addWidget(self.toe_slider)
        tonemap_groupbox.setLayout(tonemap_layout)
        self.tools.extend([self.exposure_slider,
                           self.tonemap_checkbox,
                           self.toe_slider])
        # --- add all layouts
        self.layout.addWidget(pre_inversion_groupbox)
        self.layout.addWidget(inversion_groupbox)
        self.layout.addWidget(custom_grading_groupbox)
        self.layout.addWidget(tonemap_groupbox)
        self.setLayout(self.layout)


class EditingDisplay(QWidget):
    def __init__(self, parent=None):
        global PIPELINE
        super().__init__(parent)
        # internal tracking vars
        self.edit_params = EditParams()

        # image preview / canvas
        self.image_preview = PrimaryImageView(self)
        self.performing_batch_export = False
        self.images_to_export = 0
        self.images_exported = 0

        # controls
        # demo controls
        self.current_file_label = QLabel("NO FILE LOADED")
        self.current_file_label.setSizePolicy(QSizePolicy.Policy.Preferred,
                                              QSizePolicy.Policy.Fixed)
        self.current_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_button = QPushButton("Load Image")
        self.load_folder_button = QPushButton("Load Roll (Folder)")
        self.paste_parameters_to_roll_button = QPushButton("Paste Current Parameters to Entire Roll")
        self.profile_warning = QLabel("NIP processes all images in Linear Rec.2020.")
        self.profile_picker = QComboBox()
        self.profile_picker.addItem("sRGB Gamma 2.2",
                                    userData="srgb")
        self.profile_picker.addItem("Linear Rec.2020 Gamma 1.0",
                                    userData="rec2020")
        self.save_button = QPushButton("Save Processed Image")
        self.batch_export_button = QPushButton("Export Entire Roll")
        self.format_combobox = QComboBox()
        # add items to combobox
        for item in SAVE_FORMATS:
            self.format_combobox.addItem(item['display'],
                                         userData=item['data'])
        # file management side panel
        self.file_list = QListWidget()
        # editing side panel
        self.scrollable_sidebar = QScrollArea()
        self.scrollable_sidebar.setWidgetResizable(True)
        self.scrollable_sidebar.setMinimumWidth(350)
        self.scrollable_sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tool_panel = ToolPanel()
        self.scrollable_sidebar.setWidget(self.tool_panel)
        # statusbar
        self.message_queue = []
        self.statusbar = QLabel("No status to display.")
        self.statusbar.setSizePolicy(QSizePolicy.Policy.Preferred,
                                     QSizePolicy.Policy.Fixed)

        # define layouts
        # top-level layout
        self.layout = QVBoxLayout()
        # setup header
        self.headerbar_layout = QHBoxLayout()
        self.headerbar_layout.addWidget(self.current_file_label)
        # setup file management layout
        self.management_sidebar = QWidget()
        self.management_layout = QVBoxLayout()
        self.load_layout = QHBoxLayout()
        self.export_layout = QVBoxLayout()
        self.format_layout = QHBoxLayout()
        self.load_layout.addWidget(self.load_button)
        self.load_layout.addWidget(self.load_folder_button)
        self.format_layout.addWidget(self.profile_picker)
        self.format_layout.addWidget(self.format_combobox)
        self.export_layout.addWidget(self.profile_warning)
        self.export_layout.addLayout(self.format_layout)
        self.export_layout.addWidget(self.save_button)
        self.export_layout.addWidget(self.batch_export_button)
        self.management_layout.addLayout(self.load_layout)
        self.management_layout.addWidget(self.paste_parameters_to_roll_button)
        self.management_layout.addWidget(self.file_list)
        self.management_layout.addLayout(self.export_layout)
        self.management_sidebar.setLayout(self.management_layout)
        # editing layout
        self.editing_layout = QSplitter(Qt.Horizontal)
        self.editing_layout.setHandleWidth(16)
        self.editing_layout.addWidget(self.management_sidebar)
        self.editing_layout.addWidget(self.image_preview)
        self.editing_layout.addWidget(self.scrollable_sidebar)
        # add all layouts and wrap it all up
        self.layout.addLayout(self.headerbar_layout)
        self.layout.addWidget(self.editing_layout)
        self.layout.addWidget(self.statusbar)
        self.setLayout(self.layout)

        # wiring up functions
        # include some quick shorthand variables for readability
        tp = self.tool_panel
        ip = self.image_preview
        # ep = self.edit_params
        self.load_button.clicked.connect(self.open_load_dialog)
        self.load_folder_button.clicked.connect(self.open_load_folder_dialog)
        self.paste_parameters_to_roll_button.clicked.connect(self.paste_parameters_to_entire_index)
        self.save_button.clicked.connect(self.open_save_dialog)
        self.batch_export_button.clicked.connect(self.open_batch_export_dialog)
        self.file_list.itemDoubleClicked.connect(self.handle_list_view_doubleclick)
        PIPELINE.rawLoaded.connect(self.update_current_filename_display)
        PIPELINE.messageRaised.connect(self.push_message)
        PIPELINE.editParamsUpdated.connect(self.set_edit_params_from_pipeline)
        PIPELINE.photoIndexUpdated.connect(self.update_list_view)
        PIPELINE.saveFinished.connect(self.update_batch_export_status)
        tp.ffc_button.clicked.connect(self.open_ffc_selection_dialog)
        tp.copy_params_button.clicked.connect(self.copy_parameters)
        tp.paste_params_button.clicked.connect(self.paste_parameters)
        tp.pre_inv_rotate_left.clicked.connect(lambda: self.update_rotation(1))
        tp.pre_inv_rotate_right.clicked.connect(lambda: self.update_rotation(-1))
        tp.estimate_ratios_button.clicked.connect(self.estimate_ratios)
        # this is super weird and fragile with many edge cases
        # i.e. two pickers can be activated at once
        # but I think I actually kinda like that?
        # allow pickers to trigger picker mode
        tp.pre_inv_wb_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.pre_inv_wb_picker.valueChanged.connect(PIPELINE.pick_color_from_image)
        # tp.pivot_picker.pickRequested.connect(ip.view.enable_pick_mode)
        # tp.pivot_picker.valueChanged.connect(PIPELINE.pick_color_from_image)
        tp.grading_wb_picker.pickRequested.connect(ip.view.enable_pick_mode)
        tp.grading_wb_picker.valueChanged.connect(PIPELINE.pick_color_from_image)
        # allow view to send values back
        # ip.view.pointPicked.connect(tp.pivot_picker.finish_pick)
        ip.view.pointPicked.connect(tp.pre_inv_wb_picker.finish_pick)
        ip.view.pointPicked.connect(tp.grading_wb_picker.finish_pick)
        # update pivot if pivot picked
        # tp.pivot_picker.colorChanged.connect(self.update_pivot)
        # update grading panel if grading wb picked
        tp.grading_wb_picker.colorChanged.connect(self.update_grading_panel)
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

    def update_grading_panel(self, color):
        tp = self.tool_panel
        tp.red_tune_slider.setValue(color[1] - color[0])
        tp.green_tune_slider.setValue(color[1] - color[1])
        tp.blue_tune_slider.setValue(color[1] - color[2])

    def estimate_ratios(self, color):
        global PIPELINE
        tp = self.tool_panel
        if PIPELINE.min_percentile is not None and \
           PIPELINE.min_percentile.any() and \
           PIPELINE.max_percentile is not None and \
           PIPELINE.max_percentile.any():
            red_ratio, blue_ratio = estimate_inversion_ratios(PIPELINE.min_percentile,
                                                              PIPELINE.max_percentile)
            tp.red_ratio_slider.setValue(red_ratio)
            tp.blue_ratio_slider.setValue(blue_ratio)
            tp.pivot_slider.setValue(PIPELINE.min_percentile[1])
           
    def copy_parameters(self):
        global CLIPBOARD
        CLIPBOARD = self.edit_params.copy()
        self.push_message("Image parameters copied to clipboard.")

    def paste_parameters(self):
        global CLIPBOARD, PIPELINE
        if CLIPBOARD and PIPELINE:
            self.push_message("Image parameters pasted from clipboard.")
            new_edit_params = CLIPBOARD.copy()
            PIPELINE.process_image(new_edit_params.copy(), force_refresh=True)
            self.set_edit_params_from_pipeline()

    def paste_parameters_to_entire_index(self):
        global CLIPBOARD

        # first make sure user understands what they're doing
        reply = QMessageBox.question(self,
                                     "WARNING",
                                     "This will copy the parameters in your "
                                     "clipboard to every image in the file "
                                     "list, updating their sidecars in the "
                                     "process. Please confirm you want to do "
                                     "this.")
        if reply == QMessageBox.StandardButton.No:
            print("NOT copying parameters to entire roll!", file=sys.stderr)
            return
        print("Copying parameters to entire roll!", file=sys.stderr)
        
        if CLIPBOARD:
            self.push_message("Applied clipboard parameters to entire roll.")
            new_edit_params = CLIPBOARD.copy()
            for entry in PHOTO_INDEX:
                PHOTO_INDEX[entry]['edit_params'] = new_edit_params.copy()
                update_sidecar(entry)

    def update_edit_params(self):
        global PIPELINE, LOADED_RAW_PATH
        ep = self.edit_params
        tp = self.tool_panel
        ep.ffc_image = tp.ffc_label.text()
        ep.crop_inset = tp.crop_inset_slider.value()
        ep.crop_shift_h = tp.crop_shift_h_slider.value()
        ep.crop_shift_v = tp.crop_shift_v_slider.value()
        ep.skip_inversion = tp.skip_inversion_checkbox.isChecked()
        ep.bw_mode = tp.bw_checkbox.isChecked()
        ep.base_color_xy = tp.pre_inv_wb_picker.value()
        ep.base_color = tp.pre_inv_wb_picker.colorValue()
        ep.pivot = tp.pivot_slider.value()
        ep.red_ratio = tp.red_ratio_slider.value()
        ep.blue_ratio = tp.blue_ratio_slider.value()
        ep.green_exponent = tp.contrast_slider.value()
        ep.red_gain = tp.red_gain_slider.value()
        ep.green_gain = tp.green_gain_slider.value()
        ep.blue_gain = tp.blue_gain_slider.value()
        ep.wb_xy = tp.grading_wb_picker.value()
        ep.wb_reference = tp.grading_wb_picker.colorValue()
        ep.wb_red = tp.red_tune_slider.value()
        ep.wb_green = tp.green_tune_slider.value()
        ep.wb_blue = tp.blue_tune_slider.value()
        ep.final_exposure = tp.exposure_slider.value()
        ep.tonemap = tp.tonemap_checkbox.isChecked()
        ep.toe = tp.toe_slider.value()
        if PIPELINE:
            PIPELINE.process_image(ep.copy())

    def set_edit_params_from_pipeline(self):
        self.edit_params = PIPELINE.edit_params.copy()
        ep = self.edit_params
        tp = self.tool_panel
        tp.ffc_label.setText(ep.ffc_image)
        tp.crop_inset_slider.setValue(ep.crop_inset)
        tp.crop_shift_h_slider.setValue(ep.crop_shift_h)
        tp.crop_shift_v_slider.setValue(ep.crop_shift_v)
        tp.skip_inversion_checkbox.setChecked(ep.skip_inversion)
        tp.bw_checkbox.setChecked(ep.bw_mode)
        tp.pre_inv_wb_picker.update_color(ep.base_color)
        tp.pivot_slider.setValue(ep.pivot)
        tp.red_ratio_slider.setValue(ep.red_ratio)
        tp.blue_ratio_slider.setValue(ep.blue_ratio)
        tp.contrast_slider.setValue(ep.green_exponent)
        tp.red_gain_slider.setValue(ep.red_gain)
        tp.green_gain_slider.setValue(ep.green_gain)
        tp.blue_gain_slider.setValue(ep.blue_gain)
        tp.grading_wb_picker.update_color(ep.wb_reference)
        tp.red_tune_slider.setValue(ep.wb_red)
        tp.green_tune_slider.setValue(ep.wb_green)
        tp.blue_tune_slider.setValue(ep.wb_blue)
        tp.exposure_slider.setValue(ep.final_exposure)
        tp.tonemap_checkbox.setChecked(ep.tonemap)
        tp.toe_slider.setValue(ep.toe)


    def trigger_raw_load(self, image_path):
        global LOADED_RAW_PATH
        if image_path:
            self.push_message(f"Loading {image_path} from disk.")
            PIPELINE.load_raw_file(image_path)
            LOADED_RAW_PATH = image_path

    def handle_list_view_doubleclick(self, list_widget: QListWidget):
        raw_text = list_widget.text()
        self.trigger_raw_load(raw_text)

    def open_ffc_selection_dialog(self):
        global PIPELINE
        filter_string = "Camera RAW ("
        filter_string += " ".join(f"*.{e}" for e in RAW_EXTENSIONS)
        filter_string += ");; All Files (*)"
        image_path, _ = QFileDialog.getOpenFileName(
            None,
            caption="Select RAW File",
            filter=filter_string
        )
        self.tool_panel.ffc_label.setText(image_path)
        self.push_message(f"FFC image set to {image_path}.")
        self.update_edit_params()


    def open_load_dialog(self):
        global PIPELINE, LOADED_RAW_PATH
        filter_string = "Camera RAW ("
        filter_string += " ".join(f"*.{e}" for e in RAW_EXTENSIONS)
        filter_string += ");; All Files (*)"
        image_path, _ = QFileDialog.getOpenFileName(
            None,
            caption="Select RAW File",
            filter=filter_string
        )
        self.trigger_raw_load(image_path)

    def open_load_folder_dialog(self):
        selected_path = QFileDialog.getExistingDirectory()
        folder_path = Path(selected_path)
        # reset the photo index (include only current folder)
        # this is why I called them "evil mutable global variables" >:^) hehe
        PHOTO_INDEX.clear()
        # [1:] required to strip period at start of f.suffix.lower()
        files = [f for f in folder_path.iterdir() if \
                 f.is_file() and f.suffix.lower()[1:] in RAW_EXTENSIONS]
        print(files, file=sys.stderr)
        for f in files:
            name = str(f)
            # load each file into the PHOTO_INDEX
            PHOTO_INDEX[name] = {}
            if 'edit_params' not in PHOTO_INDEX[name]:
                print(f"Loading edit_params from sidecar for {name}.",
                      file=sys.stderr)
                PHOTO_INDEX[name]['edit_params'] = load_params_from_sidecar(name)
        self.update_list_view()

    def open_save_dialog(self):
        global PIPELINE, LOADED_RAW_PATH
        current_format = self.format_combobox.currentData()
        current_profile = self.profile_picker.currentData()
        default_filter = "TIFF Files (*.tiff *.tif)"
        default_extension = "tiff"
        if current_format.startswith('j'):
            default_filter = "JPEG Files (*.jpg *.jpeg)"
            default_extension = "jpg"
        new_dialog = QFileDialog()
        new_dialog.setAcceptMode(QFileDialog.AcceptSave)
        new_dialog.setNameFilter(default_filter)
        new_dialog.setDefaultSuffix(default_extension)
        if new_dialog.exec() == QDialog.DialogCode.Accepted:
            image_path = new_dialog.selectedFiles()
            PIPELINE.save_final_image(image_to_save=LOADED_RAW_PATH,
                                      output_path=image_path[0],
                                      image_format=current_format,
                                      icc_profile=current_profile)

    def open_batch_export_dialog(self):
        global PIPELINE
        current_format = self.format_combobox.currentData()
        current_profile = self.profile_picker.currentData()
        selected_path = QFileDialog.getExistingDirectory()
        if selected_path:
            self.push_message("Initiating batch export!")
            folder_path = Path(selected_path)
            images_to_export = 0
            for entry in PHOTO_INDEX:
                images_to_export += 1
                entry_path = Path(entry)
                new_suffix = ".tiff"
                if current_format == "ju8":
                    new_suffix = ".jpg"
                final_path = folder_path / f"{entry_path.stem}{new_suffix}"
                PIPELINE.save_final_image(image_to_save=entry,
                                          output_path=final_path,
                                          image_format=current_format,
                                          icc_profile=current_profile)
            self.images_to_export = images_to_export
            self.performing_batch_export = True

    def update_batch_export_status(self):
        if self.performing_batch_export:
            self.images_exported += 1
            self.push_message(f"Exported {self.images_exported} of "
                              f"{self.images_to_export}.")
            if self.images_exported >= self.images_to_export:
                self.performing_batch_export = False
                self.push_message("Finished exporting all images!")
            

    def update_list_view(self):
        self.file_list.clear()
        # sort the PHOTO_INDEX dictionary before redisplay
        sorted_dictionary = dict(sorted(PHOTO_INDEX.items()))
        for entry in sorted_dictionary:
            self.file_list.addItem(entry)

    def update_current_filename_display(self, filename: str):
        self.current_file_label.setText(filename)
        self.push_message("Loading finished.")

    def push_message(self, message: str = None):
        if message:
            self.message_queue.append(message)
        length = len(self.message_queue)
        if length > 2:
            self.statusbar.setText(f"{self.message_queue[length - 3]}\n"
                                   f"{self.message_queue[length - 2]}\n"
                                   f"{self.message_queue[length - 1]}")
        elif length > 1:
            self.statusbar.setText(f"{self.message_queue[length - 2]}\n"
                                   f"{self.message_queue[length - 1]}")
        elif length == 1:
            self.statusbar.setText(f"{self.message_queue[length - 1]}")


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
    # wait for threads
    exit_code = app.exec()
    print("Waiting for all threads to finish!", file=sys.stderr)
    PIPELINE.threadpool.waitForDone()
    print("All threads finished; exiting!", file=sys.stderr)
    sys.exit(exit_code)
