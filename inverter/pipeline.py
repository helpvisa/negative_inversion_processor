# pipeline to process image from load -> preview
# custom class that stores:
#     - source image (unmodifed)
#     - edit_params (per-image)
#     - intermediate working image at each pipeline stage
#         - pre-inversion
#         - inversion
#         - grading
#         - final
import numpy as np
from scipy import ndimage
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox
from deepdiff import DeepDiff
from threads import WorkerThreadPool
from edit_params import EditParams, Stage
from processing import (load_raw_image, rotate_image,
                        white_balance, density_balance,
                        average_sample_point,
                        invert_to_density, density_to_luminance,
                        convert_to_grayscale_from_g,
                        apply_addition, apply_gain)
from custom_widgets import ColorPicker
from colour_management import aces_tonemap
from global_vars import REC2020_WEIGHTS


# we must subclass QObject to leverage signals
class ProcessingPipeline(QObject):
    previewUpdated = Signal()
    editParamsUpdated = Signal()
    colorPicked = Signal(tuple[float, float, float])

    def __init__(self, max_preview_size=1200, analysis_inset=0.8):
        super().__init__()
        self.threadpool = WorkerThreadPool()
        self.edit_params = EditParams()
        self.analysis_inset = analysis_inset
        self.raw_width = 0
        self.raw_height = 0
        self.auto_analysis_bounds = None
        self.source_image = None
        self.source_image_small = None
        self.final_preview = None
        self.max_preview_size = max_preview_size
        self.preview_scale = 1.0
        # track a median copy for each stage of the pipeline
        self.pre_inv_inter = []
        self.inv_inter = []
        self.ratio_inter = []
        self.grade_inter = []
        # and kick off an initial edit
        # self.pre_inv_process()

    # perform a deep comparison of self.edit_params and the new EditParams
    # passed in to determine where in the pipeline the reprocess needs to occur
    def process_image(self, new_edit_params):
        if (self.edit_params != new_edit_params):
            difference = DeepDiff(self.edit_params, new_edit_params)
            self.edit_params = new_edit_params
            # eventually, we check the difference to update only where
            # a change has actually occurred
            self.pre_inv_process()

    def pre_inv_process(self):
        ep = self.edit_params

        def current():
            self.pre_inv_inter = rotate_image(self.source_image_small,
                                              ep.rotation)
            # shift analysis bounds based on image rotation
            if ep.rotation % 2:
                # swizzle
                self.update_analysis_bounds(self.raw_height, self.raw_width)
            else:
                self.update_analysis_bounds()
            if ep.base_color_xy:
                wb_adjustment = white_balance(self.pre_inv_inter,
                                              custom_wb_point=ep.base_color_xy)
                self.pre_inv_inter = apply_gain(self.pre_inv_inter,
                                                wb_adjustment['values'])
            else:
                wb_adjustment = white_balance(self.pre_inv_inter,
                                              colourspace_weights=REC2020_WEIGHTS,
                                              region=self.auto_analysis_bounds)
                self.pre_inv_inter = apply_gain(self.pre_inv_inter,
                                                wb_adjustment['values'])

        def proceed():
            self.inv_process()

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def inv_process(self):
        ep = self.edit_params

        def current():
            if not ep.skip_inversion:
                self.inv_inter = invert_to_density(self.pre_inv_inter)
            else:
                self.inv_inter = self.pre_inv_inter.copy()

        def proceed():
            self.ratio_process()
        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def ratio_process(self):
        ep = self.edit_params

        def current():
            if not ep.skip_inversion:
                if not ep.bw_mode:
                    scale, shift = density_balance(self.inv_inter,
                                                   region=self.auto_analysis_bounds,
                                                   exponent=ep.green_exponent,
                                                   red_ratio=ep.red_ratio,
                                                   blue_ratio=ep.blue_ratio)
                    self.ratio_inter = apply_gain(self.inv_inter, scale['values'])
                    self.ratio_inter = apply_addition(self.ratio_inter, shift['values'])
                else:
                    scale, shift = density_balance(self.inv_inter,
                                                   region=self.auto_analysis_bounds,
                                                   exponent=ep.green_exponent,
                                                   red_ratio=1.0,
                                                   blue_ratio=1.0)
                    self.ratio_inter = apply_gain(self.inv_inter, scale['values'])
                    self.ratio_inter = apply_addition(self.ratio_inter, shift['values'])
            else:
                self.ratio_inter = self.inv_inter.copy()

        def proceed():
            self.grade_process()

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def grade_process(self):
        ep = self.edit_params

        def current():
            self.grade_inter = self.ratio_inter.copy()
            # first apply grade
            if not ep.bw_mode:
                # gain
                gain_tuple = (ep.red_gain, ep.green_gain, ep.blue_gain)
                self.grade_inter = apply_gain(self.grade_inter, gain_tuple)
                # tune (addition)
                add_tuple = (ep.wb_red, ep.wb_green, ep.wb_blue)
                self.grade_inter = apply_addition(self.grade_inter, add_tuple)
            # then convert to luminance
            if not ep.skip_inversion:
                self.grade_inter = density_to_luminance(self.grade_inter)
            # then adjust exposure
            comp_tuple = (ep.exposure_comp, ep.exposure_comp, ep.exposure_comp)
            self.grade_inter = apply_gain(self.grade_inter, comp_tuple)
            self.final_preview = self.grade_inter.copy()
            if ep.bw_mode:
                self.final_preview = convert_to_grayscale_from_g(self.final_preview)

        def proceed():
            if ep.tonemap:
                self.tonemap_process()
            else:
                self.previewUpdated.emit()

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def tonemap_process(self):
        def current():
            self.final_preview = aces_tonemap(self.final_preview)

        def proceed():
            self.previewUpdated.emit()

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def load_raw_file(self, image_path):
        # load raw image and resize it for the preview pane
        if image_path:
            # the function to be executed within a separate thread
            def init_func():
                raw = load_raw_image(image_path).astype(np.float32) / 65535.0
                height, width, _ = raw.shape
                self.raw_width = width
                self.raw_height = height
                if width > height:
                    self.preview_scale = self.max_preview_size / width
                else:
                    self.preview_scale = self.max_preview_size / height
                self.update_analysis_bounds()
                scaled_raw = ndimage.zoom(raw, (self.preview_scale,
                                                self.preview_scale, 1), order=0)
                return raw, scaled_raw

            # the function to be executed upon thread completion
            def post_func(image_data_tuple):
                self.source_image = image_data_tuple[0]
                self.source_image_small = image_data_tuple[1]
                self.edit_params = EditParams()
                self.editParamsUpdated.emit()
                self.pre_inv_process()

            def error_func(e):
                exctype, value, error = e
                # tell user there's an issue
                QMessageBox.critical(self, "Error processing image!", error)

            # instantiate and run a thread
            # should split this out into its own function, surely
            thread = self.threadpool.instantiate_thread(init_func)
            self.threadpool.active_threads[thread.thread_id] = thread
            thread.signals.result.connect(post_func)
            thread.signals.error.connect(error_func)
            thread.signals.finished.connect(self.threadpool.remove_thread)
            self.threadpool.start(thread)

    def update_analysis_bounds(self,
                               width=None,
                               height=None):
        # replace width/height with self.raw_[w|h] if value not present
        if not width:
            width = self.raw_width
        if not height:
            height = self.raw_height
        # determine auto-analysis bounds
        a_width = int(width * self.analysis_inset * self.preview_scale)
        a_height = int(height * self.analysis_inset * self.preview_scale)
        a_start_x = int(width * self.preview_scale - a_width) // 2
        a_end_x = a_start_x + a_width
        a_start_y = int(height * self.preview_scale - a_height) // 2
        a_end_y = a_start_y + a_height
        self.auto_analysis_bounds = [[a_start_x, a_start_y],
                                     [a_end_x, a_end_y]]

    @Slot()
    def pick_color_from_image(self, point_x, point_y,
                              stage: Stage, picker: ColorPicker):
        if stage == Stage.PRE_INV:
            value = average_sample_point(self.pre_inv_inter,
                                         point_x, point_y, 8)
            picker.update_color(value)
        elif stage == Stage.INV:
            value = average_sample_point(self.inv_inter,
                                         point_x, point_y, 8)
            picker.update_color(value)
        elif stage == Stage.RATIO:
            value = average_sample_point(self.ratio_inter,
                                        point_x, point_y, 8)
            picker.update_color(value)
        elif stage == Stage.GRADE:
            value = average_sample_point(self.grade_inter,
                                        point_x, point_y, 8)
            picker.update_color(value)
        else:
            value = average_sample_point(self.final_preview,
                                        point_x, point_y, 8)
            picker.update_color(value)
