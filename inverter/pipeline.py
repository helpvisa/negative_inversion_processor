# pipeline to process image from load -> preview
# custom class that stores:
#     - source image (unmodifed)
#     - edit_params (per-image)
#     - intermediate working image at each pipeline stage
#         - pre-inversion
#         - inversion
#         - grading
#         - final
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox
from deepdiff import DeepDiff
from threads import WorkerThreadPool
from edit_params import EditParams, Stage
from processing import (load_raw_image, save_image, rotate_image,
                        white_balance, density_balance,
                        average_sample_point,
                        invert_to_density, density_to_luminance,
                        convert_to_grayscale_from_g,
                        apply_addition, apply_gain, apply_ffc)
from custom_widgets import ColorPicker
from colour_management import noritsu_tonemap, aces_tonemap, convert_to_sRGB
from sidecars import update_sidecar, load_params_from_sidecar
from global_vars import PHOTO_INDEX


# we must subclass QObject to leverage signals
class ProcessingPipeline(QObject):
    rawLoaded = Signal(str)
    saveInitiated = Signal(str)
    saveFinished = Signal(str)
    previewUpdated = Signal()
    editParamsUpdated = Signal()
    colorPicked = Signal(tuple[float, float, float])

    def __init__(self, max_preview_size=1280, analysis_inset=0.8):
        super().__init__()
        self.threadpool = WorkerThreadPool()
        self.edit_params = EditParams()
        self.analysis_inset = analysis_inset
        self.currently_loaded_filename = None
        self.raw_width = 0
        self.raw_height = 0
        self.min_percentile = None
        self.max_percentile = None
        self.ffc_image = None
        self.ffc_image_small = None
        self.source_image = None
        self.source_image_small = None
        self.final_preview = None
        self.max_preview_size = max_preview_size
        self.preview_scale = 1.0
        # track a median copy for each stage of the pipeline
        self.rotation_inter = []
        self.pre_inv_inter = []
        self.inv_inter = []
        self.ratio_inter = []
        self.grade_inter = []
        # and kick off an initial edit
        # self.pre_inv_process()

    # perform a deep comparison of self.edit_params and the new EditParams
    # passed in to determine where in the pipeline the reprocess needs to occur
    def process_image(self, new_edit_params):
        difference = DeepDiff(self.edit_params, new_edit_params)
        if difference:
            update_sidecar(self.currently_loaded_filename)
            print(difference, file=sys.stderr)
            self.edit_params = new_edit_params
            # update the global photo index
            PHOTO_INDEX[self.currently_loaded_filename]['edit_params'] = self.edit_params
            # eventually, we check the difference to update only where
            # a change has actually occurred
            changes = difference['values_changed']
            self.pre_inv_process(preview=True)

    def pre_inv_process(self, preview: bool = False):
        ep = self.edit_params

        def current():
            # pick image size and apply ffc
            if preview:
                self.rotation_inter = self.source_image_small
            else:
                self.rotation_inter = self.source_image
            if ep.ffc_image:
                if preview:
                    self.rotation_inter = apply_ffc(self.rotation_inter,
                                                    self.ffc_image_small)
                else:
                    self.rotation_inter = apply_ffc(self.rotation_inter,
                                                    self.ffc_image)
            # apply crop inset
            if ep.crop_inset < 1.0:
                crop_height = int(self.raw_height * self.preview_scale * ep.crop_inset)
                crop_width = int(self.raw_width * self.preview_scale * ep.crop_inset)
                crop_start_y = int(self.raw_height * self.preview_scale - crop_height) // 2
                crop_end_y = crop_start_y + crop_height
                crop_start_x = int(self.raw_width * self.preview_scale - crop_width) // 2
                crop_end_x = crop_start_x + crop_width
                self.rotation_inter = self.rotation_inter[crop_start_y:crop_end_y,
                                                          crop_start_x:crop_end_x]
            # apply rotation
            self.rotation_inter = rotate_image(self.rotation_inter,
                                               ep.rotation)
            # apply white balance
            if ep.base_color is not None and ep.base_color.any():
                wb_adjustment = white_balance(self.rotation_inter,
                                              rgb_value=ep.base_color)
                self.pre_inv_inter = apply_gain(self.rotation_inter,
                                                wb_adjustment['values'])
            else:
                self.pre_inv_inter = self.rotation_inter

        def proceed():
            self.inv_process(preview)

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def inv_process(self, preview: bool = False):
        ep = self.edit_params

        def current():
            if not ep.skip_inversion:
                self.inv_inter = invert_to_density(self.pre_inv_inter)
                # calculate percentiles
                # reshape flattens height, width into 1D pixel array
                array_1d = self.inv_inter.reshape(-1, 3)
                self.min_percentile = np.percentile(array_1d, 0.5, axis=0)
                self.max_percentile = np.percentile(array_1d, 99.8, axis=0)
            else:
                self.inv_inter = self.pre_inv_inter.copy()

        def proceed():
            self.ratio_process(preview)

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def ratio_process(self, preview: bool = False):
        ep = self.edit_params

        def current():
            if not ep.skip_inversion:
                if not ep.bw_mode:
                    scale, shift = density_balance(self.inv_inter,
                                                   exponent=ep.green_exponent,
                                                   red_ratio=ep.red_ratio,
                                                   blue_ratio=ep.blue_ratio,
                                                   pivot=ep.pivot,
                                                   out_brightness=ep.out_brightness)
                    self.ratio_inter = apply_gain(self.inv_inter, scale['values'])
                    self.ratio_inter = apply_addition(self.ratio_inter, shift['values'])
                else:
                    scale, shift = density_balance(self.inv_inter,
                                                   exponent=ep.green_exponent,
                                                   red_ratio=1.0,
                                                   blue_ratio=1.0,
                                                   pivot=ep.pivot,
                                                   out_brightness=ep.out_brightness)
                    self.ratio_inter = apply_gain(self.inv_inter, scale['values'])
                    self.ratio_inter = apply_addition(self.ratio_inter, shift['values'])
            else:
                self.ratio_inter = self.inv_inter.copy()

        def proceed():
            self.grade_process(preview)

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def grade_process(self, preview: bool = False):
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
            self.final_preview = self.grade_inter.copy()
            if ep.bw_mode:
                self.final_preview = convert_to_grayscale_from_g(self.final_preview)

        def proceed():
            if ep.tonemap:
                self.tonemap_process(preview)
            else:
                self.previewUpdated.emit()

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def tonemap_process(self, preview: bool = False):
        ep = self.edit_params

        def current():
            self.final_preview = aces_tonemap(self.final_preview, toe=ep.toe)

        def proceed():
            self.previewUpdated.emit()

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def save_final_image(self, output_folder: str = None):
        ep = self.edit_params
        original_path = Path(self.currently_loaded_filename)
        output_path = Path(output_folder)
        final_path = output_path / original_path.with_suffix(".tiff").name
        self.saveInitiated.emit(f"Writing {original_path.name} to {final_path}...")
        print(f"Writing {original_path.name} to {final_path}...", file=sys.stderr)

        def process_and_save():
            working_image = self.source_image
            # pre-inversion
            ffc_image = ep.ffc_image if self.ffc_image else None
            if ffc_image:
                working_image = apply_ffc(working_image, ffc_image)
            # apply crop inset
            if ep.crop_inset < 1.0:
                crop_height = int(self.raw_height * ep.crop_inset)
                crop_width = int(self.raw_width * ep.crop_inset)
                crop_start_y = int(self.raw_height - crop_height) // 2
                crop_end_y = crop_start_y + crop_height
                crop_start_x = int(self.raw_width - crop_width) // 2
                crop_end_x = crop_start_x + crop_width
                working_image = working_image[crop_start_y:crop_end_y,
                                              crop_start_x:crop_end_x]
            working_image = rotate_image(working_image, ep.rotation)
            if ep.base_color is not None and ep.base_color.any():
                wb_adjustment = white_balance(working_image,
                                              rgb_value=ep.base_color)
                working_image = apply_gain(working_image,
                                           wb_adjustment['values'])
            # inversion & ratio
            if not ep.skip_inversion:
                working_image = invert_to_density(working_image)
                red_ratio = ep.red_ratio
                blue_ratio = ep.blue_ratio
                if ep.bw_mode:
                    red_ratio = 1.0
                    blue_ratio = 1.0
                scale, shift = density_balance(working_image,
                                               exponent=ep.green_exponent,
                                               red_ratio=red_ratio,
                                               blue_ratio=blue_ratio,
                                               pivot=ep.pivot,
                                               out_brightness=ep.out_brightness)
                working_image = apply_gain(working_image, scale['values'])
                working_image = apply_addition(working_image, shift['values'])
            # grade
            if not ep.bw_mode:
                gain_tuple = (ep.red_gain, ep.green_gain, ep.blue_gain)
                working_image = apply_gain(working_image, gain_tuple)
                add_tuple = (ep.wb_red, ep.wb_green, ep.wb_blue)
                working_image = apply_addition(working_image, add_tuple)
            if not ep.skip_inversion:
                working_image = density_to_luminance(working_image)
            if ep.bw_mode:
                working_image = convert_to_grayscale_from_g(working_image)
            # tonemap
            if ep.tonemap:
                working_image = aces_tonemap(working_image, toe=ep.toe)
            # convert to sRGB; will provide option for custom colorspace soon
            working_image, _ = convert_to_sRGB(working_image)
            save_image(working_image, final_path, 'f16')
            self.saveFinished.emit(f"Finished writing {original_path.name} to {final_path}.")

        thread = self.threadpool.instantiate_thread(process_and_save)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def load_raw_file(self, image_path):
        # load raw image and resize it for the preview pane
        if image_path:
            # the function to be executed within a separate thread
            def init_func():
                raw = load_raw_image(image_path).astype(np.float32) / 65535.0
                self.currently_loaded_filename = image_path
                if image_path in PHOTO_INDEX:
                    ref = PHOTO_INDEX[image_path]
                    height, width = ref['height'], ref['width']
                    self.raw_width = width
                    self.raw_height = height
                else:
                    height, width, _ = raw.shape
                    self.raw_width = width
                    self.raw_height = height
                    PHOTO_INDEX[image_path] = {
                        "width": width,
                        "height": height,
                    }
                if width > height:
                    self.preview_scale = self.max_preview_size / width
                else:
                    self.preview_scale = self.max_preview_size / height
                # order == filter quality (0 - 5, 0 is fastest)
                scaled_raw = ndimage.zoom(raw, (self.preview_scale,
                                                self.preview_scale, 1), order=1)
                self.rawLoaded.emit(image_path)
                return raw, scaled_raw, image_path

            # the function to be executed upon thread completion
            def post_func(image_data_tuple):
                # we know it must exist now, since previous func added it
                name = image_data_tuple[2]
                ref = PHOTO_INDEX[name]
                self.source_image = image_data_tuple[0]
                self.source_image_small = image_data_tuple[1]
                if "edit_params" in ref:
                    print(f"Loading edit_params from memory for {name}.",
                          file=sys.stderr)
                    self.edit_params = ref['edit_params']
                else:
                    print(f"Loading edit_params from sidecar for {name}.",
                          file=sys.stderr)
                    self.edit_params = load_params_from_sidecar(name)
                    ref["edit_params"] = self.edit_params
                self.editParamsUpdated.emit()
                self.pre_inv_process(preview=True)

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

    def load_ffc_file(self, image_path):
        # load raw file for flat-field correction
        if image_path:
            def init_func():
                raw = load_raw_image(image_path).astype(np.float32) / 65535.0
                return raw

            def post_func(image_data):
                self.ffc_image = image_data
                self.ffc_image_small = ndimage.zoom(image_data,
                                                    (self.preview_scale,
                                                     self.preview_scale, 1),
                                                    order=0)

            def error_func(e):
                exctype, value, error = e
                QMessageBox.critical(self, "Error processing FFC image!", error)

            thread = self.threadpool.instantiate_thread(init_func)
            self.threadpool.active_threads[thread.thread_id] = thread
            thread.signals.result.connect(post_func)
            thread.signals.error.connect(error_func)
            thread.signals.finished.connect(self.threadpool.remove_thread)
            self.threadpool.start(thread)

    @Slot()
    def pick_color_from_image(self, point_x, point_y,
                              stage: Stage, picker: ColorPicker):
        if stage == Stage.PRE_INV:
            value = average_sample_point(self.rotation_inter,
                                         point_x, point_y, 16)
            picker.update_color(value)
        elif stage == Stage.INV:
            value = average_sample_point(self.pre_inv_inter,
                                         point_x, point_y, 16)
            picker.update_color(value)
        elif stage == Stage.RATIO:
            value = average_sample_point(self.inv_inter,
                                         point_x, point_y, 16)
            picker.update_color(value)
        elif stage == Stage.GRADE:
            value = average_sample_point(self.ratio_inter,
                                         point_x, point_y, 16)
            picker.update_color(value)
        else:
            value = average_sample_point(self.grade_inter,
                                         point_x, point_y, 16)
        picker.update_color(value)
