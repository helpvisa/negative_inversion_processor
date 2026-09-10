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
from PIL import ImageCms
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox
from deepdiff import DeepDiff
from threads import WorkerThreadPool
from edit_params import EditParams, Stage
from processing import (load_raw_image, save_image, rotate_image,
                        white_balance, density_balance,
                        average_sample_point,
                        invert_to_density, to_density, density_to_luminance,
                        convert_to_grayscale_from_g,
                        apply_addition, apply_gain, apply_division, apply_ffc)
from custom_widgets import ColorPicker
from colour_management import (filmic_tonemap, aces_tonemap, convert_to_sRGB)
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

    # perform a deep comparison of self.edit_params and the new EditParams
    # passed in to determine where in the pipeline the reprocess needs to occur
    def process_image(self, new_edit_params, force_refresh=False):
        if force_refresh:
            self.edit_params = new_edit_params
            self.pre_inv_process(preview=True)
        else:
            difference = DeepDiff(self.edit_params, new_edit_params)
            if difference:
                update_sidecar(self.currently_loaded_filename)
                print(difference, file=sys.stderr)
                self.edit_params = new_edit_params
                # update the global photo index
                PHOTO_INDEX[self.currently_loaded_filename]['edit_params'] = self.edit_params
                # eventually, we check the difference to update only where
                # a change has actually occurred
                changes = {}
                type_changes = {}
                if 'values_changed' in difference:
                    changes = difference['values_changed']
                if 'type_changes' in difference:
                    type_changes = difference['type_changes']
                # change occurred in pre_inv_process
                pre_inv_changes = ['root.ffc_image', 'root.rotation',
                                   'root.crop_inset', 'root.crop_shift_v',
                                   'root.crop_shift_h', 'root.base_color',
                                   'root.base_color[0]', 'root.base_color[1]',
                                   'root.base_color[2]']
                inv_changes = ['root.skip_inversion']
                ratio_changes = ['root.pivot', 'root.red_ratio', 'root.blue_ratio',
                                 'root.green_exponent', 'root.bw_mode']
                grade_changes = ['root.red_gain', 'root.green_gain',
                                 'root.blue_gain', 'root.wb_red', 'root.wb_green',
                                 'root.wb_blue']
                tonemap_changes = ['root.final_exposure', 'root.tonemap',
                                   'root.toe']
                if any(key in changes for key in pre_inv_changes) or \
                   any(key in type_changes for key in pre_inv_changes):
                    self.pre_inv_process(preview=True)
                if any(key in changes for key in inv_changes):
                    self.inv_process(preview=True)
                if any(key in changes for key in ratio_changes):
                    self.ratio_process(preview=True)
                if any(key in changes for key in grade_changes):
                    self.grade_process(preview=True)
                if any(key in changes for key in tonemap_changes):
                    self.tonemap_process(preview=True)
            

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
                crop_start_y += int(ep.crop_shift_v * self.raw_height * self.preview_scale)
                if crop_start_y < 0:
                    crop_start_y = 0
                if crop_start_y > self.raw_height:
                    crop_start_y = self.raw_height
                crop_end_y = crop_start_y + crop_height
                crop_end_y += int(ep.crop_shift_v * self.raw_height * self.preview_scale)
                if crop_end_y < 0:
                    crop_end_y = 0
                if crop_end_y > self.raw_height:
                    crop_end_y = self.raw_height
                crop_start_x = int(self.raw_width * self.preview_scale - crop_width) // 2
                crop_start_x += int(ep.crop_shift_h * self.raw_width * self.preview_scale)
                if crop_start_x < 0:
                    crop_start_x = 0
                if crop_start_x > self.raw_width:
                    crop_start_x = self.raw_width
                crop_end_x = crop_start_x + crop_width
                crop_end_x += int(ep.crop_shift_h * self.raw_width * self.preview_scale)
                if crop_end_x < 0:
                    crop_end_x = 0
                if crop_end_x > self.raw_width:
                    crop_end_x = self.raw_width
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
                self.inv_inter = to_density(self.pre_inv_inter)

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
                                                   out_brightness=ep.pivot)
                    self.ratio_inter = apply_gain(self.inv_inter, scale['values'])
                    self.ratio_inter = apply_addition(self.ratio_inter, shift['values'])
                else:
                    scale, shift = density_balance(self.inv_inter,
                                                   exponent=ep.green_exponent,
                                                   red_ratio=1.0,
                                                   blue_ratio=1.0,
                                                   pivot=ep.pivot,
                                                   out_brightness=ep.pivot)
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
                if ep.skip_inversion:
                    self.grade_inter = apply_division(self.grade_inter,
                                                      gain_tuple)
                else:
                    self.grade_inter = apply_gain(self.grade_inter, gain_tuple)
                # tune (addition)
                add_tuple = (ep.wb_red, ep.wb_green, ep.wb_blue)
                # if ep.skip_inversion:
                    # add_tuple = tuple(val * -1 for val in add_tuple)
                self.grade_inter = apply_addition(self.grade_inter, add_tuple)
            # then convert to luminance
            if not ep.skip_inversion:
                self.grade_inter = density_to_luminance(self.grade_inter)
            else:
                self.grade_inter = density_to_luminance(self.grade_inter,
                                                        scale=1.0)
            if ep.bw_mode:
                self.grade_inter = convert_to_grayscale_from_g(self.grade_inter)

        def proceed():
            self.tonemap_process(preview)

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def tonemap_process(self, preview: bool = False):
        ep = self.edit_params

        def current():
            # apply final makeup gain
            if not ep.bw_mode:
                exposure_tuple = (ep.final_exposure,
                                  ep.final_exposure,
                                  ep.final_exposure)
                self.final_preview = apply_gain(self.grade_inter, exposure_tuple)
            else:
                self.final_preview = self.grade_inter * ep.final_exposure
            if ep.tonemap:
                self.final_preview = aces_tonemap(self.final_preview, toe=ep.toe)

        def proceed():
            self.previewUpdated.emit()

        thread = self.threadpool.instantiate_thread(current)
        self.threadpool.active_threads[thread.thread_id] = thread
        thread.signals.result.connect(proceed)
        thread.signals.finished.connect(self.threadpool.remove_thread)
        self.threadpool.start(thread)

    def save_final_image(self, image_to_save: str, output_path: str = None):
        self.saveInitiated.emit(f"Writing {image_to_save} to {output_path}...")
        print(f"Writing {image_to_save} to {output_path}...", file=sys.stderr)

        def process_and_save():
            working_image = load_raw_image(image_to_save).astype(np.float32) / 65535.0
            current_height, current_width, _ = working_image.shape
            ep = EditParams()
            if image_to_save in PHOTO_INDEX:
                ref = PHOTO_INDEX[image_to_save]
                if "edit_params" in ref:
                    print(f"Loading edit_params from memory for {image_to_save}.",
                          file=sys.stderr)
                    ep = ref['edit_params']
                else:
                    print(f"Loading edit_params from sidecar for {image_to_save}.",
                          file=sys.stderr)
                    ep = load_params_from_sidecar(image_to_save)
            # pre-inversion
            # ffc_image = ep.ffc_image if self.ffc_image else None
            # if ffc_image:
                # working_image = apply_ffc(working_image, ffc_image)
            # apply crop inset
            if ep.crop_inset < 1.0:
                crop_height = int(current_height * ep.crop_inset)
                crop_width = int(current_width * ep.crop_inset)
                crop_start_y = int(current_height - crop_height) // 2
                crop_start_y += int(ep.crop_shift_v * current_width)
                if crop_start_y < 0:
                    crop_start_y = 0
                if crop_start_y > current_height:
                    crop_start_y = current_height
                crop_end_y = crop_start_y + crop_height
                crop_end_y += int(ep.crop_shift_v * current_width)
                if crop_end_y < 0:
                    crop_end_y = 0
                if crop_end_y > current_height:
                    crop_end_y = current_height
                crop_start_x = int(current_width - crop_width) // 2
                crop_start_x += int(ep.crop_shift_h * current_width)
                if crop_start_x < 0:
                    crop_start_x = 0
                if crop_start_x > current_width:
                    crop_start_x = current_width
                crop_end_x = crop_start_x + crop_width
                crop_end_x += int(ep.crop_shift_h * current_width)
                if crop_end_x < 0:
                    crop_end_x = 0
                if crop_end_x > current_width:
                    crop_end_x = current_width
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
                                               out_brightness=ep.pivot)
                working_image = apply_gain(working_image, scale['values'])
                working_image = apply_addition(working_image, shift['values'])
            else:
                working_image = to_density(working_image)
            # grade
            if not ep.bw_mode:
                gain_tuple = (ep.red_gain, ep.green_gain, ep.blue_gain)
                if ep.skip_inversion:
                    working_image = apply_division(working_image, gain_tuple)
                else:
                    working_image = apply_gain(working_image, gain_tuple)
                add_tuple = (ep.wb_red, ep.wb_green, ep.wb_blue)
                # if ep.skip_inversion:
                    # add_tuple = tuple(val * -1 for val in add_tuple)
                working_image = apply_addition(working_image, add_tuple)
            if not ep.skip_inversion:
                working_image = density_to_luminance(working_image)
            else:
                working_image = density_to_luminance(working_image, scale=1.0)
            if ep.bw_mode:
                working_image = convert_to_grayscale_from_g(working_image)
            # tonemap
            if not ep.bw_mode:
                exposure_tuple = (ep.final_exposure,
                                  ep.final_exposure,
                                  ep.final_exposure)
                working_image = apply_gain(working_image, exposure_tuple)
            else:
                working_image *= ep.final_exposure
            if ep.tonemap:
                working_image = aces_tonemap(working_image, toe=ep.toe)
            # convert to sRGB; will provide option for custom colorspace soon
            if working_image.ndim < 3:
                working_image = np.stack((working_image,
                                          working_image,
                                          working_image), axis=-1)
            working_image, sRGB_profile = convert_to_sRGB(working_image)
            save_profile = ImageCms.ImageCmsProfile(sRGB_profile).tobytes()
            save_image(working_image, output_path, 'f16', save_profile)
            self.saveFinished.emit(f"Finished writing {image_to_save} to {output_path}.")

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
                self.process_image(self.edit_params, force_refresh=True)

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
