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
from PySide6.QtCore import QObject, Signal
from deepdiff import DeepDiff
from edit_params import EditParams
from processing import (rotate_image,
                        white_balance, density_balance,
                        invert_to_density, density_to_luminance,
                        convert_to_grayscale_from_g,
                        apply_addition, apply_gain)


# we must subclass QObject to leverage signals
class ProcessingPipeline(QObject):
    previewUpdated = Signal()

    def __init__(self, source_image_data: np.ndarray, scale=1.0):
        super().__init__()
        self.edit_params = EditParams()
        self.source_image = source_image_data
        self.scale_up_factor = scale
        # track a median copy for each stage of the pipeline
        self.pre_inv_inter = []
        self.inv_inter = []
        self.ratio_inter = []
        self.grade_inter = []
        # initialize the preview as the raw loaded file
        self.final_preview = source_image_data
        # and kick off an initial edit
        self.pre_inv_process()

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
        self.pre_inv_inter = rotate_image(self.source_image,
                                          ep.rotation)
        if not ep.bw_mode:
            if ep.base_color_xy:
                wb_adjustment = white_balance(self.pre_inv_inter,
                                              custom_wb_point=ep.base_color_xy)
                self.pre_inv_inter = apply_gain(self.pre_inv_inter,
                                                wb_adjustment['values'])
        else:
            self.pre_inv_inter = convert_to_grayscale_from_g(self.pre_inv_inter)
        self.inv_process()

    def inv_process(self):
        ep = self.edit_params
        if not ep.skip_inversion:
            self.inv_inter = invert_to_density(self.pre_inv_inter)
        else:
            self.inv_inter = self.pre_inv_inter.copy()
        self.ratio_process()

    def ratio_process(self):
        ep = self.edit_params
        if not ep.skip_inversion and not ep.bw_mode:
            scale, shift = density_balance(self.inv_inter,
                                           exponent=ep.green_exponent,
                                           red_ratio=ep.red_ratio,
                                           blue_ratio=ep.blue_ratio)
            self.ratio_inter = apply_gain(self.inv_inter, scale['values'])
            self.ratio_inter = apply_addition(self.ratio_inter, shift['values'])
        else:
            self.ratio_inter = self.inv_inter.copy()
        self.grade_process()
        
    def grade_process(self):
        ep = self.edit_params
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
        self.previewUpdated.emit()
