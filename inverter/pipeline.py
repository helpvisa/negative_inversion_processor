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
from deepdiff import DeepDiff
from edit_params import EditParams


class ProcessingPipeline():
    def __init__(self, source_image_data: np.ndarray):
        self.edit_params = EditParams()
        self.source_image = source_image_data
        # track a median copy for each stage of the pipeline
        self.pre_inv_inter = []
        self.inv_inter = []
        self.grade_inter = []

    # perform a deep comparison of self.edit_params and the new EditParams
    # passed in to determine where in the pipeline the reprocess needs to occur
    def reprocess_image(self, new_edit_params):
        if (self.edit_params != new_edit_params):
            difference = DeepDiff(self.edit_params, new_edit_params)
            print(difference)
            self.edit_params = new_edit_params
