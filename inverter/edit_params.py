import copy
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple


class Stage(IntEnum):
    PRE_INV = 0
    INV     = 1
    RATIO   = 2
    GRADE   = 3
    FINAL   = 4


@dataclass
class EditParams:
    # correct the current image
    ffc_image: str = None
    rotation: int = 0
    crop_inset: float = 1.0

    # --- inversion parameters
    skip_inversion: bool = False
    bw_mode: bool = False
    # xy coord for sampling base colour of film
    base_color_xy: Optional[Tuple[int, int]] = None
    base_color: Optional[Tuple[float, float, float]] = None
    # inversion points and ratios
    pivot: float = 0.745
    red_ratio: float = 1.36
    blue_ratio: float = 0.86
    green_exponent: float = 1.5
    out_brightness: float = 0.745
    # pickers for estimating the ratios
    lo_gray_xy: Optional[Tuple[int, int]] = None
    lo_gray: Optional[Tuple[float, float, float]] = None
    hi_gray_xy: Optional[Tuple[int, int]] = None
    hi_gray: Optional[Tuple[float, float, float]] = None

    # --- final user grade
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    # wb picker
    wb_xy: Optional[Tuple[int, int]] = None
    wb_reference: Optional[Tuple[float, float, float]] = None
    # with rgb values for additional fine-tuning
    wb_red: float = 0.0
    wb_green: float = 0.0
    wb_blue: float = 0.0
    tonemap: bool = False
    toe: float = 0.0

    def copy(self) -> "EditParams":
        """
        Return a deep copy of the current parameters.
        Helpful for navigating multithreaded situations.
        """
        return copy.deepcopy(self)
