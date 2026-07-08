from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class EditParams:
    # rotate / crop the current image
    rotation: int = 0
    crop_inset: float = 100.0

    # xy coord for sampling base colour of film
    base_color_xy: Optional[Tuple[int, int]] = None

    # inversion parameters
    red_ratio: float = 1.36
    blue_ratio: float = 0.86
    green_exponent: float = 1.5
    # pickers for estimating the ratios
    lo_gray_xy: Optional[Tuple[int, int]] = None
    hi_gray_xy: Optional[Tuple[int, int]] = None
    # and an additional color picker for white-balancing (addition) post-invert
    white_balance_xy: Optional[Tuple[int ,int]] = None

    # final user grade
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0

    def copy(self) -> "EditParams":
        """
        Return a deep copy of the current parameters.
        Helpful for navigating multithreaded situations.
        """
        return EditParams(**asdict(self))
