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

import copy
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy.typing as npt


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
    crop_shift_h: float = 0.0
    crop_shift_v: float = 0.0

    # --- inversion parameters
    skip_inversion: bool = False
    bw_mode: bool = False
    # xy coord for sampling base colour of film
    base_color_xy: Optional[Tuple[int, int]] = None
    base_color: Optional[npt.NDArray[Tuple[float, float, float]]] = None
    # inversion points and ratios
    pivot: float = 0.745
    red_ratio: float = 1.36
    blue_ratio: float = 0.86
    green_exponent: float = 1.0

    # --- final user grade
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    # wb picker
    wb_xy: Optional[Tuple[int, int]] = None
    wb_reference: Optional[npt.NDArray[Tuple[float, float, float]]] = None
    # with rgb values for additional fine-tuning
    wb_red: float = 0.0
    wb_green: float = 0.0
    wb_blue: float = 0.0

    # --- brightness, look, and tonemapping
    final_exposure: float = 1.0
    tonemap: bool = False
    toe: float = 0.20

    def copy(self) -> "EditParams":
        """
        Return a deep copy of the current parameters.
        Helpful for navigating multithreaded situations.
        """
        return copy.deepcopy(self)
