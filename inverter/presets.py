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

import os
import sys
import json
import numpy as np
from dataclasses import dataclass, asdict
from edit_params import EditParams
from global_vars import PHOTO_INDEX


def save_preset(filepath, adjustments):
    with open(filepath, 'w') as output:
        # remove any entries that are labeled as "no_preset"
        output_arr = []
        for adjustment in adjustments:
            if not 'no_preset' in adjustment:
                output_arr.append(adjustment)
            else:
                print("Stripping no_preset adjustment.", file=sys.stderr)
        # convert any numpy value to a regular old float so we can serialize
        json.dump(output_arr, output, default=lambda obj: obj.item() if
                  isinstance(obj, np.generic) else obj)
        print(f"Preset saved to {filepath}!", file=sys.stderr)


def load_preset(filepath):
    with open(filepath, 'r', encoding="utf-8") as preset:
        return json.loads(preset.read())
