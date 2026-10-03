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
import json
import numpy as np
from edit_params import EditParams
from global_vars import PHOTO_INDEX


def update_sidecar(filepath):
    if filepath and filepath in PHOTO_INDEX:
        # construct sidecar
        ep = PHOTO_INDEX[filepath]['edit_params']
        payload = {
            "ffc_image": ep.ffc_image,
            "rotation": ep.rotation,
            "crop_inset": ep.crop_inset,
            "crop_shift_h": ep.crop_shift_h,
            "crop_shift_v": ep.crop_shift_v,
            "skip_inversion": ep.skip_inversion,
            "bw_mode": ep.bw_mode,
            "base_color_xy": ep.base_color_xy,
            "base_color": ep.base_color.tolist() if \
                          ep.base_color is not None and \
                          ep.base_color.any() else None,
            "pivot": ep.pivot,
            "red_ratio": ep.red_ratio,
            "blue_ratio": ep.blue_ratio,
            "green_exponent": ep.green_exponent,
            "red_gain": ep.red_gain,
            "green_gain": ep.green_gain,
            "blue_gain": ep.blue_gain,
            "wb_xy": ep.wb_xy,
            "wb_reference": ep.wb_reference.tolist() if \
                            ep.wb_reference is not None and \
                            ep.wb_reference.any() else None,
            "wb_red": ep.wb_red,
            "wb_green": ep.wb_green,
            "wb_blue": ep.wb_blue,
            "slope": ep.slope,
            "offset": ep.offset,
            "power": ep.power,
            "final_exposure": ep.final_exposure,
            "tonemap": ep.tonemap,
        }
        # construct destination path
        final_path = filepath + ".nip.json"
        with open(final_path, 'w') as output:
            json.dump(payload, output)
            print("Updated sidecar.", file=sys.stderr)
    else:
        print("No file found in index.", file=sys.stderr)


def load_params_from_sidecar(filepath):
    if filepath:
        # load sidecar
        load_path = filepath + ".nip.json"
        ep = EditParams()
        # approach for adding new values to sidecars without breaking things:
        #   - iterate through all keys in sidecar
        #   - match by name, apply logic selectively
        #   - any missing values inherit the default
        # will require rewrite of below code; do this before adding in new
        # tonemapping options / controls
        try:
            with open(load_path, 'r') as f:
                sidecar = json.loads(f.read())
                for key in sidecar:
                    if key == 'base_color_xy':
                        print(f"Setting value for {key}", file=sys.stderr)
                        bc_xy_list = sidecar[key]
                        bc_xy_tuple = (bc_xy_list[0], bc_xy_list[1]) if bc_xy_list else None
                        ep.base_color_xy = bc_xy_tuple
                    elif key == 'base_color':
                        print(f"Setting value for {key}", file=sys.stderr)
                        bc_list = sidecar['base_color']
                        bc_tuple = np.array([bc_list[0], bc_list[1], bc_list[2]]) if bc_list else None
                        ep.base_color = bc_tuple
                    elif key == 'wb_xy':
                        print(f"Setting value for {key}", file=sys.stderr)
                        wb_xy_list = sidecar['wb_xy']
                        wb_xy_tuple = (wb_xy_list[0], wb_xy_list[1]) if wb_xy_list else None
                        ep.wb_xy = wb_xy_tuple
                    elif key == 'wb_reference':
                        print(f"Setting value for {key}", file=sys.stderr)
                        wb_list = sidecar['wb_reference']
                        wb_tuple = np.array([wb_list[0], wb_list[1], wb_list[2]]) if wb_list else None
                        ep.wb_reference = wb_tuple
                    else:
                        print(f"Setting value for {key}", file=sys.stderr)
                        setattr(ep, key, sidecar[key])
                return ep
        except FileNotFoundError:
            print("No sidecar file found. Returning blank edit params.",
                  file=sys.stderr)
            return EditParams()
        except KeyError as e:
            print(f"Sidecar is missing a key and may be for another version of NIP: {e}",
                  file=sys.stderr)
            return EditParams()
    else:
        print("No file path specified.", file=sys.stderr)
        return EditParams()
