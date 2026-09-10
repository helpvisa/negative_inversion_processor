import os
import sys
import json
from pathlib import Path
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
            "final_exposure": ep.final_exposure,
            "tonemap": ep.tonemap,
            "toe": ep.toe
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
        try:
            with open(load_path, 'r') as f:
                sidecar = json.loads(f.read())
                ep.ffc_image = sidecar['ffc_image']
                ep.rotation = sidecar['rotation']
                ep.crop_inset = sidecar['crop_inset']
                ep.crop_shift_h = sidecar['crop_shift_h']
                ep.crop_shift_v = sidecar['crop_shift_v']
                ep.skip_inversion = sidecar['skip_inversion']
                ep.bw_mode = sidecar['bw_mode']
                bc_xy_list = sidecar['base_color_xy']
                bc_xy_tuple = (bc_xy_list[0], bc_xy_list[1]) if bc_xy_list else None
                ep.base_color_xy = bc_xy_tuple
                bc_list = sidecar['base_color']
                bc_tuple = np.array([bc_list[0], bc_list[1], bc_list[2]]) if bc_list else None
                ep.base_color = bc_tuple
                ep.pivot = sidecar['pivot']
                ep.red_ratio = sidecar['red_ratio']
                ep.blue_ratio = sidecar['blue_ratio']
                ep.green_exponent = sidecar['green_exponent']
                ep.red_gain = sidecar['red_gain']
                ep.green_gain = sidecar['green_gain']
                ep.blue_gain = sidecar['blue_gain']
                wb_xy_list = sidecar['wb_xy']
                wb_xy_tuple = (wb_xy_list[0], wb_xy_list[1]) if wb_xy_list else None
                ep.wb_xy = wb_xy_tuple
                wb_list = sidecar['wb_reference']
                wb_tuple = np.array([wb_list[0], wb_list[1], wb_list[2]]) if wb_list else None
                ep.wb_reference = wb_tuple
                ep.wb_red = sidecar['wb_red']
                ep.wb_green = sidecar['wb_green']
                ep.wb_blue = sidecar['wb_blue']
                ep.final_exposure = sidecar['final_exposure']
                ep.tonemap = sidecar['tonemap']
                ep.toe = sidecar['toe']
                return ep
        except FileNotFoundError:
            print("No sidebar file found. Returning blank edit params.",
                  file=sys.stderr)
            return EditParams()
        except KeyError as e:
            print(f"Sidecar is missing a key and may be for an old version of NIP: {e}",
                  file=sys.stderr)
            return EditParams()
    else:
        print("No file path specified.", file=sys.stderr)
        return EditParams()
