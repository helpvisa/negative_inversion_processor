import sys
import json
import numpy as np


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
