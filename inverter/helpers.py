import os
import sys
from pathlib import Path


def get_resource_path(relative_path):
    """
    Get absolute path to resource, relative current executable.

    Required for exporting bundled binaries with PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def determine_m43(filepath, raw):
    """
    Determine if RAW file is from a m43 sensor, and if it's a stacked hi-res
    file.

    This is important because m43 sensors seem more prone to mazing, and require
    specific demosaicing instructions when loaded. These adjustments have to
    be disabled for hi-res stacks however, making my job fun.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    print(f"SUFFIX: {suffix}", file=sys.stderr)

    if suffix not in {".orf", ".ori", ".rw2"}:
        print("Suffix not in recognised m43 sensor suffix list.",
              file=sys.stderr)
        return False
    # ORIs are always backup standard-res raws
    if suffix == ".ori":
        return True

    # determine sensor margins
    # hi-res files should have their optical black margins stripped
    # basically: regular files show a raw w/h greater than their usable w/h
    # and hence also a left / top margin that will be greater than 0
    # this only seems to work correctly if prefaced with *not*?
    # does libraw pre-crop? I'm confused!
    return not (raw.sizes.raw_width > raw.sizes.width
                or raw.sizes.raw_height > raw.sizes.height
                or raw.sizes.left_margin > 0
                or raw.sizes.top_margin > 0)
