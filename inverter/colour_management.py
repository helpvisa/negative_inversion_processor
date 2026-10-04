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

import numpy as np
import PyOpenColorIO as ocio
from colour import (RGB_to_XYZ, RGB_to_RGB, XYZ_to_RGB,
                    RGB_COLOURSPACES, CCS_ILLUMINANTS, cctf_encoding)
from PIL import ImageCms


def aces_tonemap(image_data, toe: float = 0.0):
    """
    ACES-style tonemapping from HDR to 0.0 <-> 1.0.
    Converts to ACES colour space before re-converting back to Linear Rec.2020.

    https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    See THIRD_PARTY_LICENSES.txt for more details.
    """
    # linear rec.2020 -> aces matrix
    mat_in = np.array([
        [0.9465901, 0.0459193, 0.0074905],
        [0.0127789, 0.9828225, 0.0043986],
        [0.0152831, 0.0506650, 0.9340367]
    ])

    # aces -> linear rec.2020 matrix
    mat_out = np.array([
        [0.9691157, 0.0255455, 0.0053380],
        [0.0069938, 0.9806634, 0.0123411],
        [0.0047768, -0.0381346, 1.0333577]
    ])

    v = image_data
    
    # apply linear rec.2020 -> aces
    # mat_in.T makes sure numpy respects array shape
    # skip if image is grayscale (only 1 dimension)
    is_grayscale = image_data.ndim < 3 or image_data.shape[-1] == 1
    if not is_grayscale:
        v = v @ mat_in.T

    a = v * (v + 0.0245786) - (0.000090537 * toe)
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    tonemapped = a / b

    # apply aces -> linear rec.2020
    out_data = tonemapped
    if not is_grayscale:
        out_data = out_data @ mat_out.T
    return np.clip(out_data, 0.0, 1.0)


def linear_to_ACEScc(image_data):
    """
    Convert the given image data from Linear Rec.2020 to ACEScc.
    """
    acescc_image = RGB_to_RGB(
        image_data,
        RGB_COLOURSPACES['ITU-R BT.2020'],
        RGB_COLOURSPACES['ACEScc'],
        chromatic_adaptation_transform='CAT02',
        apply_cctf_decoding=False,  # input is already linear
        apply_cctf_encoding=True
    )
    return acescc_image


def acescc_to_linear(image_data):
    """
    Convert the given image data from Linear Rec.2020 to ACEScc.
    """
    linear_image = RGB_to_RGB(
        image_data,
        RGB_COLOURSPACES['ACEScc'],
        RGB_COLOURSPACES['ITU-R BT.2020'],
        chromatic_adaptation_transform='CAT02',
        apply_cctf_decoding=False,  # input is already logarithmic
        apply_cctf_encoding=True
    )
    return linear_image


def linear_to_sRGB(image_data):
    """
    Convert the given image data from Linear Rec.2020 to sRGB.

    Could probably be refactored to use RGB_to_RGB. Does it matter though?
    """
    xyz_conversion = RGB_to_XYZ(
        image_data,
        RGB_COLOURSPACES['ITU-R BT.2020'],
        CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
        apply_cctf_decoding=False
    )
    sRGB_image = XYZ_to_RGB(
        xyz_conversion,
        RGB_COLOURSPACES['sRGB'],
        apply_cctf_encoding=True
    )
    # create and return an sRGB colour profile along with it
    sRGB_profile = ImageCms.createProfile("sRGB")
    return sRGB_image, sRGB_profile


def convert_to_g22(image_data):
    """
    Convert the given grayscale image data with an sRGB-like gamma 2.2
    transfer function.
    """
    return cctf_encoding(image_data, function='sRGB')


def ocio_load_and_create_config(config_path=None) -> ocio.Config:
    """
    Load an OCIO config file from the given path, or fallback to the
    config specified in the $OCIO environment variable.
    """
    if config_path:
        config = ocio.Config.CreateFromFile(config_path)
    else:
        config = ocio.GetCurrentConfig()
    return config


def ocio_convert_colorspace(image_data, ocio_config, src, dest) -> None:
    """
    Convert from one colourspace to another using a given OCIO configuration.
    """
    processor = ocio_config.getProcessor(src, dest)
    cpu_processor = processor.getDefaultCPUProcessor()
    # directly modifies the image_data
    cpu_processor.applyRGB(image_data)


# icc actually not supported by Baker, so this is totally broken; DON'T USE!
def ocio_bake_icc(ocio_config, src, dest) -> bytes:
    baker = ocio.Baker()
    baker.setConfig(ocio_config)
    baker.setInputSpace(src)
    baker.setTargetSpace(dest)
    baker.setFormat("icc")
    return baker.bake()
