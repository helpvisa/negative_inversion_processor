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
import math
import numpy as np
import PyOpenColorIO as ocio
from colour import (RGB_to_XYZ, XYZ_to_RGB, RGB_to_RGB,
                    RGB_COLOURSPACES, CCS_ILLUMINANTS,
                    cctf_encoding, cctf_decoding, read_LUT)
from PIL import ImageCms


def get_supported_colour_profiles():
    return sorted(RGB_COLOURSPACES.keys())


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


def sigmoid_paper_response(value, magnitude,
                           film_fog, film_power,
                           paper_exp, paper_power) -> float:
    clamped_val = math.max(value, 0.0)
    film_response = math.pow(film_fog + clamped_val, film_power)
    paper_response = magnitude * math.pow(film_response / (paper_exp + film_response),
                                          paper_power)
    if math.isnan(paper_response):
        return magnitude
    return paper_response


def sigmoid_tonemap(image_data,
                    white_target=1.0, black_target=0.0152,
                    contrast=1.5, skew=-0.2, hue_preservation=0.0,
                    red_inset=0.1, green_inset=0.1, blue_inset=0.15,
                    # rotations are in radians (2deg, -1deg, -3deg)
                    red_rot=0.0350, green_rot=-0.0175, blue_rot=-0.0524,
                    purity=0.0):
    """
    Sigmoid-style tonemapper inspired by darktable. Defaults based on "smooth"
    preset.

    https://github.com/darktable-org/darktable/blob/master/src/iop/sigmoid.c
    """
    # calculate reference values
    middle_gray = 0.1845
    film_power = contrast
    paper_power = 1.0
    magnitude = 1.0
    film_fog = 1.0
    paper_exposure = math.pow(film_fog + middle_gray, film_power) * ((magnitude / middle_gray) - 1.0)
    delta = 1e-6
    # slope = 

    red = image_data[:, :, 0]
    green = image_data[:, :, 1]
    blue = image_data[:, :, 2]
    # response_r = red.pow(,)


def linear_to_ACEScct(image_data):
    """
    Convert the given image data's transfer function from linear to ACEScct.

    Notably, uses Linear Rec.2020 primaries to avoid imaginary colours and
    prevent negative values from being introduced into the pipeline when
    converting back to a linear transfer function.

    This is purely a difference in the transfer function
    """
    cct_image = cctf_encoding(image_data, function='ACEScct')
    return cct_image


def acescct_to_linear(image_data):
    """
    Convert the given image data's transfer function from ACEScct to
    linear. See note in `linear_to_ACEScct` above.
    """
    linear_image = cctf_decoding(image_data, function='ACEScct')
    # clamp the minimum value to 0.0; sometimes ACEScct likes to return negative
    # values here when re-linearizing. Is clamping evil? Not sure, but it looks
    # much much better
    return np.maximum(linear_image, 0.0)


def convert_colourspace(image_data, input_colourspace, output_colourspace):
    """
    Convert from any supported 'input_colourspace' to any supported
    'output_colourspace'. Preserves grayscale.
    """
    supported_profiles = get_supported_colour_profiles()
    if input_colourspace in supported_profiles and \
       output_colourspace in supported_profiles:
        is_grayscale = True if image_data.ndim < 3 else False
        input_image = image_data
        if is_grayscale:
            input_image = np.stack((image_data,
                                    image_data,
                                    image_data), axis=-1)
        output_image = RGB_to_RGB(input_image,
                                  RGB_COLOURSPACES[input_colourspace],
                                  RGB_COLOURSPACES[output_colourspace],
                                  chromatic_adaptation_transform="CAT02",
                                  apply_cctf_decoding=True,
                                  apply_cctf_encoding=True)
        if is_grayscale:
            output_image = output_image[:, :, 1]
        return output_image
    print("ERROR: Cannot convert colourspace! Profile not suppported,"
          "Returning original, unconverted image.",
          file=sys.stderr)
    return image_data


def apply_lut(image_data, input_colourspace, output_colourspace, lut_name):
    """
    Apply 3D LUT, accounting for input and output colourspaces.
    """
    lut = read_LUT(lut_name)
    # reshape image_data into flat 0-dimension list
    is_grayscale = True if image_data.ndim < 3 else False
    input_image = image_data
    if is_grayscale:
        input_image = np.stack((image_data,
                                image_data,
                                image_data), axis=-1)
    ref_image = convert_colourspace(input_image,
                                    "ITU-R BT.2020", input_colourspace)
    flat_image = ref_image.reshape(-1, 3)
    graded_image = lut.apply(flat_image)
    # now reshape back
    final_image = convert_colourspace(graded_image.reshape(input_image.shape),
                                      output_colourspace, "ITU-R BT.2020")
    if is_grayscale:
        final_image = final_image[:, :, 1]
    return final_image


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
