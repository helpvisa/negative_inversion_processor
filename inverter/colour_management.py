import numpy as np
import PyOpenColorIO as ocio
from colour import RGB_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES, CCS_ILLUMINANTS
from PIL import ImageCms
from processing import convert_to_grayscale
from global_vars import REC2020_WEIGHTS


ACES_I_MAT = [[0.59719, 0.35458, 0.04823],
              [0.07600, 0.90834, 0.01566],
              [0.02840, 0.13383, 0.83777]]
ACES_O_MAT = [[ 1.60475, -0.53108, -0.07367],
              [-0.10208,  1.10813, -0.00605],
              [-0.00327, -0.07276,  1.07602]]


def aces_tonemap(image_data):
    """
    ACES-style tonemapping from HDR to 0.0 <-> 1.0.
    Uses an approximate curve that doesn't require altering our primaries.

    See https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    (^ MIT License)
    """
    global ACES_I_MAT, ACES_O_MAT
    # aces_in = np.matmul(image_data, ACES_I_MAT)
    a = image_data * (image_data + 0.0245786) - 0.000090537
    b = image_data * (0.983729 * image_data + 0.4329510) + 0.238081
    return a / b
    # return np.matmul(a / b, ACES_O_MAT)


def reinhard_tonemap(image_data, key=1.0):
    """
    Basic tonemapping from HDR to 0.0 <-> 1.0.
    Can cause strange hue shifts due to film carrier creeping into image.
    Key value represents max luminance.
    """
    lum = convert_to_grayscale(image_data, REC2020_WEIGHTS)
    log_lum = np.log(lum + 1e-8)
    log_mean = np.mean(log_lum)
    log_avg_lum = np.exp(log_mean)
    scaled_lum = lum * (key / log_avg_lum)
    compressed_lum = scaled_lum / (1 + scaled_lum)
    final_scale = compressed_lum / (lum + 1e-8)
    return image_data * final_scale[:, :, np.newaxis]


def convert_to_sRGB(image_data):
    """
    Convert the given image data from Linear Rec2020 to sRGB.
    """
    sRGB_conversion = RGB_to_XYZ(
        image_data,
        RGB_COLOURSPACES['ITU-R BT.2020'],
        CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'],
        apply_cctf_decoding=False
    )
    sRGB_image = XYZ_to_RGB(
        sRGB_conversion,
        RGB_COLOURSPACES['sRGB'],
        apply_cctf_encoding=True
    )
    # create and return an sRGB colour profile along with it
    sRGB_profile = ImageCms.createProfile("sRGB")
    return sRGB_image, sRGB_profile


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
