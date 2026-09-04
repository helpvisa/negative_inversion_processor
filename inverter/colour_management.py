import numpy as np
import PyOpenColorIO as ocio
from colour import RGB_to_XYZ, XYZ_to_RGB, RGB_COLOURSPACES, CCS_ILLUMINANTS
from PIL import ImageCms
from processing import convert_to_grayscale
from global_vars import REC2020_WEIGHTS


def noritsu_tonemap(image_data, contrast: float = 1.0, toe: float = 0.0):
    """
    Affect overall tone using 'noritsu-style' tone curve
    https://github.com/rohanpandula/noritsu-tool/blob/main/noritsu/render.py
    """
    v = image_data
    # affect overall tone using 'noritsu-style' tone curve
    # https://github.com/rohanpandula/noritsu-tool/blob/main/noritsu/render.py
    k = contrast
    pts_x = np.array([0.0, 0.08, 0.25, 0.5, 0.75, 0.92, 1.0])
    # s-shaped curve, midtones pinned
    mid = 0.5
    pts_y = mid + (pts_x - mid) * k
    # toe
    pts_y = pts_y + toe * (1.0 - pts_x) * np.exp(-pts_x * 4.0)
    pts_y = np.clip(pts_y, 0.0, 1.0)
    # make sure line is straight (no hills or valleys == monotonic)
    for i in range(1, len(pts_y)):
        if pts_y[i] < pts_y[i - 1]:
            pts_y[i] = pts_y[i - 1]
    v = np.interp(v, pts_x, pts_y)
    return np.clip(v, 0.0, 1.0)


def aces_tonemap(image_data, toe: float = 0.0):
    """
    ACES-style tonemapping from HDR to 0.0 <-> 1.0.
    Converts to ACES colour space before re-converting back to Linear Rec.2020.

    See https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    (^ MIT License)
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


def agx_tonemap(image_data):
    """
    AgX-style tonemapping from HDR to 0.0 <-> 1.0.
    Performed directly in Linear Rec.2020 colour space.
    """
    


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
