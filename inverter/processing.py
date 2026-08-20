import sys
import rawpy
import tifffile
import numpy as np


def load_raw_image(path):
    print(f"Loading image: {path}", file=sys.stderr)
    with rawpy.imread(path) as raw:
        # read in camera sensor data at 16bpp int with D65 white balance
        d65_balance = raw.daylight_whitebalance
        return raw.postprocess(use_camera_wb=False,
                               user_wb=d65_balance,
                               no_auto_bright=True,
                               half_size=False,
                               output_bps=16,
                               gamma=(1.0, 1.0),
                               output_color=rawpy.ColorSpace.Rec2020)


def save_image(image_data, output_path, tiff_format, icc_profile):
    # default case; save as f32 (input format from processing pipeline)
    image_to_save = image_data
    # set predictor if floating point (default)
    p_val = 3
    if tiff_format == "u8":
        image_data = np.clip(image_data, a_min=0, a_max=1)
        image_to_save = np.multiply(image_data, 255).astype(np.uint8)
        p_val = None
    elif tiff_format == "u16":
        image_data = np.clip(image_data, a_min=0, a_max=1)
        image_to_save = np.multiply(image_data, 65535).astype(np.uint16)
        p_val = None
    elif tiff_format == "f16":
        image_to_save = image_data.astype(np.float16)
    tifffile.imwrite(output_path,
                     image_to_save,
                     compression="zlib",
                     compressionargs={"level":9},
                     predictor=p_val,
                     iccprofile=icc_profile)
    print(f"Saved inversion to {output_path}", file=sys.stderr)


def rotate_image(image_data, direction):
    return np.rot90(image_data, k=direction)


# many of the functions below should probably be altered to directly modify
# image data instead of passing it all around and here and there
# it would save on a lot of memory shuffling!
def apply_addition(image_data, adjustment: tuple[float, float, float]):
    """
    Perform addition / subtraction to a given working image and return the
    result.

    Adjustment is a tuple representing (r, g, b)
    """
    red_new = image_data[:, :, 0].copy() + adjustment[0]
    green_new = image_data[:, :, 1].copy() + adjustment[1]
    blue_new = image_data[:, :, 2].copy() + adjustment[2]
    return np.stack([red_new, green_new, blue_new], axis=2)


def apply_gain(image_data, adjustment: tuple[float, float, float]):
    """
    Perform multiplication on a given working image and return the
    result.

    Adjustment is a tuple representing (r, g, b)
    """
    red_new = image_data[:, :, 0].copy() * adjustment[0]
    green_new = image_data[:, :, 1].copy() * adjustment[1]
    blue_new = image_data[:, :, 2].copy() * adjustment[2]
    return np.stack([red_new, green_new, blue_new], axis=2)


def apply_power(image_data, adjustment: tuple[float, float, float]):
    """
    Raise image channels to a given exponent and return the result.

    Adjustment is a tuple representing (r, g, b)
    """
    red_new = np.pow(image_data[:, :, 0].copy(), adjustment[0])
    green_new = np.pow(image_data[:, :, 1].copy(), adjustment[1])
    blue_new = np.pow(image_data[:, :, 2].copy(), adjustment[2])
    return np.stack([red_new, green_new, blue_new], axis=2)


def divide_by_image(a_data: np.ndarray, b_data: np.ndarray):
    """
    Divide one array of image data by another.
    Used for flat-field correction.
    """
    result = np.zeros_like(a_data, dtype=float)
    np.divide(a_data, b_data,
              out=result,
              where=(b_data != 0))
    return result


def apply_ffc(image_data: np.ndarray, ffc_data: np.ndarray,
              strength: float = 1.0) -> np.ndarray:
    print(f"Applying flat-field correction (strength {strength})",
          file=sys.stderr)
    scalar = np.median(ffc_data)
    corrected_image = divide_by_image(image_data,
                                      ffc_data * strength)
    # apply gain correction
    corrected_image = corrected_image * scalar
    return corrected_image


def average_sample_point(image_data, sample_x, sample_y, kernel):
    """
    Average a box around a given sample point and return the value
    as an f32 numpy array.
    """
    offset = int(kernel / 2)
    x_min = int(sample_x - offset)
    x_max = int(sample_x + offset)
    y_min = int(sample_y - offset)
    y_max = int(sample_y + offset)
    average = np.array([0, 0, 0], dtype=np.float32)
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if (y > 0 and y < image_data.shape[0]) and \
               (x > 0 and x < image_data.shape[1]):
                average += image_data[y, x]
    return average / (kernel * kernel)
    

def invert_to_density(image_data):
    """
    Invert a given film negative into a logarithmic density space where it can
    be further corrected before having its densities mapped back to linear
    luminance.
    """
    # we calculate density from "transmittance" using log10(1/x)
    # see https://abpy.github.io/2023/08/20/color-neg.html
    # clip rgb values to avoid discolouration outside the negative itself,
    # which should theoretically still be within a 0 - 1 range at this point
    return np.log10(1 / np.clip(image_data, a_min=1e-3, a_max=1.0))


def density_to_luminance(image_data, scale=0.01):
    """
    Map an image from "density space" into its final luminance values.

    This often makes the film carrier look insane. Please ignore this.
    """
    return np.power(10, image_data) * scale


def convert_to_grayscale(image_data, colourspace_weights):
    return np.dot(image_data, colourspace_weights)


def convert_to_grayscale_from_g(image_data):
    return image_data[:, :, 1]


def find_brightest_luminance_spot(image_data, colourspace_weights):
    # find max luminance value in region
    pre_lum_values = np.dot(image_data, colourspace_weights)
    # and find point of maximum luminance
    max_lum_y, max_lum_x = np.unravel_index(np.argmax(pre_lum_values),
                                            pre_lum_values.shape)
    # now grab the rgb values at that point
    return average_sample_point(image_data, max_lum_x, max_lum_y, 16)


def find_darkest_luminance_spot(image_data, colourspace_weights):
    # find min luminance value in region
    pre_lum_values = np.dot(image_data, colourspace_weights)
    # and find point of min luminance
    max_lum_y, max_lum_x = np.unravel_index(np.argmin(pre_lum_values),
                                            pre_lum_values.shape)
    # now grab the rgb values at that point
    return average_sample_point(image_data, max_lum_x, max_lum_y, 16)


# TODO: this one needs some work
def shift_blacks(image_data, analysis_inset):
    # inset the region to avoid out of bounds values (negative carrier)
    height, width, channels = image_data.shape
    n_area_width = int(width * analysis_inset)
    n_area_height = int(height * analysis_inset)
    start_x = (width - n_area_width) // 2
    end_x = start_x + n_area_width
    start_y = (height - n_area_height) // 2
    end_y = start_y + n_area_height
    region = [[start_x, start_y], [end_x, end_y]]
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    lowest_rgb_value = np.argmax(analysis_region)
    if lowest_rgb_value < 0.0:
        print("PROCESS: negative values found, black level shifted back above zero")
        y, x, channel = np.unravel_index(lowest_rgb_value, analysis_region.shape)
        min_val = analysis_region[y, x, channel]
        image_data -= min_val
    adjustment = {
        "type": "shift_blacks",
        "inset": analysis_inset
    }
    return adjustment


def white_balance(image_data, region=None, colourspace_weights=None,
                  custom_wb_point=None, mode='mult'):
    """
    White balance a "brightest point" visible within the bounding box of the
    provided region. This becomes the inverted film's new "black point".

    In most cases, this is none other than the unexposed film base itself.

    Return a dictionary reprsenting the type of adjustment and its values.
    """
    max_rgb_values = None
    if custom_wb_point:
        print(f"CUSTOM: Custom balance point: {custom_wb_point}",
              file=sys.stderr)
        max_rgb_values = average_sample_point(image_data,
                                              custom_wb_point[0],
                                              custom_wb_point[1],
                                              16)
    elif region:
        # create pre-inversion analysis region
        analysis_region = image_data[region[0][1]:region[1][1],
                                     region[0][0]:region[1][0]]
        max_rgb_values = find_brightest_luminance_spot(analysis_region,
                                                       colourspace_weights)
    # determine reference exponents for balance
    factor = max_rgb_values[1]
    if mode == 'add':
        rc = factor - max_rgb_values[0]
        gc = 0
        bc = factor - max_rgb_values[2]
    else:
        rc = factor / max_rgb_values[0]
        gc = 1.0
        bc = factor / max_rgb_values[2]
    print(f"ADJUSTMENT: Balancing ({mode} mode)\n"
          f"            RED:   {rc}\n"
          f"            GREEN: {gc}\n"
          f"            BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": mode,
        "values": (rc, gc, bc)
    }
    return adjustment


def density_balance(image_data, region=None, exponent=1.5, ref_point_in=None,
                    red_ratio=1.36, blue_ratio=0.86):
    """
    Multiply the individual colour channels until equalized at a given point.
    Neutralizes colour casts in highlights and shadows.
    """
    ref_in = None
    # introduce facility to use a baseline value from another image here?
    if ref_point_in:
        print(f"CUSTOM: Custom reference point: {ref_point_in}",
              file=sys.stderr)
        ref_in = image_data[ref_point_in[1], ref_point_in[0]]
    # elif region:
        # analysis_region = image_data[region[0][1]:region[1][1],
                                     # region[0][0]:region[1][0]]
        # estimate a reference median point in the image
        # r_med = np.median(analysis_region[:, :, 0])
        # g_med = np.median(analysis_region[:, :, 1])
        # b_med = np.median(analysis_region[:, :, 2])
        # ref_in = [r_med, g_med, b_med]
    else:
        # r_med = np.median(image_data[:, :, 0])
        # g_med = np.median(image_data[:, :, 1])
        # b_med = np.median(image_data[:, :, 2])
        # ref_in = [r_med, g_med, b_med]
        ref_in = np.array([0.565, 0.565, 0.565])
    # target output value (ref_in -> ref_out)
    # eventually, we will want to make this tweakable instead of
    # tweaking the post-inversion exposure with gain
    ref_out = np.array([0.565, 0.565, 0.565])
    # use ratios to determine channel exponents
    rexp = red_ratio * exponent
    gexp = exponent
    bexp = blue_ratio * exponent
    print(f"ADJUSTMENT: Density scale:\n"
          f"            RED:   {rexp}\n"
          f"            GREEN: {gexp}\n"
          f"            BLUE:  {bexp}",
          file=sys.stderr)
    # determine per-channel offset factor
    radd = ref_out[0] - (ref_in[0] * rexp)
    gadd = ref_out[1] - (ref_in[1] * gexp)
    badd = ref_out[2] - (ref_in[2] * bexp)
    print(f"ADJUSTMENT: Density shift:\n"
          f"            RED:   {radd}\n"
          f"            GREEN: {gadd}\n"
          f"            BLUE:  {badd}",
          file=sys.stderr)
    # but we must actually return two adjustments for a "scale-and-shift"
    # scale must occur first
    scale_adjustment = {
        "type": "mult",
        "values": (rexp, gexp, bexp)
    }
    shift_adjustment = {
        "type": "add",
        "values": (radd, gadd, badd)
    }
    return scale_adjustment, shift_adjustment


def normalize_image(image_data, colourspace_weights, analysis_inset=0.5):
    """
    Normalize image data so that it stays between 0 and 1 (in floating point).

    If colourspace_weights are not provided, we default to Rec.2020 weights.
    """
    # we'll need to inset the normalization region to avoid out of bounds
    # values caused by negative carriers on the edge of the image
    height, width, channels = image_data.shape
    n_area_width = int(width * analysis_inset)
    n_area_height = int(height * analysis_inset)
    start_x = (width - n_area_width) // 2
    end_x = start_x + n_area_width
    start_y = (height - n_area_height) // 2
    end_y = start_y + n_area_height
    region = [[start_x, start_y], [end_x, end_y]]

    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    lum_values = find_brightest_luminance_spot(analysis_region,
                                               colourspace_weights)
    factor = np.max(lum_values)
    image_data /= factor
    adjustment={
        "type": "norm",
        "values": colourspace_weights,
        "inset": analysis_inset
    }
    return adjustment


def process_all_adjustments(image_data, adjustments):
    print("INFO: Applying all adjustments to create final image...",
          file=sys.stderr)
    working_data = image_data.copy()
    for adjustment in adjustments:
        type = adjustment["type"]
        if type == "invert_to_density":
            print("PROCESS: invert to density",
                  file=sys.stderr)
            working_data = invert_to_density(working_data)
        elif type == "density_to_luminance":
            print("PROCESS: density to luminance",
                  file=sys.stderr)
            working_data = density_to_luminance(working_data)
        elif type == "add" or type == "addition":
            print(f"PROCESS: addition\n"
                  f"         {tuple(float(x) for x in adjustment["values"])}",
                  file=sys.stderr)
            working_data = apply_addition(working_data,
                                          adjustment["values"])
        elif type == "mult" or type == "gain":
            print(f"PROCESS: gain\n"
                  f"         {tuple(float(x) for x in adjustment["values"])}",
                  file=sys.stderr)
            working_data = apply_gain(working_data,
                                      adjustment["values"])
        elif type == "pow" or type == "power":
            print(f"PROCESS: power\n"
                  f"         {tuple(float(x) for x in adjustment["values"])}",
                  file=sys.stderr)
            working_data = apply_power(working_data,
                                       adjustment["values"])
        elif type == "norm":
            print("PROCESS: normalize", file=sys.stderr)
            normalize_image(working_data, adjustment["values"],
                            adjustment["inset"])
        elif type == "shift_blacks":
            shift_blacks(working_data, adjustment["inset"])
    return working_data
