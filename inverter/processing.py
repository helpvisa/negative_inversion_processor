import sys
import math
import rawpy
import tifffile
import scipy
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
                     photometric="rgb",
                     compression="zlib",
                     compressionargs={"level":9},
                     predictor=p_val,
                     iccprofile=icc_profile)
    print(f"Saved inversion to {output_path}", file=sys.stderr)


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
    inverse_x_points = np.array([0.001, 0.25, 0.5, 0.75, 1.0])
    inverse_y_points = np.log10(1.0 / inverse_x_points)
    inverse_spline = scipy.interpolate.CubicSpline(inverse_x_points,
                                                   inverse_y_points)
    # clip rgb values to avoid discolouration outside the negative itself,
    # which should theoretically still be within a 0 - 1 range at this point
    return inverse_spline(np.clip(image_data, a_min=0.0, a_max=1.0))


def density_to_luminance(image_data):
    """
    Map an image from "density space" into its final luminance values.
    """
    lum_x_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    lum_y_points = np.power(10, lum_x_points) * 0.01
    lum_spline = scipy.interpolate.CubicSpline(lum_x_points,
                                               lum_y_points)
    return lum_spline(image_data)


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


def white_balance(image_data, region, colourspace_weights,
                  custom_wb_point=None, mode='mult'):
    """
    White balance a "brightest point" visible within the bounding box of the
    provided region. This becomes the inverted film's new "black point".

    In most cases, this is none other than the unexposed film base itself.

    Return a dictionary reprsenting the type of adjustment and its values.
    """
    # create pre-inversion analysis region
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    max_rgb_values = None
    if custom_wb_point:
        print(f"CUSTOM: Custom balance point: {custom_wb_point}",
              file=sys.stderr)
        max_rgb_values = average_sample_point(image_data,
                                              custom_wb_point[0],
                                              custom_wb_point[1],
                                              16)
    else:
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
    print(f"ADJUSTMENT: Balancing whites ({mode} mode)\n"
          f"            RED:   {rc}\n"
          f"            GREEN: {gc}\n"
          f"            BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": mode,
        "values": (rc, gc, bc)
    }
    return adjustment


def density_balance(image_data, region, colourspace_weights,
                    exponent=1.0, ref_points=None):
    """
    Multiply the individual colour channels until equalized at a given point.
    Neutralizes colour casts in highlights and shadows.
    """
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    if ref_points:
        clear = image_data[ref_points[1], ref_points[0]]
        dense = image_data[ref_points[3], ref_points[2]]
        print(f"CUSTOM: Custom reference points: {ref_points}",
              file=sys.stderr)
    else:
        # estimate two clear/dense values in the image
        dense = find_brightest_luminance_spot(analysis_region,
                                              colourspace_weights)
        clear = find_darkest_luminance_spot(analysis_region,
                                            colourspace_weights)
    # determine density using green channel
    if (dense[1] < clear[1]):
        clear, dense = dense, clear
    density_ratio = clear[1] / dense[1]
    # these become our balancing exponents
    rc = (clear[0] / dense[0]) / density_ratio * exponent
    gc = exponent
    bc = (clear[2] / dense[2]) / density_ratio * exponent
    print(f"ADJUSTMENT: Density balance:\n"
          f"            RED:   {rc}\n"
          f"            GREEN: {gc}\n"
          f"            BLUE:  {bc}",
          file=sys.stderr)
    density_adjustment = {
        "type": "mult",
        "values": (rc, gc, bc)
    }
    # we also use the "dense" spot as a reference illumination
    # apply some channel multipliers to ensure reference illum doesn't change
    # rr = dense[0] - clear[0] * rc
    # gr = dense[1] - clear[1] * gc
    # br = dense[2] - clear[2] * bc
    # print(f"ADJUSTMENT: Reference illumination adjustment:\n"
    #       f"            RED:   {rr}\n"
    #       f"            GREEN: {gr}\n"
    #       f"            BLUE:  {br}",
    #       file=sys.stderr)
    # ref_adjustment = {
    #     "type": "add",
    #     "values": (rr, gr, br)
    # }
    return density_adjustment


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
            print(f"PROCESS: normalize\n"
                  f"         primaries: {tuple(float(x) for x in adjustment["values"])}\n"
                  f"         {adjustment["inset"]}",
                  file=sys.stderr)
            normalize_image(working_data, adjustment["values"],
                            adjustment["inset"])
    return working_data
