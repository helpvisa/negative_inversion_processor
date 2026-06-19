import sys
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
    inverse_x_points = np.array([0.001, 0.25, 0.5, 0.75, 1.0,
                                 1.25, 1.5, 1.75, 2.0])
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
        gc = 1.0
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


def density_balance(image_data, region, colourspace_weights, exponent=1.0):
    """
    Multiply the individual colour channels until equalized at a given point.
    Neutralizes colour casts in highlights and shadows.
    """
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    brightest_spot = find_brightest_luminance_spot(analysis_region,
                                                   colourspace_weights)
    # estimate two median values in the image
    r_med_1 = np.median(image_data[:, :, 0])
    g_med_1 = np.median(image_data[:, :, 1])
    b_med_1 = np.median(image_data[:, :, 2])
    r_med_2 = brightest_spot[0] / 24.0
    g_med_2 = brightest_spot[1] / 24.0
    b_med_2 = brightest_spot[2] / 24.0
    # determine their density using green channel
    clear = [r_med_1, g_med_1, b_med_1]
    dense = [r_med_2, g_med_2, b_med_2]
    if (dense[1] < clear[1]):
        clear, dense = dense, clear
    density_ratio = dense[1] / clear[1]
    # these become our balancing exponents
    rc = (clear[0] / dense[0]) * density_ratio * exponent
    gc = (clear[1] / dense[1]) * density_ratio * exponent
    bc = (clear[2] / dense[2]) * density_ratio * exponent
    print(f"ADJUSTMENT: Density balance:\n"
          f"            RED:   {rc}\n"
          f"            GREEN: {gc}\n"
          f"            BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": "mult",
        "values": (rc, gc, bc)
    }
    return adjustment


def normalize_image(image_data, region, colourspace_weights):
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    lum_values = find_brightest_luminance_spot(analysis_region,
                                               colourspace_weights)
    factor = np.max(lum_values)
    adjustment = {
        "type": "mult",
        "values": (1/factor, 1/factor, 1/factor)
    }
    return adjustment


def shift_blacks_to_zero(image_data, region, colourspace_weights):
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    lum_values = np.dot(analysis_region, colourspace_weights)
    lum_x, lum_y = np.unravel_index(np.argmin(lum_values),
                                    lum_values.shape)
    lum_values = analysis_region[lum_x, lum_y]
    adjustment = {
        "type": "add",
        "values": (-lum_values[0], -lum_values[1], -lum_values[2])
    }
    return adjustment


def process_all_adjustments(image_data, adjustments):
    print("Applying all adjustments to create final image...",
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
    return working_data
