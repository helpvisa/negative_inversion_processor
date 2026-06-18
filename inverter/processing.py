import sys
import scipy
import numpy as np


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


def emulsion_wb(image_data, region, colourspace_weights,
                custom_wb_point=None, exponent=1.0):
    """
    White balance a non-inverted camera negative to the "brightest point"
    visible within the bounding box of the provided region. This becomes the
    inverted film's new "black point".

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
        # find max luminance value in region
        pre_lum_values = np.dot(analysis_region, colourspace_weights)
        # and find point of maximum luminance
        max_lum_y, max_lum_x = np.unravel_index(np.argmax(pre_lum_values),
                                                pre_lum_values.shape)
        # now grab the rgb values at that point
        max_rgb_values = average_sample_point(analysis_region,
                                              max_lum_x, max_lum_y,
                                              8)
    print(f"INFO: Balance (pre-inversion):\n"
          f"      RED:   {max_rgb_values[0]}\n"
          f"      GREEN: {max_rgb_values[1]}\n"
          f"      BLUE:  {max_rgb_values[2]}",
          file=sys.stderr)
    # always apply initial balance around green channel
    mult = max_rgb_values[1]
    rc = (mult / max_rgb_values[0]) * exponent
    gc = exponent
    bc = (mult / max_rgb_values[2]) * exponent
    print(f"ADJUSTMENT: Balance changes (pre-inversion):\n"
          f"          RED:   {rc}\n"
          f"          GREEN: {gc}\n"
          f"          BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": "mult",
        "values": (rc, gc, bc)
    }
    return adjustment


def density_wb(image_data, region, colourspace_weights,
               custom_gray_points=None):
    """
    Multiply individual channels around a gray point average from a light
    and dark gray patch.

    Equalizes all values clustered around gray, neutralizing the primary colour
    cast of a given negative.
    """
    normalization_factor = 1.0
    rgb_values_1 = None
    rgb_values_2 = None
    if custom_gray_points:
        print(f"CUSTOM: Custom gray points: {custom_gray_points}",
              file=sys.stderr)
        rgb_values_1 = average_sample_point(image_data,
                                            custom_gray_points[0],
                                            custom_gray_points[1],
                                            16)
        rgb_values_2 = average_sample_point(image_data,
                                            custom_gray_points[2],
                                            custom_gray_points[3],
                                            16)
    else:
        # update analysis region from current working image
        analysis_region = image_data[region[0][1]:region[1][1],
                                     region[0][0]:region[1][0]]
        # normalize luminance data
        post_lum_values = np.dot(analysis_region,
                                 colourspace_weights)
        max_post_lum_y, max_post_lum_x = np.unravel_index(np.argmax(post_lum_values),
                                                          post_lum_values.shape)
        max_post_lum_rgb_values = analysis_region[max_post_lum_y,
                                                  max_post_lum_x]
        normalization_factor = np.max(max_post_lum_rgb_values)
        post_lum_values = post_lum_values / normalization_factor
        analysis_region = analysis_region / normalization_factor
        low_gray = 0.9
        high_gray = 0.36
        # find the closest luminance point to 'middle gray'
        proximity_to_low_gray = np.abs(post_lum_values - low_gray)
        proximity_to_high_gray = np.abs(post_lum_values - high_gray)
        low_mid_gray_y, low_mid_gray_x = np.unravel_index(np.argmin(proximity_to_low_gray),
                                                          proximity_to_low_gray.shape)
        high_mid_gray_y, high_mid_gray_x = np.unravel_index(np.argmin(proximity_to_high_gray),
                                                          proximity_to_high_gray.shape)
        rgb_values_1 = average_sample_point(analysis_region,
                                            low_mid_gray_x,
                                            low_mid_gray_y,
                                            16)
        rgb_values_2 = average_sample_point(analysis_region,
                                            high_mid_gray_x,
                                            high_mid_gray_y,
                                            16)
    low_gray_values = rgb_values_1
    high_gray_values = rgb_values_2
    if np.sum(low_gray_values) > np.sum(high_gray_values):
        low_gray_values = rgb_values_2
        high_gray_values = rgb_values_1
    mid_gray_rgb_values = high_gray_values - low_gray_values
    print(f"INFO: Middle gray RGB values (normalized):\n"
          f"      RED:   {mid_gray_rgb_values[0]}\n"
          f"      GREEN: {mid_gray_rgb_values[1]}\n"
          f"      BLUE:  {mid_gray_rgb_values[2]}",
          file=sys.stderr)
    # always apply mults around green channel
    mid_mult = mid_gray_rgb_values[1]
    if mid_mult == 0.0:
        mid_mult = 1.0
    rc = mid_gray_rgb_values[0] / mid_mult
    gc = mid_gray_rgb_values[1] / mid_mult
    bc = mid_gray_rgb_values[2] / mid_mult
    # print out change
    print(f"ADJUSTMENT: Middle gray adjustment:\n"
          f"          RED:   {rc}\n"
          f"          GREEN: {gc}\n"
          f"          BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": "mult",
        "values": (rc, gc, bc)
    }
    return adjustment


def density_grayworld(image_data, region):
    """
    Average all colour values inside the region of the provided image. The
    remaining RGB value should represent middle gray, and can be used to
    correct colour casts.
    """
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    gray_mean = np.mean(analysis_region, axis=(0, 1))
    print(f"INFO: Gray world RGB values:\n"
          f"      RED:   {gray_mean[0]}\n"
          f"      GREEN: {gray_mean[1]}\n"
          f"      BLUE:  {gray_mean[2]}",
          file=sys.stderr)
    # always balance around green channel
    mult = gray_mean[1]
    if mult == 0.0:
        mult = 1.0
    rc = gray_mean[0] / mult
    gc = gray_mean[1] / mult
    bc = gray_mean[2] / mult
    # print out change
    print(f"ADJUSTMENT: Gray world adjustment:\n"
          f"          RED:   {rc}\n"
          f"          GREEN: {gc}\n"
          f"          BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": "mult",
        "values": (rc, gc, bc)
    }
    return adjustment


def density_balance_gain(image_data, region, colourspace_weights,
                         custom_point=None):
    """
    Multiply the individual colour channels until equalized at a given point.
    Neutralizes colour casts in highlights and shadows.

    `custom_point` is a point in density space.
    """
    wb_lum_rgb_values = None
    if custom_point:
        print(f"CUSTOM: Custom point: {custom_point}",
              file=sys.stderr)
        wb_lum_rgb_values = average_sample_point(image_data,
                                                 custom_point[0],
                                                 custom_point[1],
                                                 16)
    else:
        # update analysis region
        analysis_region = image_data[region[0][1]:region[1][1],
                                     region[0][0]:region[1][0]]
        # find brightest luminance point and balance around it
        wb_lum_values = np.dot(analysis_region, colourspace_weights)
        wb_lum_y = 0
        wb_lum_x = 0
        wb_lum_y, wb_lum_x = np.unravel_index(np.argmax(wb_lum_values),
                                                wb_lum_values.shape)
        wb_lum_rgb_values = average_sample_point(analysis_region,
                                                 wb_lum_x,
                                                 wb_lum_y,
                                                 16)
    print(f"INFO: Final peak values:\n"
          f"      RED:   {wb_lum_rgb_values[0]}\n"
          f"      GREEN: {wb_lum_rgb_values[1]}\n"
          f"      BLUE:  {wb_lum_rgb_values[2]}",
          file=sys.stderr)
    # always apply final balance around green channel
    wb_mult = wb_lum_rgb_values[1]
    rc = (wb_mult / wb_lum_rgb_values[0])
    gc = 1.0
    bc = (wb_mult / wb_lum_rgb_values[2])
    print(f"ADJUSTMENT: Final balance:\n"
          f"          RED:   {rc}\n"
          f"          GREEN: {gc}\n"
          f"          BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": "mult",
        "values": (rc, gc, bc)
    }
    return adjustment


def normalize_image(image_data, region, colourspace_weights):
    analysis_region = image_data[region[0][1]:region[1][1],
                                 region[0][0]:region[1][0]]
    lum_values = np.dot(analysis_region, colourspace_weights)
    lum_x, lum_y = np.unravel_index(np.argmax(lum_values),
                                    lum_values.shape)
    lum_values = analysis_region[lum_x, lum_y]
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
    print("PROCESS: applying all adjustments to create final image...",
          file=sys.stderr)
    working_data = image_data.copy()
    for adjustment in adjustments:
        type = adjustment["type"]
        if type == "invert_to_density":
            print("PROCESS: invert to density",
                  file=sys.stderr)
            working_data = invert_to_density(working_data)
        if type == "density_to_luminance":
            print("PROCESS: density to luminance",
                  file=sys.stderr)
            working_data = density_to_luminance(working_data)
        if type == "add" or type == "addition":
            print(f"PROCESS: addition\n"
                  f"         {tuple(float(x) for x in adjustment["values"])}",
                  file=sys.stderr)
            working_data = apply_addition(working_data,
                                          adjustment["values"])
        if type == "mult" or type == "gain":
            print(f"PROCESS: gain\n"
                  f"         {tuple(float(x) for x in adjustment["values"])}",
                  file=sys.stderr)
            working_data = apply_gain(working_data,
                                      adjustment["values"])
        if type == "pow" or type == "power":
            print(f"PROCESS: power\n"
                  f"         {tuple(float(x) for x in adjustment["values"])}",
                  file=sys.stderr)
            working_data = apply_power(working_data,
                                       adjustment["values"])
    return working_data
