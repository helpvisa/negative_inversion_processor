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
    Invert a given film negative into a linear density space where it can
    be further corrected before having its densities mapped to luminance.
    """
    # we calculate density from "transmittance" using log10(1/x)
    # see https://abpy.github.io/2023/08/20/color-neg.html
    inverse_x_points = np.array([0.001, 0.25, 0.5, 0.75, 1.0])
    inverse_y_points = np.log10(1.0 / inverse_x_points)
    inverse_spline = scipy.interpolate.CubicSpline(inverse_x_points,
                                                   inverse_y_points)
    # normalize rgb values to avoid weird inversion errors
    image_max = np.max(image_data)
    normalized_image_data = image_data / image_max
    return inverse_spline(normalized_image_data)


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
                custom_black_point=None):
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
    if custom_black_point:
        print(f"CUSTOM: Custom black point: {custom_black_point}",
              file=sys.stderr)
        max_rgb_values = average_sample_point(image_data,
                                              custom_black_point[0],
                                              custom_black_point[1],
                                              16)
    else:
        # find max luminance value in region
        # TODO: maybe we normalize and find value closest to RGB(1.0, 1.0, 1.0)?
        #       we could do this for all min/max checks
        pre_lum_values = np.dot(analysis_region, colourspace_weights)
        # and find point of maximum luminance
        max_lum_y, max_lum_x = np.unravel_index(np.argmax(pre_lum_values),
                                                pre_lum_values.shape)
        # now grab the rgb values at that point
        max_rgb_values = average_sample_point(analysis_region,
                                              max_lum_x, max_lum_y,
                                              8)
    print(f"INFO: White balance on emulsion (pre-inversion):\n"
          f"      RED:   {max_rgb_values[0]}\n"
          f"      GREEN: {max_rgb_values[1]}\n"
          f"      BLUE:  {max_rgb_values[2]}",
          file=sys.stderr)
    # apply initial balance around channel with strongest cast
    mult = np.max(max_rgb_values)
    rc = mult / max_rgb_values[0]
    gc = mult / max_rgb_values[1]
    bc = mult / max_rgb_values[2]
    print(f"ADJUSTMENT: White balance on emulsion (pre-inversion):\n"
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
               custom_gray_point=None):
    """
    Multiply individual channels around a gray point average from a light
    and dark gray patch.

    Equalizes all values clustered around gray, neutralizing the primary colour
    cast of a given negative.
    """
    normalization_factor = 1.0
    mid_gray_rgb_values = None
    if custom_gray_point:
        print(f"CUSTOM: Custom gray point: {custom_gray_point}",
              file=sys.stderr)
        mid_gray_rgb_values = image_data[custom_gray_point[1],
                                         custom_gray_point[0]]
        # adjust this point to actual mid-gray
        mid_gray_luminance = mid_gray_rgb_values.dot(colourspace_weights)
        normalization_factor = mid_gray_luminance / 0.18
        mid_gray_rgb_values /= normalization_factor
    else:
        # update analysis region from current working image
        analysis_region = image_data[region[0][1]:region[1][1],
                                     region[0][0]:region[1][0]]
        # normalize image data
        post_lum_values = np.dot(analysis_region,
                                 colourspace_weights)
        max_post_lum_y, max_post_lum_x = np.unravel_index(np.argmax(post_lum_values),
                                                          post_lum_values.shape)
        max_post_lum_rgb_values = analysis_region[max_post_lum_y,
                                                  max_post_lum_x]
        normalization_factor = np.max(max_post_lum_rgb_values)
        # re-assign to avoid overwiting image_data, since
        # analysis_region references image_data directly
        analysis_region = analysis_region / normalization_factor
        low_gray = np.array([0.09, 0.09, 0.09], dtype=np.float32)
        high_gray = np.array([0.36, 0.36, 0.36], dtype=np.float32)
        # find the closest luminance point to 'middle gray'
        proximity_to_low_gray = np.linalg.norm(analysis_region - low_gray,
                                               axis=-1)
        proximity_to_high_gray = np.linalg.norm(analysis_region - high_gray,
                                                axis=-1)
        low_mid_gray_y, low_mid_gray_x = np.unravel_index(np.argmin(proximity_to_low_gray),
                                                          proximity_to_low_gray.shape)
        high_mid_gray_y, high_mid_gray_x = np.unravel_index(np.argmin(proximity_to_high_gray),
                                                          proximity_to_high_gray.shape)
        low_mid_gray_rgb_values = average_sample_point(analysis_region,
                                                       low_mid_gray_x,
                                                       low_mid_gray_y,
                                                       16)
        high_mid_gray_rgb_values = average_sample_point(analysis_region,
                                                        high_mid_gray_x,
                                                        high_mid_gray_y,
                                                        16)
        mid_gray_rgb_values = high_mid_gray_rgb_values - low_mid_gray_rgb_values
    print(f"INFO: Middle gray RGB values (normalized):\n"
          f"      RED:   {mid_gray_rgb_values[0]}\n"
          f"      GREEN: {mid_gray_rgb_values[1]}\n"
          f"      BLUE:  {mid_gray_rgb_values[2]}",
          file=sys.stderr)
    # apply mults to channel densities to balance them
    mid_mult = max(mid_gray_rgb_values)
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
    mult = max(gray_mean)
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
                         custom_point=None, black_point=False):
    """
    Multiply the individual colour channels until equalized at a given point.
    Neutralizes colour casts in highlights and shadows.

    `custom_point` can be either a white or black point in density space.

    The optional `black_point` parameter is only used when `custom_point` is
    not provided, in the case that a black point should be determined
    automatically.
    """
    wb_lum_rgb_values = None
    if custom_point:
        print(f"CUSTOM: Custom white point: {custom_point}",
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
        if black_point:
            wb_lum_y, wb_lum_x = np.unravel_index(np.argmin(wb_lum_values),
                                                  wb_lum_values.shape)
        else:
            wb_lum_y, wb_lum_x = np.unravel_index(np.argmax(wb_lum_values),
                                                  wb_lum_values.shape)
        wb_lum_rgb_values = average_sample_point(analysis_region,
                                                 wb_lum_x,
                                                 wb_lum_y,
                                                 16)
    print(f"INFO: Final {"black" if black_point else "white"} balance values:\n"
          f"      RED:   {wb_lum_rgb_values[0]}\n"
          f"      GREEN: {wb_lum_rgb_values[1]}\n"
          f"      BLUE:  {wb_lum_rgb_values[2]}",
          file=sys.stderr)
    # always apply final balance around green channel
    wb_mult = wb_lum_rgb_values[1]
    rc = wb_mult / (wb_lum_rgb_values[0] if wb_lum_rgb_values[0] != 0.0 else 1.0)
    gc = wb_mult / (wb_lum_rgb_values[1] if wb_lum_rgb_values[1] != 0.0 else 1.0)
    bc = wb_mult / (wb_lum_rgb_values[2] if wb_lum_rgb_values[2] != 0.0 else 1.0)
    print(f"ADJUSTMENT: Final {"black" if black_point else "white"} balance:\n"
          f"          RED:   {rc}\n"
          f"          GREEN: {gc}\n"
          f"          BLUE:  {bc}",
          file=sys.stderr)
    adjustment = {
        "type": "mult",
        "values": (rc, gc, bc)
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
        if type == "add":
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
    return working_data
