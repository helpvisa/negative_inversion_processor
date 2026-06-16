import sys
import scipy
import numpy as np

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
    return inverse_spline(image_data)


def density_to_luminance(image_data):
    """
    Map an image from "density space" into its final luminance values.
    """
    lum_x_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    lum_y_points = np.power(10, lum_x_points) * 0.01
    lum_spline = scipy.interpolate.CubicSpline(lum_x_points,
                                               lum_y_points)
    return lum_spline(image_data)


def emulsion_wb(image_data, blurred_data, region, colourspace_weights,
                custom_black_point=None):
    """
    White balance a non-inverted camera negative to the "brightest point"
    visible within the bounding box of the provided region. This becomes the
    inverted film's new "black point".

    In most cases, this is none other than the unexposed film base itself.
    """
    # create pre-inversion analysis region
    analysis_region = blurred_data[region[0][1]:region[1][1],
                                   region[0][0]:region[1][0]]
    max_rgb_values = None
    if custom_black_point:
        print(f"CUSTOM: Custom black point: {custom_black_point}",
              file=sys.stderr)
        max_rgb_values = image_data[custom_black_point[1],
                                    custom_black_point[0]]
    else:
        # find max luminance value in region
        # TODO: maybe we normalize and find value closest to RGB(1.0, 1.0, 1.0)?
        #       we could do this for all min/max checks
        pre_lum_values = np.dot(analysis_region, colourspace_weights)
        # and find point of maximum luminance
        max_lum_y, max_lum_x = np.unravel_index(np.argmax(pre_lum_values),
                                                pre_lum_values.shape)
        # now grab the rgb values at that point
        max_rgb_values = analysis_region[max_lum_y, max_lum_x]
    print(f"INFO: White balance on emulsion (pre-inversion):\n"
          f"      RED:   {max_rgb_values[0]}\n"
          f"      GREEN: {max_rgb_values[1]}\n"
          f"      BLUE:  {max_rgb_values[2]}",
          file=sys.stderr)
    red_channel_wb = image_data[:, :, 0].copy()
    green_channel_wb = image_data[:, :, 1].copy()
    blue_channel_wb = image_data[:, :, 2].copy()
    # apply initial balance around channel with strongest cast
    wb_mult = np.max(max_rgb_values)
    rc_wb = wb_mult / max_rgb_values[0]
    gc_wb = wb_mult / max_rgb_values[1]
    bc_wb = wb_mult / max_rgb_values[2]
    red_channel_wb *= rc_wb
    green_channel_wb *= gc_wb
    blue_channel_wb *= bc_wb
    new_image_data = np.stack([red_channel_wb,
                               green_channel_wb,
                               blue_channel_wb],
                              axis=2)
    print(f"ADJUSTED: White balance on emulsion (pre-inversion):\n"
          f"          RED:   {rc_wb}\n"
          f"          GREEN: {gc_wb}\n"
          f"          BLUE:  {bc_wb}",
          file=sys.stderr)
    # return image data and tuple representing value adjustments in rgb
    return new_image_data, (rc_wb, gc_wb, bc_wb)


def density_wb(image_data, blurred_data, region, colourspace_weights,
               custom_gray_point=None):
    """
    Add or subtract from a determined middle-gray point to attempt equalization
    of all values clustered around middle gray, neutralizing the primary colour
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
        analysis_region = blurred_data[region[0][1]:region[1][1],
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
        low_mid_gray_rgb_values = analysis_region[low_mid_gray_y,
                                                  low_mid_gray_x]
        high_mid_gray_rgb_values = analysis_region[high_mid_gray_y,
                                                   high_mid_gray_x]
        mid_gray_rgb_values = high_mid_gray_rgb_values - low_mid_gray_rgb_values
    print(f"INFO: Middle gray RGB values (normalized):\n"
          f"      RED:   {mid_gray_rgb_values[0]}\n"
          f"      GREEN: {mid_gray_rgb_values[1]}\n"
          f"      BLUE:  {mid_gray_rgb_values[2]}",
          file=sys.stderr)
    # apply mults to channel densities to balance them
    mid_mult = max(mid_gray_rgb_values)
    red_channel_mid_gray = image_data[:, :, 0].copy()
    green_channel_mid_gray = image_data[:, :, 1].copy()
    blue_channel_mid_gray = image_data[:, :, 2].copy()
    rc_change = mid_gray_rgb_values[0] / mid_mult
    gc_change = mid_gray_rgb_values[1] / mid_mult
    bc_change = mid_gray_rgb_values[2] / mid_mult
    red_channel_mid_gray *= rc_change
    green_channel_mid_gray *= gc_change
    blue_channel_mid_gray *= bc_change
    new_image_data = np.stack([red_channel_mid_gray,
                              green_channel_mid_gray,
                              blue_channel_mid_gray],
                             axis=2)
    # print out total aggregate change
    print(f"ADJUSTED: Middle gray adjustment:\n"
          f"          RED:   {rc_change}\n"
          f"          GREEN: {gc_change}\n"
          f"          BLUE:  {bc_change}",
          file=sys.stderr)
    # return image data and tuple representing adjustments in rgb
    return new_image_data, (rc_change, gc_change, bc_change)


def density_balance_gain(image_data, blurred_data, region, colourspace_weights,
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
        wb_lum_rgb_values = image_data[custom_point[1],
                                       custom_point[0]]
    else:
        # update analysis region
        analysis_region = blurred_data[region[0][1]:region[1][1],
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
        wb_lum_rgb_values = analysis_region[wb_lum_y,
                                            wb_lum_x]
    print(f"INFO: Final {"black" if black_point else "white"} balance values:\n"
          f"      RED:   {wb_lum_rgb_values[0]}\n"
          f"      GREEN: {wb_lum_rgb_values[1]}\n"
          f"      BLUE:  {wb_lum_rgb_values[2]}",
          file=sys.stderr)
    # always apply final balance around green channel
    wb_mult = wb_lum_rgb_values[1]
    red_channel_wb_final = image_data[:, :, 0].copy()
    green_channel_wb_final = image_data[:, :, 1].copy()
    blue_channel_wb_final = image_data[:, :, 2].copy()
    rc_wb = wb_mult / wb_lum_rgb_values[0]
    gc_wb = wb_mult / wb_lum_rgb_values[1]
    bc_wb = wb_mult / wb_lum_rgb_values[2]
    red_channel_wb_final *= rc_wb
    green_channel_wb_final *= gc_wb
    blue_channel_wb_final *= bc_wb
    new_image_data = np.stack([red_channel_wb_final,
                               green_channel_wb_final,
                               blue_channel_wb_final],
                              axis=2)
    print(f"ADJUSTED: Final {"black" if black_point else "white"} balance:\n"
          f"          RED:   {rc_wb}\n"
          f"          GREEN: {gc_wb}\n"
          f"          BLUE:  {bc_wb}",
          file=sys.stderr)
    # return new_image_data and tuple containing adjustments in rgb
    return new_image_data, (rc_wb, gc_wb, bc_wb)


def density_custom_gain(image_data, custom_gain: tuple[float, float, float]):
    """
    Apply arbitrary multiplication across the three RGB colour channels.
    """
    red_channel = image_data[:, :, 0].copy() * custom_gain[0]
    green_channel = image_data[:, :, 1].copy() * custom_gain[1]
    blue_channel = image_data[:, :, 2].copy() * custom_gain[2]
    new_image_data = np.stack([red_channel,
                               green_channel,
                               blue_channel],
                              axis=2)
    # return only new image data; we must already have the adjustments
    return new_image_data
