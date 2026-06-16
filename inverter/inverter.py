import sys
import math
from types import SimpleNamespace
import numpy as np
import rawpy
import tifffile
from scipy import ndimage
import parse_cli_arguments
from processing import (invert_to_density, density_to_luminance,
                        emulsion_wb, density_wb,
                        density_balance_gain, density_custom_gain)


def load_raw_image(path):
    print(f"Processing image: {path}", file=sys.stderr)
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


def process_negative(args):
    """
    Process a negative, derived from a camera RAW file.

    `args` represents a complete collection of configuration options, derived
    primarily from the options exposed in the command-line interface.
    So long as a dictionary of the appropriate values are provided, the args
    do not need to come from the CLI itself.

    A full list of applicable options is as follows:
        - `image_path`
        - `preset`
        - `generate_preset`
        - `analysis_width`
        - `analysis_height`
        - `exposure_comp`
        - `skip_inversion`
        - `skip_auto_adjustments`
        - `debug_analysis_region`
        - `red_balance`
        - `green_balance`
        - `blue_balance`
        - `custom_white_point`
        - `custom_gray_point`
        - `custom_black_point`

    See `parse_cli_arguments.py` for more detailed information about each of
    the individual arguments available in the `args` dictionary.
    """
    # convert args from a dictionary into a namespace
    if type(args) is dict:
        args = SimpleNamespace(**args)

    # convert camera raw to 32-bit floating point
    working_image = load_raw_image(args.image_path).astype(np.float32) / 65535.0
    print("Creating blurred analysis image...", file=sys.stderr)
    # will have to updated the color balance of the blurred image as well
    # color balance off this image, then re-apply the transformations to the base?
    blurred_image = ndimage.gaussian_filter(working_image, sigma=(32, 32, 0))

    if args.resize:
        x_factor = args.resize[0] / working_image.shape[1]
        y_factor = args.resize[1] / working_image.shape[0]
        args.analysis_width = int(math.floor(args.analysis_width * x_factor))
        args.analysis_height = int(math.floor(args.analysis_height * y_factor))
        if args.custom_white_point:
            args.custom_white_point[0] = int(math.floor(args.custom_white_point[0] * x_factor))
            args.custom_white_point[1] = int(math.floor(args.custom_white_point[1] * y_factor))
        if args.custom_gray_point:
            args.custom_gray_point[0] = int(math.floor(args.custom_gray_point[0] * x_factor))
            args.custom_gray_point[1] = int(math.floor(args.custom_gray_point[1] * y_factor))
        if args.custom_black_point:
            args.custom_black_point[0] = int(math.floor(args.custom_black_point[0] * x_factor))
            args.custom_black_point[1] = int(math.floor(args.custom_black_point[1] * y_factor))
        resize_factor = (x_factor, y_factor, 1)
        working_image = ndimage.zoom(working_image,
                                     resize_factor,
                                     order=3)

    # perform adjustments in working colour space
    rec2020_lum_weights = np.array([0.2627, 0.6780, 0.05903])
    # determine and build analysis region
    image_height, image_width, image_channels = working_image.shape
    analysis_start_x = (image_width - args.analysis_width) // 2
    analysis_end_x = analysis_start_x + args.analysis_width
    analysis_start_y = (image_height - args.analysis_height) // 2
    analysis_end_y = analysis_start_y + args.analysis_height
    # store as array of [x,y] arrays
    analysis_bounding_box = [[analysis_start_x, analysis_start_y],
                             [analysis_end_x, analysis_end_y]]

    # perform pre-inversion white balance
    if not args.skip_auto_adjustments:
        custom_black_point = args.custom_black_point if args.custom_black_point else None
        working_image, ewb_adjustments = emulsion_wb(working_image,
                                                     blurred_image,
                                                     analysis_bounding_box,
                                                     rec2020_lum_weights,
                                                     custom_black_point)

    # invert image to begin work in "density space"
    if not args.skip_inversion:
        working_image = invert_to_density(working_image)

        if not args.skip_auto_adjustments:
            # find spot closest to middle gray for density adjustment
            custom_gray_point = args.custom_gray_point if args.custom_gray_point else None
            working_image, dwb_adjustments = density_wb(working_image,
                                                        blurred_image,
                                                        analysis_bounding_box,
                                                        rec2020_lum_weights,
                                                        custom_gray_point)
            # apply an automated white balance based on white point
            custom_white_point = args.custom_white_point if args.custom_white_point else None
            working_image, dbgw_adjustments = density_balance_gain(working_image,
                                                                   blurred_image,
                                                                   analysis_bounding_box,
                                                                   rec2020_lum_weights,
                                                                   custom_white_point)

        # apply a final, user-adjustable gain to balance in density space
        working_image = density_custom_gain(working_image, (args.red_balance,
                                                            args.green_balance,
                                                            args.blue_balance))
        if args.red_balance != 1.0 or args.green_balance != 1.0 or args.blue_balance != 1.0:
            print(f"ADJUSTED: User white balance:\n"
                  f"          RED:   {args.red_balance}\n"
                  f"          GREEN: {args.green_balance}\n"
                  f"          BLUE:  {args.blue_balance}",
                  file=sys.stderr)

        # map density to luminance
        working_image = density_to_luminance(working_image)

    # shift blacks back to zero
    final_analysis_region = blurred_image[analysis_bounding_box[0][1]:analysis_bounding_box[1][1],
                                          analysis_bounding_box[0][0]:analysis_bounding_box[1][0]]
    final_lum_values = np.dot(final_analysis_region, rec2020_lum_weights)
    final_lum_x, final_lum_y = np.unravel_index(np.argmin(final_lum_values),
                                                final_lum_values.shape)
    final_lum_rgb_values = final_analysis_region[final_lum_x, final_lum_y]
    final_red_offset = working_image[:, :, 0].copy() - final_lum_rgb_values[0]
    final_green_offset = working_image[:, :, 1].copy() - final_lum_rgb_values[1]
    final_blue_offset = working_image[:, :, 2].copy() - final_lum_rgb_values[2]
    working_image = np.stack([final_red_offset,
                              final_green_offset,
                              final_blue_offset],
                             axis=2)
    # apply output exposure compensation
    final_image = working_image * args.exposure_comp

    # display the analysis region and points if debug toggle enabled
    if args.debug_analysis_region:
        final_image[analysis_start_y-32:analysis_start_y+32,
                    analysis_start_x-32:analysis_start_x+32] = [1.0, 0.0, 0.0]
        final_image[analysis_end_y-32:analysis_end_y+32,
                    analysis_end_x-32:analysis_end_x+32] = [1.0, 0.0, 0.0]
        if args.custom_black_point:
            final_image[args.custom_black_point[1]-32:args.custom_black_point[1]+32,
                        args.custom_black_point[0]-32:args.custom_black_point[0]+32] = [0.0, 1.0, 0.0]
        if args.custom_white_point:
            final_image[args.custom_white_point[1]-32:args.custom_white_point[1]+32,
                        args.custom_white_point[0]-32:args.custom_white_point[0]+32] = [0.0, 0.0, 1.0]
        if args.custom_gray_point:
            final_image[args.custom_gray_point[1]-32:args.custom_gray_point[1]+32,
                        args.custom_gray_point[0]-32:args.custom_gray_point[0]+32] = [0.0, 1.0, 1.0]
    # finally, we return image data
    # TODO: also return tuples of adjustments for saving presets
    return final_image


def main():
    args = parse_cli_arguments.parse_user_arguments()
    if not args.image_path or not args.output_path:
        print("ERROR: Please specify either:\n"
              "         1. a raw file and output path\n",
              "        2. a raw file, preset, and output path\n",
              "        3. a raw file and preset output path (if generating a preset)\n"
              "Run me with '-h' to see the full suite of options.",
              file=sys.stderr)
        exit(1)

    # process negatives
    final_image = process_negative(args)
    # save image to disk
    # do we possess an icc profile to embed?
    icc_tag = None
    if args.icc:
        with open(args.icc, "rb") as icc_file:
            icc_profile_bytes = icc_file.read()
            icc_tag = (34675, 7,
                       len(icc_profile_bytes), icc_profile_bytes,
                       True)
    else:
        print("WARN: Image is in linear Rec2020 colour space,\n"
              "      but you have not provided an ICC profile to embed.\n"
              "      Most image viewers will NOT display it correctly\n"
              "      without an embedded gamma 1.0 Rec2020 ICC profile.",
              file=sys.stderr)
    tifffile.imwrite(args.output_path,
                     final_image.astype(np.float16),
                     photometric="rgb",
                     compression="zlib",
                     compressionargs={"level":9},
                     predictor=3,
                     extratags=[icc_tag] if icc_tag else [])
    print(f"Done processing {args.image_path}!", file=sys.stderr)


if __name__ == "__main__":
    main()
