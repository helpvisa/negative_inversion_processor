import sys
import argparse
import numpy as np
import rawpy
import tifffile
import colour
import scipy


def main():
    # parse user-supplied arguments
    parser = argparse.ArgumentParser(prog="Neg Inverter",
                                     description="Automatically invert your film negatives")
    # input / output
    parser.add_argument('image_path',
                        type=str, default=None,
                        help="Input RAW file to process.")
    parser.add_argument('output_path', type=str, default=None,
                        help="Output path for final processed image.")
    # lut applications
    parser.add_argument('--input-lut', '-l',
                        type=str, default=None,
                        help="1D LUT applied into working colour space.")
    parser.add_argument('--output-lut', '-L',
                        type=str, default=None,
                        help="1D LUT applied out of working colour space.")
    # icc profile embedding
    parser.add_argument('--icc', '-p',
                        type=str, default=None,
                        help="ICC profile to embed in final TIFF export.")
    # image modifiers
    parser.add_argument('--analysis-width', '-W',
                        type=int, default=5000,
                        help="Size of analysis bounding box (width).")
    parser.add_argument('--analysis-height', '-H',
                        type=int, default=3333,
                        help="Size of analysis bounding box (height).")
    parser.add_argument('--bp-margin',
                        type=float, default=0.0,
                        help="Black point correction safety margin (lift)")
    parser.add_argument('--exposure-comp',
                        type=float, default=0.7,
                        help="Apply post-inversion exposure compensation.")
    parser.add_argument('--skip-inversion',
                        action='store_true',
                        help="Do not apply any inversion.")
    parser.add_argument('--skip-auto-adjustments',
                        action='store_true',
                        help="Do not apply any auto-adjustments.")
    parser.add_argument('--debug-analysis-region',
                        action='store_true',
                        help="Show bounds of analysis region.")
    parser.add_argument('--wb-steps', '-S',
                        type=int, default=6,
                        help="Number of times to re-balance around mid-gray.")
    parser.add_argument('--red-balance', '-R',
                        type=float, default=1.0,
                        help="Custom red balance (in density space).")
    parser.add_argument('--green-balance', '-G',
                        type=float, default=1.0,
                        help="Custom green balance (in density space).")
    parser.add_argument('--blue-balance', '-B',
                        type=float, default=1.0,
                        help="Custom blue balance (in density space).")
    args = parser.parse_args()


    # read in raw file
    with rawpy.imread(args.image_path) as raw:
        # read in camera sensor data at 16bpp int with D65 white balance
        d65_balance = raw.daylight_whitebalance
        image_data_uint16 = raw.postprocess(use_camera_wb=False,
                                            user_wb=d65_balance,
                                            no_auto_bright=True,
                                            half_size=False,
                                            output_bps=16,
                                            gamma=(1.0, 1.0),
                                            output_color=rawpy.ColorSpace.Rec2020)

        # convert data to 32-bit floating point
        image_data_f32 = image_data_uint16.astype(np.float32) / 65535.0

        # apply input lut (if present)
        working_image = image_data_f32
        if args.input_lut:
            input_lut = colour.read_LUT(args.input_lut)
            working_image = input_lut.apply(working_image)

        # perform adjustments in working colour space
        rec2020_lum_weights = np.array([0.2627, 0.6780, 0.05903])
        # determine and acqiure analysis region
        image_height, image_width, image_channels = working_image.shape
        analysis_start_x = (image_width - args.analysis_width) // 2
        analysis_end_x = analysis_start_x + args.analysis_width
        analysis_start_y = (image_height - args.analysis_height) // 2
        analysis_end_y = analysis_start_y + args.analysis_height

        # create pre-inversion analysis regoin
        analysis_region_pre = working_image[analysis_start_y:analysis_end_y,
                                            analysis_start_x:analysis_end_x]
        # perform pre-inversion white balance
        if not args.skip_auto_adjustments:
            # find max luminance value in region
            pre_lum_values = np.dot(analysis_region_pre, rec2020_lum_weights)
            # and find point of maximum luminance
            max_lum_y, max_lum_x = np.unravel_index(np.argmax(pre_lum_values),
                                                    pre_lum_values.shape)
            # now grab the rgb values at that point
            max_rgb_values = analysis_region_pre[max_lum_y, max_lum_x]
            print(f"INFO: Darkest values (pre-inversion):\n"
                  f"      RED:   {max_rgb_values[0]}\n"
                  f"      GREEN: {max_rgb_values[1]}\n"
                  f"      BLUE:  {max_rgb_values[2]}",
                  file=sys.stderr)
            red_channel_wb = working_image[:, :, 0].copy()
            green_channel_wb = working_image[:, :, 1].copy()
            blue_channel_wb = working_image[:, :, 2].copy()
            wb_mult = np.max(max_rgb_values)
            red_channel_wb *= wb_mult / max_rgb_values[0]
            green_channel_wb *= wb_mult / max_rgb_values[1]
            blue_channel_wb *= wb_mult / max_rgb_values[2]
            working_image = np.stack([red_channel_wb,
                                      green_channel_wb,
                                      blue_channel_wb],
                                     axis=2)

        # invert image directly
        if not args.skip_inversion:
            # we calculate density from "transmittance" using log10(1/x)
            # TODO: normalize before applying 1D output lut
            print("WARN: Inverting the image is based on density, and may\n"
                  "      create values above 1.0 which break output LUTs.",
                  file=sys.stderr)
            inverse_x_points = np.array([0.001, 0.25, 0.5, 0.75, 1.0])
            inverse_y_points = np.log10(1.0 / inverse_x_points)
            inverse_spline = scipy.interpolate.CubicSpline(inverse_x_points,
                                                           inverse_y_points)
            working_image = inverse_spline(working_image)

            # automatically determine min/max point for each channel
            if not args.skip_auto_adjustments:
                # re-find black point
                # update analysis region
                analysis_region_post = working_image[analysis_start_y:analysis_end_y,
                                                     analysis_start_x:analysis_end_x]
                # find darkest luminance point
                min_lum_values = np.dot(analysis_region_post, rec2020_lum_weights)
                min_lum_y, min_lum_x = np.unravel_index(np.argmin(min_lum_values),
                                                        min_lum_values.shape)
                # now grab the rgb values at that point
                min_rgb_values = analysis_region_post[min_lum_y, min_lum_x]
                print(f"INFO: Darkest values (post-inversion):\n"
                      f"      RED:   {min_rgb_values[0]}\n"
                      f"      GREEN: {min_rgb_values[1]}\n"
                      f"      BLUE:  {min_rgb_values[2]}",
                      file=sys.stderr)
                # add auto-adjustments (offset + density adjustment)
                red_channel_auto = working_image[:, :, 0].copy()
                green_channel_auto = working_image[:, :, 1].copy()
                blue_channel_auto = working_image[:, :, 2].copy()
                red_channel_auto -= min_rgb_values[0] - args.bp_margin
                green_channel_auto -= min_rgb_values[1] - args.bp_margin
                blue_channel_auto -= min_rgb_values[2] - args.bp_margin
                working_image = np.stack([red_channel_auto,
                                          green_channel_auto,
                                          blue_channel_auto],
                                         axis=2)
                # find spot closest to middle gray for density adjustment
                for i in range(args.wb_steps):
                    # update analysis region from current working image
                    analysis_region_post = working_image[analysis_start_y:analysis_end_y,
                                                         analysis_start_x:analysis_end_x]
                    # normalize image data
                    post_lum_values = np.dot(analysis_region_post,
                                             rec2020_lum_weights)
                    max_post_lum_y, max_post_lum_x = np.unravel_index(np.argmax(post_lum_values),
                                                                      post_lum_values.shape)
                    max_post_lum_rgb_values = analysis_region_post[max_post_lum_y,
                                                                   max_post_lum_x]
                    normalization_factor = np.max(max_post_lum_rgb_values)
                    # re-assign to avoid overwiting working_image
                    # (analysis_region_post references working_image directly)
                    analysis_region_post = analysis_region_post / normalization_factor
                    post_lum_values_norm = np.dot(analysis_region_post,
                                                  rec2020_lum_weights)
                    # find the closest luminance point to 'middle gray'
                    mid_gray_idx = np.argmin(np.abs(post_lum_values_norm - 0.18))
                    mid_gray_y, mid_gray_x = np.unravel_index(mid_gray_idx,
                                                              post_lum_values_norm.shape)
                    mid_gray_rgb_values = analysis_region_post[mid_gray_y,
                                                               mid_gray_x]
                    print(f"INFO: Middle gray RGB values (normalized):\n"
                          f"      RED:   {mid_gray_rgb_values[0]}\n"
                          f"      GREEN: {mid_gray_rgb_values[1]}\n"
                          f"      BLUE:  {mid_gray_rgb_values[2]}",
                          file=sys.stderr)
                    # apply adds and mults to channel densities to balance them
                    mid_gray_mult = np.max(mid_gray_rgb_values)
                    red_channel_mid_gray = working_image[:, :, 0].copy()
                    green_channel_mid_gray = working_image[:, :, 1].copy()
                    blue_channel_mid_gray = working_image[:, :, 2].copy()
                    red_channel_mid_gray += (mid_gray_mult - mid_gray_rgb_values[0]) * normalization_factor
                    green_channel_mid_gray += (mid_gray_mult - mid_gray_rgb_values[1]) * normalization_factor
                    blue_channel_mid_gray += (mid_gray_mult - mid_gray_rgb_values[2]) * normalization_factor
                    working_image = np.stack([red_channel_mid_gray,
                                              green_channel_mid_gray,
                                              blue_channel_mid_gray],
                                             axis=2)

        # apply a final, user-adjustable gain to balance in density space
        red_channel_ub = working_image[:, :, 0].copy() * args.red_balance
        green_channel_ub = working_image[:, :, 1].copy() * args.green_balance
        blue_channel_ub = working_image[:, :, 2].copy() * args.blue_balance
        working_image = np.stack([red_channel_ub,
                                  green_channel_ub,
                                  blue_channel_ub],
                                 axis=2)

        # map density to luminance
        if not args.skip_inversion:
            lum_x_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            lum_y_points = np.power(10, lum_x_points) * 0.01
            lum_spline = scipy.interpolate.CubicSpline(lum_x_points,
                                                       lum_y_points)
            working_image = lum_spline(working_image)

        # apply output lut (if present)
        final_image = working_image
        if args.output_lut:
            output_lut = colour.read_LUT(args.output_lut)
            final_image = output_lut.apply(working_image)

        # add output gain and black point adjustment
        if not args.skip_auto_adjustments:
            black_pt_idx = np.unravel_index(np.sum(final_image, axis=-1).argmin(),
                                            final_image.shape[:2])
            darkest_color = final_image[black_pt_idx]
            final_image -= darkest_color
        final_image *= args.exposure_comp

        if args.debug_analysis_region:
            final_image[analysis_start_y:analysis_start_y + 64,
                        analysis_start_x:analysis_start_x + 64] = [1.0, 0.0, 0.0]
            final_image[analysis_end_y:analysis_end_y + 64,
                        analysis_end_x:analysis_end_x + 64] = [1.0, 0.0, 0.0]

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


if __name__ == "__main__":
    main()
