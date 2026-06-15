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
    parser.add_argument('--red-balance', '-r',
                        type=float, default=1.0,
                        help="Multiplier for red channel (pre-inversion).")
    parser.add_argument('--green-balance', '-g',
                        type=float, default=1.0,
                        help="Multiplier for blue channel (pre-inversion).")
    parser.add_argument('--blue-balance', '-b',
                        type=float, default=1.0,
                        help="Multiplier for green channel (pre-inversion).")
    parser.add_argument('--analysis-width', '-W',
                        type=int, default=4000,
                        help="Size of analysis bounding box (width).")
    parser.add_argument('--analysis-height', '-H',
                        type=int, default=2666,
                        help="Size of analysis bounding box (height).")
    parser.add_argument('--bp-margin',
                        type=float, default=0.01,
                        help="Black point correction safety margin (lift)")
    parser.add_argument('--exposure-comp',
                        type=float, default=1.0,
                        help="Apply post-inversion exposure compensation.")
    parser.add_argument('--skip-inversion',
                        action='store_true',
                        help="Do not apply any inversion.")
    parser.add_argument('--skip-auto-adjustments',
                        action='store_true',
                        help="Do not apply any auto-adjustments.")
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
        # determine and acqiure analysis region
        if not args.skip_auto_adjustments:
            image_height, image_width, image_channels = working_image.shape
            analysis_start_x = (image_width - args.analysis_width) // 2
            analysis_end_x = analysis_start_x + image_width
            analysis_start_y = (image_height - args.analysis_height) // 2
            analysis_end_y = analysis_start_y + image_height
            analysis_region = working_image[analysis_start_y:analysis_end_y,
                                            analysis_start_x:analysis_end_x]
            # find darkest point in each channel individually (ignore axis 2)
            min_rgb_values = np.min(analysis_region, axis=(0, 1))
            print(f"INFO: Darkest values:\n"
                  f"      RED:   {min_rgb_values[0]}\n"
                  f"      GREEN: {min_rgb_values[1]}\n"
                  f"      BLUE:  {min_rgb_values[2]}\n",
                  file=sys.stderr)
            # TODO: find spot closest to middle gray and use it to white balance
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

        # apply manual white balance
        balanced_red_channel = working_image[:, :, 0].copy() * args.red_balance
        balanced_green_channel = working_image[:, :, 1].copy() * args.green_balance
        balanced_blue_channel = working_image[:, :, 2].copy() * args.blue_balance
        working_image = np.stack([balanced_red_channel,
                                  balanced_green_channel,
                                  balanced_blue_channel],
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
