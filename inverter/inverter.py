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
    parser.add_argument('--invert', '-i',
                        action='store_true',
                        help="Directly invert the supplied image "
                             "(skip 1d LUT).")
    parser.add_argument('--map-density', '-m',
                        action='store_true',
                        help="Apply density to luminance mapping on output.")
    parser.add_argument('--red-balance', '-r',
                        type=float, default=1.0,
                        help="Multiplier for red channel (pre-inversion).")
    parser.add_argument('--green-balance', '-g',
                        type=float, default=1.0,
                        help="Multiplier for blue channel (pre-inversion).")
    parser.add_argument('--blue-balance', '-b',
                        type=float, default=1.0,
                        help="Multiplier for green channel (pre-inversion).")
    parser.add_argument('--red-power', '-R',
                        type=float, default=1.0,
                        help="Power for red channel (post-inversion).")
    parser.add_argument('--green-power', '-G',
                        type=float, default=1.0,
                        help="Power for green channel (post-inversion).")
    parser.add_argument('--blue-power', '-B',
                        type=float, default=1.0,
                        help="Power for blue channel (post-inversion).")
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
                                            output_color=rawpy.ColorSpace.raw)

        # convert data to 32-bit floating point
        image_data_f32 = image_data_uint16.astype(np.float32) / 65535.0
        # and move it into a linear Rec. 2020 space
        image_data_xyz = colour.RGB_to_XYZ(image_data_f32, "sRGB")
        image_data_rec2020 = colour.XYZ_to_RGB(image_data_xyz, "ITU-R BT.2020")

        # apply input lut (if present)
        working_image = image_data_rec2020
        if args.input_lut:
            input_lut = colour.read_LUT(args.input_lut)
            working_image = input_lut.apply(image_data_rec2020)

        # perform adjustments in working colour space
        # apply white balance
        balanced_red_channel = working_image[:, :, 0].copy() * args.red_balance
        balanced_green_channel = working_image[:, :, 1].copy() * args.green_balance
        balanced_blue_channel = working_image[:, :, 2].copy() * args.blue_balance
        working_image = np.stack([balanced_red_channel,
                                  balanced_green_channel,
                                  balanced_blue_channel],
                                 axis=2)
        # invert image directly
        if args.invert:
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
        # adjust channel power
        power_red_channel = np.power(working_image[:, :, 0].copy(),
                                     args.red_power)
        power_green_channel = np.power(working_image[:, :, 1].copy(),
                                       args.green_power)
        power_blue_channel = np.power(working_image[:, :, 2].copy(),
                                      args.blue_power)
        working_image = np.stack([power_red_channel,
                                  power_green_channel,
                                  power_blue_channel],
                                 axis=2)
        # map density to luminance
        if args.map_density:
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
