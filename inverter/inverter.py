import sys
import argparse
import numpy as np
import rawpy
import tifffile
import colour


def main():
    # parse user-supplied arguments
    parser = argparse.ArgumentParser(prog="Neg Inverter",
                                     description="Automatically invert your film negatives")
    parser.add_argument('image_path',
                        type=str, default=None,
                        help="Input RAW file to process.")
    parser.add_argument('output_path', type=str, default=None,
                        help="Output path for final processed image.")
    parser.add_argument('--input-lut', '-l',
                        type=str, default=None,
                        help="LUT applied into working colour space.")
    parser.add_argument('--output-lut', '-L',
                        type=str, default=None,
                        help="LUT applied out of working colour space.")
    parser.add_argument('--icc', '-i',
                        type=str, default=None,
                        help="ICC profile to embed in final TIFF export.")

    args = parser.parse_args()
    # read in 
    with rawpy.imread(args.image_path) as raw:
        # read in camera sensor data at 16bpp int
        image_data_uint16 = raw.postprocess(use_camera_wb=True,
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

        # apply modifications in working space

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
