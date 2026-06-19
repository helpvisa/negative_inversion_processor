import argparse


def parse_user_arguments():
    # parse user-supplied arguments
    parser = argparse.ArgumentParser(prog="Neg Inverter",
                                     description="Automatically invert your film negatives")
    # input / output
    parser.add_argument('image_path',
                        type=str, default=None,
                        nargs='?',
                        help="Input RAW file to process.")
    parser.add_argument('output_path', type=str, default=None,
                        nargs='?',
                        help="Output path for final processed image.")
    parser.add_argument('--tiff-format', type=str, default="f16",
                        help="Format in which to save output (f16, f32, u8, u16).")
    # create and load presets
    parser.add_argument('--preset',
                        type=str, default=None,
                        help="Preset to apply to image (no analysis)")
    parser.add_argument('--generate-preset',
                        action='store_true',
                        help="Generate an inversion preset at the output path.")
    # icc profile embedding and ocio config loading
    parser.add_argument('--icc', '-p',
                        type=str, default=None,
                        help="ICC profile to embed in final TIFF export.")
    parser.add_argument('--ocio-config',
                        type=str, default=None,
                        help="OCIO configuration file to load.")
    parser.add_argument('--ocio-in',
                        type=str, default=None,
                        help="OCIO input setting to use. Should be some form of "
                             "Linear Rec.2020.")
    parser.add_argument('--ocio-out',
                        type=str, default=None,
                        help="OCIO output setting to use.")
    # image modifiers
    parser.add_argument('--blur',
                        type=int, default=None,
                        help="Optionally blur the working image to average pixel values.")
    parser.add_argument('--resize',
                        type=float, default=1.0,
                        help="Resize the image for the processing pipeline.")
    parser.add_argument('--analysis-inset', '-A',
                        type=float, default=0.5,
                        help="How far to inset analysis bounding box (default 0.5).")
    parser.add_argument('--analysis-height', '-H',
                        type=int, default=2666,
                        help="Size of analysis bounding box (height).")
    parser.add_argument('--exposure-comp',
                        type=float, default=1.0,
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
    parser.add_argument('--red-balance', '-R',
                        type=float, default=1.0,
                        metavar="FLOAT",
                        help="Custom red balance (in density space).")
    parser.add_argument('--green-balance', '-G',
                        type=float, default=1.0,
                        metavar="FLOAT",
                        help="Custom green balance (in density space).")
    parser.add_argument('--blue-balance', '-B',
                        type=float, default=1.0,
                        metavar="FLOAT",
                        help="Custom blue balance (in density space).")
    parser.add_argument('--wb-point',
                        nargs=2,
                        metavar=("X1", "Y1"),
                        type=int, default=None,
                        help="XY point around which to white balance, space-separated"
                             "(this is usually the emulsion itself).")
    parser.add_argument('--ref-points',
                        nargs=4,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        type=int, default=None,
                        help="XY points to use as neutral gray reference, space-separated.")
    parser.add_argument('--exponent',
                        type=float, default=1.5,
                        help="Adjust global contrast (default 1.0).")
    parser.add_argument('--skip-normalize',
                        action='store_true',
                        help="Normalize final output between 0 and 1 (default true).")
    return parser.parse_args()
