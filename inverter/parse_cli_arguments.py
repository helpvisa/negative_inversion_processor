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
    # create and load presets
    parser.add_argument('--preset',
                        type=str, default=None,
                        help="Preset to apply to image (no analysis)")
    parser.add_argument('--generate-preset',
                        action='store_true',
                        help="Generate an inversion preset.")
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
    parser.add_argument('--custom-white-point',
                        nargs=2,
                        type=int, default=None,
                        help="XY of point to balance around white, space-separated.")
    parser.add_argument('--custom-gray-point',
                        nargs=2,
                        type=int, default=None,
                        help="XY of point to balance around gray, space-separated.")
    parser.add_argument('--custom-black-point',
                        nargs=2,
                        type=int, default=None,
                        help="XY of point to balance around black, space-separated"
                             "(this is usually the emulsion itself).")
    return parser.parse_args()
