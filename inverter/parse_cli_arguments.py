# This file is part of Negative Inversion Processor.
#
# Negative Inversion Processor  is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# Negative Inversion Processor is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# Negative Inversion Processor. If not, see <https://www.gnu.org/licenses/>. 

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
                        type=float, default=0.75,
                        help="How far to inset analysis bounding box (default 0.75).")
    parser.add_argument('--crop-inset', '-C',
                        type=float, default=1.0,
                        help="How far into the frame to crop (default 1.0 = no crop,"
                             "< 1.0 = percentage of frame to keep).")
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
    parser.add_argument('--red-gain', '-R',
                        type=float, default=1.0,
                        metavar="FLOAT",
                        help="Custom red gain (in density space).")
    parser.add_argument('--green-gain', '-G',
                        type=float, default=1.0,
                        metavar="FLOAT",
                        help="Custom green gain (in density space).")
    parser.add_argument('--blue-gain', '-B',
                        type=float, default=1.0,
                        metavar="FLOAT",
                        help="Custom blue gain (in density space).")
    parser.add_argument('--red-offset',
                        type=float, default=0.0,
                        metavar="FLOAT",
                        help="Custom red offset (in density space).")
    parser.add_argument('--green-offset',
                        type=float, default=0.0,
                        metavar="FLOAT",
                        help="Custom green offset (in density space).")
    parser.add_argument('--blue-offset',
                        type=float, default=0.0,
                        metavar="FLOAT",
                        help="Custom blue offset (in density space).")
    parser.add_argument('--wb-point',
                        nargs=2,
                        metavar=("X1", "Y1"),
                        type=int, default=None,
                        help="XY point around which to white balance, space-separated"
                             "(this is usually the emulsion itself).")
    parser.add_argument('--ref-point',
                        nargs=2,
                        metavar=("X1", "Y1"),
                        type=int, default=None,
                        help="XY point around which to balance density, "
                             "space-separated.")
    parser.add_argument('--exponent',
                        type=float, default=1.5,
                        help="Adjust global contrast (default 1.0).")
    parser.add_argument('--red-ratio',
                        type=float, default=1.36,
                        help="Adjust red ratio of film base.")
    parser.add_argument('--blue-ratio',
                        type=float, default=0.86,
                        help="Adjust blue ratio of film base.")
    parser.add_argument('--pivot',
                        type=float, default=0.745,
                        help="Middle gray pivot point.")
    parser.add_argument('--normalize',
                        action='store_true',
                        help="Normalize final output between 0 and 1 (default true).")
    parser.add_argument('--bw',
                        action='store_true',
                        help="Save final image in black and white.")
    parser.add_argument('--ffc',
                        type=str, default=None,
                        help="Specify an image to use for flat-field correction.")
    parser.add_argument('--ffc-strength',
                        type=float, default=1.0,
                        help="Strength of flat-field correction.")
    parser.add_argument('--tonemap',
                        action='store_true',
                        help="Apply ACES tonemapping to final output.")
    parser.add_argument('--toe',
                        type=float, default=1.0,
                        help="Toe strength when applying tonemapping.")
    return parser.parse_args()
