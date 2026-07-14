import sys
import math
from types import SimpleNamespace
import numpy as np
from scipy import ndimage
import parse_cli_arguments
from presets import save_preset, load_preset
from PIL import ImageCms
from colour_management import (convert_to_sRGB, ocio_load_and_create_config,
                               ocio_convert_colorspace)
from processing import (load_raw_image, save_image,
                        invert_to_density, density_to_luminance,
                        shift_blacks, white_balance, density_balance,
                        apply_gain, apply_addition, normalize_image,
                        convert_to_grayscale_from_g, process_all_adjustments)


def process_negative(source_image, args):
    """
    Process a negative, derived from a camera RAW file.
    The working_image must be an f32 numpy array.

    `args` represents a complete collection of configuration options, derived
    primarily from the options exposed in the command-line interface.
    So long as a dictionary of the appropriate values are provided, the args
    do not need to come from the CLI itself.

    A full list of applicable options is as follows:
        - `image_path`
        - `blur`
        - `preset`
        - `generate_preset`
        - `analysis_inset`
        - `exposure_comp`
        - `skip_inversion`
        - `skip_auto_adjustments`
        - `debug_analysis_region`
        - `red_ratio`
        - `blue_ratio`
        - `red_gain`
        - `green_gain`
        - `blue_gain`
        - `wb_point`

    See `parse_cli_arguments.py` for more detailed information about each of
    the individual arguments available in the `args` dictionary.
    """
    # convert args from a dictionary into a namespace
    if type(args) is dict:
        args = SimpleNamespace(**args)

    # create working image from source and apply green channel exponent
    working_image = source_image.copy()

    # should probably be reworked to use UV of image
    if args.resize and args.resize != 1.0:
        if args.wb_point:
            args.wb_point[0] = int(math.floor(args.wb_point[0] * args.resize))
            args.wb_point[1] = int(math.floor(args.wb_point[1] * args.resize))
        resize_factor = (args.resize, args.resize, 1)
        working_image = ndimage.zoom(working_image,
                                     resize_factor,
                                     order=3)
    if args.blur:
        print("Creating blurred analysis image...", file=sys.stderr)
        working_image = ndimage.gaussian_filter(working_image, sigma=(args.blur,
                                                                      args.blur,
                                                                      0))

    # use reference rec2020 luminance weights
    rec2020_lum_weights = np.array([0.2627, 0.6780, 0.05903])
    # store adjustments in array for application and export
    adjustments = []
    # determine and build analysis region
    image_height, image_width, image_channels = working_image.shape
    analysis_width = int(image_width * args.analysis_inset)
    analysis_height = int(image_height * args.analysis_inset)
    analysis_start_x = (image_width - analysis_width) // 2
    analysis_end_x = analysis_start_x + analysis_width
    analysis_start_y = (image_height - analysis_height) // 2
    analysis_end_y = analysis_start_y + analysis_height
    # store as array of [x,y] arrays
    analysis_bounding_box = [[analysis_start_x, analysis_start_y],
                             [analysis_end_x, analysis_end_y]]

    # perform pre-inversion white balance
    if not args.skip_auto_adjustments:
        new_adjustment = white_balance(working_image, analysis_bounding_box,
                                       rec2020_lum_weights, args.wb_point,
                                       mode='mult')
        working_image = apply_gain(working_image, new_adjustment["values"])
        adjustments.append(new_adjustment.copy())

    # invert image to begin work in "density space"
    if not args.skip_inversion:
        new_adjustment = {"type": "invert_to_density", "values": None}
        working_image = invert_to_density(working_image)
        adjustments.append(new_adjustment.copy())

        if not args.skip_auto_adjustments:
            # adjust the individual densities to eliminate colour casts
            scale_adjustment, shift_adjustment = density_balance(working_image,
                                                                 analysis_bounding_box,
                                                                 args.exponent,
                                                                 None,
                                                                 args.red_ratio,
                                                                 args.blue_ratio)
            working_image = apply_gain(working_image, scale_adjustment["values"])
            working_image = apply_addition(working_image, shift_adjustment["values"])
            adjustments.append(scale_adjustment.copy())
            adjustments.append(shift_adjustment.copy())
            # readjust white balance
            new_adjustment = white_balance(working_image, analysis_bounding_box,
                                           rec2020_lum_weights, args.wb_point,
                                           mode='add')
            working_image = apply_addition(working_image, new_adjustment["values"])
            adjustments.append(new_adjustment.copy())

        if args.red_gain != 1.0 or args.green_gain != 1.0 or args.blue_gain != 1.0:
            # apply a user-adjustable gain to balance in density space
            new_adjustment = {
                "type": "mult",
                "values": (args.red_gain,
                           args.green_gain,
                           args.blue_gain)
            }
            working_image = apply_gain(working_image, new_adjustment["values"])
            adjustments.append(new_adjustment.copy())
            print(f"ADJUSTMENT: User gain:\n"
                  f"          RED:   {args.red_gain}\n"
                  f"          GREEN: {args.green_gain}\n"
                  f"          BLUE:  {args.blue_gain}",
                  file=sys.stderr)
        # map density to luminance
        new_adjustment = {"type": "density_to_luminance", "values": None}
        working_image = density_to_luminance(working_image)
        adjustments.append(new_adjustment.copy())

    # normalize image back into 0-1 range to prevent clipping
    if args.normalize:
        print("INFO: Normalizing final output.")
        new_adjustment = normalize_image(working_image, rec2020_lum_weights,
                                         args.analysis_inset)
        adjustments.append(new_adjustment.copy())
    # apply exposure compensation
    if args.exposure_comp != 1.0:
        new_adjustment = {
            "type": "gain",
            "values": (args.exposure_comp,
                       args.exposure_comp,
                       args.exposure_comp)
        }
        working_image = apply_gain(working_image, new_adjustment["values"])
        adjustments.append(new_adjustment.copy())

    # TODO: display the analysis region and points if debug toggle enabled
    return adjustments


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
    source_image = load_raw_image(args.image_path).astype(np.float32) / 65535.0
    adjustments = []
    if args.preset:
        adjustments = load_preset(args.preset)
    else:
        adjustments = process_negative(source_image, args)
    if args.generate_preset:
        save_preset(args.output_path, adjustments)
    else:
        final_image = process_all_adjustments(source_image, adjustments).astype(np.float32)
        # convert to b&w
        if args.bw:
            print("PROCESS: Convert to black and white.", file=sys.stderr)
            final_image = convert_to_grayscale_from_g(final_image)
        # apply crop if necessary
        if args.crop_inset < 1.0:
            print("PROCESS: Applying crop.", file=sys.stderr)
            # *rest required as number of channels not returned when b&w
            final_height, final_width, *rest = final_image.shape
            crop_width = int(final_width * args.crop_inset)
            crop_height = int(final_height * args.crop_inset)
            crop_start_x = (final_width - crop_width) // 2
            crop_end_x = crop_start_x + crop_width
            crop_start_y = (final_height - crop_height) // 2
            crop_end_y = crop_start_y + crop_height
            final_image = final_image[crop_start_y:crop_end_y,
                                      crop_start_x:crop_end_x]
        # save image to disk
        # do we possess an icc profile to embed?
        icc_profile = None
        if args.icc:
            with open(args.icc, "rb") as icc_file:
                icc_profile = icc_file.read()
        elif args.ocio_config and args.ocio_in and args.ocio_out:
            print("INFO: Using OCIO configuration.", file=sys.stderr)
            ocio_config = ocio_load_and_create_config(args.ocio_config)
            ocio_convert_colorspace(final_image,
                                    ocio_config, args.ocio_in, args.ocio_out)
            icc_profile = None
        else:
            print("WARN: Image was processed in linear Rec2020 colour space,\n"
                  "      but you have not provided an ICC profile to embed.\n"
                  "      Image will be converted to and saved as sRGB.",
                  file=sys.stderr)
            final_image, icc_profile = convert_to_sRGB(final_image)
            # convert the icc profile into bytes for tifffile to accept it
            icc_profile = ImageCms.ImageCmsProfile(icc_profile).tobytes()
        save_image(final_image, args.output_path, args.tiff_format, icc_profile)
    print(f"Done processing {args.image_path}!", file=sys.stderr)


if __name__ == "__main__":
    main()
