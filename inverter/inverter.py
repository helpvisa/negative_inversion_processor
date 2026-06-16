# system
import sys
import numpy as np
import rawpy
import tifffile
import scipy
# custom
import parse_cli_arguments
from processing import (invert_to_density, density_to_luminance,
                        emulsion_wb, density_wb,
                        density_balance_gain, density_custom_gain)


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

    # read in raw file
    with rawpy.imread(args.image_path) as raw:
        print(f"Processing image: {args.image_path}", file=sys.stderr)
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
        working_image = image_data_uint16.astype(np.float32) / 65535.0

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
                                                         analysis_bounding_box,
                                                         rec2020_lum_weights,
                                                         custom_black_point)

        # invert image to begin work in "density space"
        if not args.skip_inversion:
            working_image = invert_to_density(working_image)

            if not args.skip_auto_adjustments:
                # find spot closest to middle gray for density adjustment
                total_aggregate_change = [0, 0, 0]
                for i in range(args.wb_steps):
                    print(f"INFO: (ITERATION {i+1} of {args.wb_steps})",
                          file=sys.stderr)
                    custom_gray_point = args.custom_gray_point if args.custom_gray_point else None
                    working_image, dwb_adjustments = density_wb(working_image,
                                                                analysis_bounding_box,
                                                                rec2020_lum_weights,
                                                                custom_gray_point)
                    # break if we are already "fully balanced"
                    if dwb_adjustments[0] == dwb_adjustments[1] == dwb_adjustments[2]:
                        print("INFO: Perfectly balanced! Wrapping up early.",
                              file=sys.stderr)
                        break
                    # aggregate changes across loops
                    total_aggregate_change[0] += dwb_adjustments[0]
                    total_aggregate_change[1] += dwb_adjustments[1]
                    total_aggregate_change[2] += dwb_adjustments[2]
                # print out total aggregate change
                print(f"ADJUSTED: Total aggregate colour offsets:\n"
                      f"          RED:   {total_aggregate_change[0]}\n"
                      f"          GREEN: {total_aggregate_change[1]}\n"
                      f"          BLUE:  {total_aggregate_change[2]}",
                      file=sys.stderr)
                # apply an automated white balance based on white point
                custom_white_point = args.custom_white_point if args.custom_white_point else None
                working_image, dbgw_adjustments = density_balance_gain(working_image,
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
