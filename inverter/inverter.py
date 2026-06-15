# system
import sys
import numpy as np
import rawpy
import tifffile
import colour
import scipy
# custom
import parse_cli_arguments


def main():
    args = parse_cli_arguments.parse_user_arguments()
    if not args.image_path or not args.output_path:
        print(f"ERROR: Please specify either:\n"
              f"         1. a raw file and output path\n",
              f"        2. a raw file, preset, and output path\n",
              f"        3. a raw file and preset output path (if generating a preset)\n"
              f"Run me with '-h' to see the full suite of options.",
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
        # determine and acqiure analysis region
        image_height, image_width, image_channels = working_image.shape
        analysis_start_x = (image_width - args.analysis_width) // 2
        analysis_end_x = analysis_start_x + args.analysis_width
        analysis_start_y = (image_height - args.analysis_height) // 2
        analysis_end_y = analysis_start_y + args.analysis_height

        # create pre-inversion analysis region
        analysis_region_pre = working_image[analysis_start_y:analysis_end_y,
                                            analysis_start_x:analysis_end_x]
        # perform pre-inversion white balance
        if not args.skip_auto_adjustments:
            max_rgb_values = None
            if args.custom_black_point:
                print(f"CUSTOM: Custom black point: {args.custom_black_point}",
                      file=sys.stderr)
                max_rgb_values = working_image[args.custom_black_point[1],
                                               args.custom_black_point[0]]
            else:
                # find max luminance value in region
                pre_lum_values = np.dot(analysis_region_pre, rec2020_lum_weights)
                # and find point of maximum luminance
                max_lum_y, max_lum_x = np.unravel_index(np.argmax(pre_lum_values),
                                                        pre_lum_values.shape)
                # now grab the rgb values at that point
                max_rgb_values = analysis_region_pre[max_lum_y, max_lum_x]
            print(f"INFO: White balance on emulsion (pre-inversion):\n"
                  f"      RED:   {max_rgb_values[0]}\n"
                  f"      GREEN: {max_rgb_values[1]}\n"
                  f"      BLUE:  {max_rgb_values[2]}",
                  file=sys.stderr)
            red_channel_wb = working_image[:, :, 0].copy()
            green_channel_wb = working_image[:, :, 1].copy()
            blue_channel_wb = working_image[:, :, 2].copy()
            wb_mult = np.max(max_rgb_values)
            rc_wb = wb_mult / max_rgb_values[0]
            gc_wb = wb_mult / max_rgb_values[1]
            bc_wb = wb_mult / max_rgb_values[2]
            red_channel_wb *= rc_wb
            green_channel_wb *= gc_wb
            blue_channel_wb *= bc_wb
            working_image = np.stack([red_channel_wb,
                                      green_channel_wb,
                                      blue_channel_wb],
                                     axis=2)
            print(f"ADJUSTED: White balance on emulsion (pre-inversion):\n"
                  f"          RED:   {rc_wb}\n"
                  f"          GREEN: {gc_wb}\n"
                  f"          BLUE:  {bc_wb}",
                  file=sys.stderr)

        # invert image directly
        if not args.skip_inversion:
            # we calculate density from "transmittance" using log10(1/x)
            inverse_x_points = np.array([0.001, 0.25, 0.5, 0.75, 1.0])
            inverse_y_points = np.log10(1.0 / inverse_x_points)
            inverse_spline = scipy.interpolate.CubicSpline(inverse_x_points,
                                                           inverse_y_points)
            working_image = inverse_spline(working_image)

            # automatically determine min/max point for each channel
            if not args.skip_auto_adjustments:
                # find spot closest to middle gray for density adjustment
                total_aggregate_change = [0, 0, 0]
                for i in range(args.wb_steps):
                    normalization_factor = 1.0
                    mid_gray_rgb_values = None
                    if args.custom_gray_point:
                        print(f"CUSTOM: Custom gray point: {args.custom_gray_point}",
                              file=sys.stderr)
                        mid_gray_rgb_values = working_image[args.custom_gray_point[1],
                                                            args.custom_gray_point[0]]
                        # adjust this point to actual mid-gray
                        mid_gray_luminance = mid_gray_rgb_values.dot(rec2020_lum_weights)
                        normalization_factor = mid_gray_luminance / 0.18
                        mid_gray_rgb_values /= normalization_factor
                    else:
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
                    print(f"INFO: (ITERATION {i+1} of {args.wb_steps}) "
                          f"Middle gray RGB values (normalized):\n"
                          f"      RED:   {mid_gray_rgb_values[0]}\n"
                          f"      GREEN: {mid_gray_rgb_values[1]}\n"
                          f"      BLUE:  {mid_gray_rgb_values[2]}",
                          file=sys.stderr)
                    # break if we are already "fully balanced"
                    if mid_gray_rgb_values[0] == mid_gray_rgb_values[1] == mid_gray_rgb_values[2]:
                        print(f"INFO: Perfectly balanced! Wrapping up early.",
                              file=sys.stderr)
                        break
                    # apply adds and mults to channel densities to balance them
                    mid_gray_mult = np.max(mid_gray_rgb_values)
                    red_channel_mid_gray = working_image[:, :, 0].copy()
                    green_channel_mid_gray = working_image[:, :, 1].copy()
                    blue_channel_mid_gray = working_image[:, :, 2].copy()
                    rc_change = (mid_gray_mult - mid_gray_rgb_values[0]) * normalization_factor
                    gc_change = (mid_gray_mult - mid_gray_rgb_values[1]) * normalization_factor
                    bc_change = (mid_gray_mult - mid_gray_rgb_values[2]) * normalization_factor
                    red_channel_mid_gray += rc_change
                    green_channel_mid_gray += gc_change
                    blue_channel_mid_gray += bc_change
                    working_image = np.stack([red_channel_mid_gray,
                                              green_channel_mid_gray,
                                              blue_channel_mid_gray],
                                             axis=2)
                    # aggregate changes across loops
                    total_aggregate_change[0] += rc_change
                    total_aggregate_change[1] += gc_change
                    total_aggregate_change[2] += bc_change
        # print out total aggregate change
        print(f"ADJUSTED: Total aggregate colour offsets:\n"
              f"          RED:   {total_aggregate_change[0]}\n"
              f"          GREEN: {total_aggregate_change[1]}\n"
              f"          BLUE:  {total_aggregate_change[2]}",
              file=sys.stderr)

        # apply an automated white balance based on white point
        if not args.skip_auto_adjustments:
            max_wb_lum_rgb_values = None
            if args.custom_white_point:
                print(f"CUSTOM: Custom white point: {args.custom_white_point}",
                      file=sys.stderr)
                max_wb_lum_rgb_values = working_image[args.custom_white_point[1],
                                                      args.custom_white_point[0]]
            else:
                # update analysis region
                analysis_region_wb = working_image[analysis_start_y:analysis_end_y,
                                                   analysis_start_x:analysis_end_x]
                # find brightest luminance point and balance around it
                wb_lum_values = np.dot(analysis_region_wb, rec2020_lum_weights)
                max_wb_lum_y, max_wb_lum_x = np.unravel_index(np.argmax(wb_lum_values),
                                                              wb_lum_values.shape)
                max_wb_lum_rgb_values = analysis_region_wb[max_wb_lum_y,
                                                           max_wb_lum_x]
            print(f"INFO: Final white balance values:\n"
                  f"      RED:   {max_wb_lum_rgb_values[0]}\n"
                  f"      GREEN: {max_wb_lum_rgb_values[1]}\n"
                  f"      BLUE:  {max_wb_lum_rgb_values[2]}",
                  file=sys.stderr)
            final_wb_mult = np.max(max_wb_lum_rgb_values)
            red_channel_wb_final = working_image[:, :, 0].copy()
            green_channel_wb_final = working_image[:, :, 1].copy()
            blue_channel_wb_final = working_image[:, :, 2].copy()
            rc_wb_final = final_wb_mult / max_wb_lum_rgb_values[0]
            gc_wb_final = final_wb_mult / max_wb_lum_rgb_values[1]
            bc_wb_final = final_wb_mult / max_wb_lum_rgb_values[2]
            red_channel_wb_final *= rc_wb_final
            green_channel_wb_final *= gc_wb_final
            blue_channel_wb_final *= bc_wb_final
            working_image = np.stack([red_channel_wb_final,
                                      green_channel_wb_final,
                                      blue_channel_wb_final],
                                     axis=2)
            print(f"ADJUSTED: Final white balance:\n"
                  f"          RED:   {rc_wb_final}\n"
                  f"          GREEN: {gc_wb_final}\n"
                  f"          BLUE:  {bc_wb_final}",
                  file=sys.stderr)

        # apply a final, user-adjustable gain to balance in density space
        red_channel_ub = working_image[:, :, 0].copy() * args.red_balance
        green_channel_ub = working_image[:, :, 1].copy() * args.green_balance
        blue_channel_ub = working_image[:, :, 2].copy() * args.blue_balance
        working_image = np.stack([red_channel_ub,
                                  green_channel_ub,
                                  blue_channel_ub],
                                 axis=2)
        if args.red_balance != 1.0 or args.green_balance != 1.0 or args.blue_balance != 1.0:
            print(f"ADJUSTED: User white balance:\n"
                  f"          RED:   {args.red_balance}\n"
                  f"          GREEN: {args.green_balance}\n"
                  f"          BLUE:  {args.blue_balance}",
                  file=sys.stderr)

        # map density to luminance
        if not args.skip_inversion:
            lum_x_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            lum_y_points = np.power(10, lum_x_points) * 0.01
            lum_spline = scipy.interpolate.CubicSpline(lum_x_points,
                                                       lum_y_points)
            working_image = lum_spline(working_image)

        # apply output exposure compensation
        final_image = working_image * args.exposure_comp

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
