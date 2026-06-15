import sys
import parse_cli_arguments


def parse_configuration_file(preset_path):
    """
    Reads custom preset file format.
    This is probably a terrible, terrible parser.
    Why didn't I just use JSON? I'unno.
    """
    print(f"INFO: Reading {preset_path}",
          file=sys.stderr)
    with open(preset_path, "r", encoding="utf-8") as preset:
        processable_lines = [line for line in preset if not line.strip()
                                                                .startswith("#")]
        # convert to a single line
        one_liner = ""
        for line in processable_lines:
            # remove silly newline characters
            line = line.replace("\n", "")
            # I see you too, Windows users
            line = line.replace("\r", "")
            one_liner += line
        # split on semicolon
        raw_params = one_liner.split(";")
        params = {
            "black_adjustment": raw_params[0].split(","),
            "gray_adjustment": raw_params[1].split(","),
            "white_adjustment": raw_params[2].split(","),
            "user_adjustment": raw_params[3].split(",")
        }
        return params


if __name__ == "__main__":
    args = parse_cli_arguments.parse_user_arguments()
    print(parse_configuration_file(args.preset))
