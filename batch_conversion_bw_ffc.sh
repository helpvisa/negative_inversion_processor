#!/bin/sh

# example of how to perform batch conversions using the CLI utility
# $1 is directory, $2 is raw file extension (NEF, for example)
# $3 is preset profile, $4 is output directory
# $5 is ffc image
# $6 is location of neg_inverter project

WORK_DIR="."
if [ -n "$6" ]; then
    WORK_DIR="$6"
fi

for item in "$1"/*."$2"; do
    if [ "$5" != "basename $item" ]; then
        python3 "$WORK_DIR/inverter/inverter.py" \
            -p "$WORK_DIR/bw-profile.icc" \
            --preset "$3" \
            --bw \
            --crop-inset 0.89 \
            --ffc "$5" \
            "$item" \
            "$4/$(basename $item .$2).tiff"
    else
        # this shit don't work
        echo "Skipping FFC source file."
    fi
done
