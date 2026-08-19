#!/bin/sh

# example of how to perform batch conversions using the CLI utility
# $1 is directory
# $2 is raw file extension (NEF, for example)
# $3 is output directory
# $4 is ffc image
# $5 is red ratio
# $6 is blue ratio
# $7 is location of neg_inverter project

WORK_DIR="."
if [ -n "$7" ]; then
    WORK_DIR="$7"
fi

for item in "$1"/*."$2"; do
    if [ "$4" != "basename $item" ]; then
        python3 "$WORK_DIR/inverter/inverter.py" \
            -p "$WORK_DIR/profile.icc" \
            --red-ratio "$5" \
            --blue-ratio "$6" \
            --crop-inset 0.91 \
            --ffc "$4" \
            "$item" \
            "$3/$(basename "$item" ".$2").tiff"
    else
        # this clause simply don't work
        echo "Skipping FFC source file."
    fi
done
