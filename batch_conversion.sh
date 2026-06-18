#!/bin/sh

# example of how to perform batch conversions using the CLI utility
# $1 is directory, $2 is raw file extension (NEF, for example)
# $3 is preset profile, $4 is output directory
# $5 is directory of neg_invert project (default current)

WORK_DIR="."
if [ -n "$5" ]; then
    WORK_DIR="$5"
fi

for item in "$1"/*."$2"; do
    python3 "$WORK_DIR/inverter/inverter.py" \
        -p "$WORK_DIR/profile.icc" \
        --preset "$3" \
        "$item" \
        "$4/$(basename $item .$2).tiff"
done
