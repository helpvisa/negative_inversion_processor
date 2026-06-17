#!/bin/sh

# example of how to perform batch conversions using the CLI utility
# $1 is directory, $2 is raw file extension (NEF, for example)
# $3 is preset profile, $4 is directory of neg_invert project (default current)
# $5 is output directory

WORK_DIR="."
if [ -n "$4" ]; then
    WORK_DIR="$4"
fi

for item in "$1"/*."$2"; do
    python3 "$WORK_DIR/inverter/inverter.py" \
        -p "$WORK_DIR/profile.icc" \
        --preset "$3" \
        "$item" \
        "$5/$(basename $item .$2).tiff"
done
