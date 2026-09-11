#!/bin/sh

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

# example of how to perform batch conversions using the CLI utility
# $1 is directory
# $2 is raw file extension (NEF, for example)
# $3 is output directory
# $4 is ffc image
# $5 is preset
# $6 is location of neg_inverter project

WORK_DIR="."
if [ -n "$6" ]; then
    WORK_DIR="$6"
fi

for item in "$1"/*."$2"; do
    if [ "$4" != "basename $item" ]; then
        python3 "$WORK_DIR/inverter/inverter.py" \
            -p "$WORK_DIR/icc_profiles/Gray-elle-V4-g10.icc" \
            --bw \
            --preset "$5" \
            --crop-inset 0.91 \
            --ffc "$4" \
            "$item" \
            "$3/$(basename "$item" ".$2").tiff"
    else
        # this clause simply don't work
        echo "Skipping FFC source file."
    fi
done
