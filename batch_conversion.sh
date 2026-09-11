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
# $1 is directory, $2 is raw file extension (NEF, for example)
# $3 is preset profile, $4 is output directory
# $5 is directory of neg_invert project (default current)

WORK_DIR="."
if [ -n "$5" ]; then
    WORK_DIR="$5"
fi

for item in "$1"/*."$2"; do
    python3 "$WORK_DIR/inverter/inverter.py" \
        -p "$WORK_DIR/icc_profiles/Rec2020-elle-V4-g10.icc" \
        --preset "$3" \
        --crop-inset 0.89 \
        "$item" \
        "$4/$(basename $item .$2).tiff"
done
