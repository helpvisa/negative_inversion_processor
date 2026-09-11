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

# evil mutable global variables
GLOBAL_FLAGS = {
    "platform": None
}

REC2020_WEIGHTS = [0.2627, 0.6780, 0.0593]
SAVE_FORMATS = [
    {'data': "u8", 'display': "TIFF (8-bit integer)"},
    {'data': "u16", 'display': "TIFF (16-bit integer)"},
    {'data': "f16", 'display': "TIFF (16-bit floating-point)"},
    {'data': "f32", 'display': "TIFF (32-bit floating-point)"},
    {'data': "ju8", 'display': "JPEG (8-bit integer)"}
]
RAW_EXTENSIONS= [
    "3fr", "ari", "arw", "bay", "bmq", "braw", "cap", "iiq", "eip", "cr2", 
    "cr3", "crw", "dcs", "dcr", "drf", "k25", "kdc", "dng", "erf", "fff", 
    "gpr", "mdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef", "ptx", 
    "pxn", "r3d", "raf", "raw", "rw2", "rwl", "sr2", "srf", "srw", "x3f"
]

# index of photos that have been loaded
# layout for each entry is:
#   - file_name (index)
#   - width
#   - height
#   - edit_params
PHOTO_INDEX = dict()
