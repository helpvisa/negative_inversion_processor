# evil mutable global variables
GLOBAL_FLAGS = {
    "platform": None
}

REC2020_WEIGHTS = [0.2627, 0.6780, 0.0593]

# combobox globals
SAVE_FORMATS = [
    {'data': "u8", 'display': "TIFF (8-bit integer)"},
    {'data': "u16", 'display': "TIFF (16-bit integer)"},
    {'data': "f16", 'display': "TIFF (16-bit floating-point)"},
    {'data': "f32", 'display': "TIFF (32-bit floating-point)"},
    {'data': "ju8", 'display': "JPEG (8-bit integer)"}
]

# index of photos that have been loaded
# layout for each entry is:
#   - file_name (index)
#   - width
#   - height
#   - edit_params
PHOTO_INDEX = dict()
