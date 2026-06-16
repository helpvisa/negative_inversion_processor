from functools import partial
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from inverter import process_negative


# cute little wrapper class for Tk
class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()


# global variables
LOADED_NEGATIVE = None


# callback functions
def load_negative():
    global LOADED_NEGATIVE
    LOADED_NEGATIVE = filedialog.askopenfilename()
    print(LOADED_NEGATIVE)


def preview_negative(canvas_widget: tk.Canvas, image_id):
    args = {
        "image_path": LOADED_NEGATIVE,
        "resize": (600, 400),
        "analysis_width": 3800,
        "analysis_height": 2400,
        "exposure_comp": 0.5,
        "skip_inversion": False,
        "skip_auto_adjustments": False,
        "debug_analysis_region": False,
        "wb_steps": 10,
        "red_balance": 1.0,
        "green_balance": 1.0,
        "blue_balance": 1.0,
        "custom_white_point": None,
        "custom_gray_point": None,
        "custom_black_point": None
    }
    print(args["image_path"])
    processed_neg = process_negative(args)
    pil_image = Image.fromarray(processed_neg.astype(np.uint8))
    tk_image = ImageTk.PhotoImage(pil_image)
    canvas_widget.itemconfig(image_id, image=tk_image)


# initialize gui
app = App()

# define gui
app.master.title("Neg Inverter")
app.master.minsize(800, 600)
choose_negative = tk.Button(app, text="Load Negative",
                          command=load_negative)
choose_negative.pack()
image_canvas = tk.Canvas(app, width=600, height=400)
image_id = image_canvas.create_image(0, 0, anchor="nw")
image_canvas.pack()
conversion_action = partial(preview_negative, image_canvas, image_id)
convert_negative = tk.Button(app, text="Convert",
                             command=conversion_action)
convert_negative.pack()
# filedialog.askopenfile(mode='r')

# run gui
app.mainloop()
