"""
What you should know about browser:

* (Browser() << KonIQ) + (Browser() << CLIVE) -- close the first one will close all
* sync mode: show results of different methods. (qmap comparison, or one showing qmap, one showing the results, very flexible)

# Browser(methods=['PaQ2PiQ-BM', 'PaQ2PiQ-RM']) << KonIQ

# Browser() << KoNViD
"""

from .bunch import IqaDataBunch
from cached_property import cached_property
# IqaData, Rois0123Label, cached_property
from pathlib import Path
import os, io
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk # put it after tkinter to overwrite tkinter.Image
import numpy as np  # np.roll
import logging

"""
# %% browse one database
from fastiqa.bunches.iqa.im2mos import *
from fastiqa.iqa import *
dls = Im2MOS(bs=2)
dls.bs
dls.df
dls.get_df()
# %%
dls2 = dls << CLIVE
dls2.bs
dls.bs

# %%
# dls.show_batch()
dls.bs


# %%
self = IqaDataBunch() << CLIVE
self.df
# %%
from fastiqa.iqa import *
from fastiqa.browser import *
self = Browser() << LIVE_FB_IQA # CLIVE
self

propobj = getattr(self.__class__, 'index', None)
propobj
# %%
self.df

self.reload()
print(self._df_view)
self.df
# %%

# NOTE: exit to run next browser
Browser(KonIQ)
Browser(FLIVE)
Browser(FLIVE640)

# %% browse multiple database at the same time
from fastiqa.gui import *; Browser(FLIVE640) + Browser(CLIVE) + Browser(KonIQ)


# %%
from fastiqa.browser import *
from fastiqa.iqa import *
# Browser() << KonIQ
(Browser() << KonIQ) + (Browser() << CLIVE)
# %%
a.label_types
a.label_col
# Browser() << CLIVE
#


# %%


from fastiqa.vqa import *



# Browser << KonIQ << CLIVE
VidBrowser() << KoNViD
# %%
"""

class Browser(IqaDataBunch):
    # TODO label_types: also show predictions
    pred = None
    fn = None
    img = None
    tk_img = None
    canvas = None
    _index = 0
    percent = 1  # 100%
    cfg_rectangle = {}
    hide_scores = False
    opt_bbox_width = [4, 0, 1]
    out_dir = Path('')
    _df_view = None
    width = None
    height = None
    # label_types = None # 'mos', # 'mos', 'PaQ2PiQ', 'NIQE'
    label_range = None # map the mos to (0, 100)
    roi_col =  [["left", "top", "right", "bottom"]]

    @cached_property
    def label_cols(self):
        return self.label_col if isinstance( self.label_col, (list, tuple) ) else [self.label_col]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_df(self):
        df = super().get_df()
        self.opt_label_col = self.label_cols
        self.roi_col = np.array(self.roi_col).reshape(-1,4)
        self.opt_roi_col_idx = list(range(len(self.roi_col)))
        # if self.width is not None: #  and not isinstance(self.label_col, (list, tuple)):
        if len(self.opt_label_col) == 1: # no roi location
            df['left'] = 0
            df['top'] = 0
            if self.width is not None:
                df['right'] = self.width
                df['bottom'] = self.height
            else:
                df['bottom'] = df['height']
                df['right'] = df['width']
        if self.label_range is not None:
            print('scores are mapped to (0, 100) for browsing')
            min, max = self.label_range
            for col in self.opt_label_col:
                df[col] = (df[col] - min )*100/(max-min)

        if self.pred is not None:
            print('sort by pred error')
            if len(self.pred) != len(df): # only valid set
                df = df[df.is_valid]

            df['pred'] = self.pred
            assert len(self.pred) == len(df), 'number of predictions does not match with number of actual values'
            assert len(df[df['pred'].isna()]) == 0, "self.pred = df['output'].tolist()"
            df['pred_err'] = df['pred'] - df[self.opt_label_col[0]]
            df = df.sort_values(by='pred_err', ignore_index=True) # pred > target, pred < target
        return df

    def __add__(self, other):
        other.window = Toplevel(master=self.window)
        other.load_frame()
        return self

    def load_frame(self):
        self.reload()
        self.frame = Frame(self.window, width=500, height=400, bd=1)
        self.frame.pack()
        self.frame.bind("<Key>", self.on_key)  # canvas covered by image don't response to key press...
        self.frame.bind("<Left>", self.prev)
        self.frame.bind("<Right>", self.next)
        self.frame.bind("<Up>", self.prev_mode)
        self.frame.bind("<Down>", self.next_mode)
        self.frame.bind("<Escape>", self.exit)
        self.canvas = Canvas(self.frame)
        # self.canvas.bind("<Button-1>", self.callback)
        self.frame.focus_set()
        self.window.protocol("WM_DELETE_WINDOW", self.exit)
        self.show()

    @cached_property
    def window(self):
        return Tk()

    def _repr_html_(self):
        self.load_frame()
        return self.window.mainloop()

    @property
    def index(self):
        return self._index

    @index.setter # __setattr__ conflict
    def index(self, value):
        logging.debug('index:', value)
        self._index = int(value) % len(self._df_view)

    def show(self):
        # suffix
        # zscore? prefix
        #
        def add_bbox(roi_col_idx):
            #x1, x2 = self._df_view['left' + suffix][self.index], self._df_view['right' + suffix][self.index]
            # y1, y2 = self._df_view['top' + suffix][self.index], self._df_view['bottom' + suffix][self.index]
            roi_col = self.roi_col[roi_col_idx]
            x1, y1, x2, y2 = self._df_view.loc[self.index, roi_col].tolist()

            color = 'lightgreen' if roi_col_idx == self.opt_roi_col_idx[0] else 'yellow'
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=self.opt_bbox_width[0], **self.cfg_rectangle)

            if not self.hide_scores:
                # TODO self.label_cols[0]   mos or zscore (add score_mode)
                # show all predictions?  mos, zscore, pred
                # assert type(self.label_col) != list
                s = f"{self._df_view[self.label_cols[roi_col_idx]][self.index]:.1f}"
                # if len(self.opt_roi_col_idx)==1 and self.pred is not None:
                if roi_col_idx==0 and self.pred is not None: # image score
                    s = f"Actual: {s} / Predication: {self.pred[self.index]:.1f}"  # load from the table!!!!
                text = self.canvas.create_text((x1, y1), anchor=NW, text=s)
                r = self.canvas.create_rectangle(self.canvas.bbox(text), fill=color, outline=color)
                self.canvas.tag_lower(r, text)

        self.fn = self._df_view[self.fn_col][self.index]
        file = self.path / self.folder / (str(self.fn)+self.fn_suffix) # some database (e.g. KoNViD, AVA) contain fn typed int, convert it first
        self.img = self.open_image(file)
        width, height = self.img.size
        # PIL image
        self.tk_img = ImageTk.PhotoImage(self.img)
        # tk_img = ImageTk.PhotoImage(im)
        # self.canvas.itemconfig(self.image_on_canvas, image=tk_img)
        # then it will be optimized, showing nothing

        self.canvas.delete("all")
        self.canvas.config(width=width, height=height)

        self.canvas.create_image(0, 0, image=self.tk_img, anchor=NW)

        # only for Rois0123Label
        # if isinstance(self.label, Rois0123Label):
        for idx in self.opt_roi_col_idx:
            add_bbox(idx)
        # add_bbox('_image')
        # add_bbox('_patch_1')
        # add_bbox('_patch_2')
        # add_bbox('_patch_3')

        # self.image_on_canvas =
        # self.canvas.itemconfig(self.image_on_canvas, image=self.tk_img)
        #
        # self.canvas.coords(self.patch1_on_canvas,
        #                    self._df_view.left_patch_1[self.index],
        #                    self._df_view.top_patch_1[self.index],
        #                    self._df_view.right_patch_1[self.index],
        #                    self._df_view.bottom_patch_1[self.index],
        #                    )

        self.canvas.pack()
        fn = self._df_view[self.fn_col][self.index]
        self.window.title(f'[{width}x{height}]({self.index + 1}/{len(self._df_view)}: {self.percent * 100:.2f}%) {fn}')

    # some API to custom your browser
    def open_image(self, file):
        """

        :param file:
        :return: a PIL image
        """
        return Image.open(file)  # "../data/FLIVE/EE371R/cj23478+019.jpg"
        # if self.apply_img_proc: im = self.img_proc(im)


    def prev(self, event=None):
        self.index -= 1
        self.show()

    def next(self, event=None):
        self.index += 1
        self.show()

    def prev_mode(self, event=None):
        self.opt_roi_col_idx = np.roll(self.opt_roi_col_idx, -1)
        self.show()

    def next_mode(self, event=None):
        self.opt_roi_col_idx = np.roll(self.opt_roi_col_idx, 1)
        self.show()

    # def reset(self, event):
    #     self.valid_mos = None

    def exit(self, event=None):
        self.window.destroy()

    def filter(self, func):
        df = self._df_view[func(self._df_view)]
        if len(df) == 0:
            messagebox.showwarning("Warning", "No image found!")
        else:
            self.percent = len(df) / len(self._df_view)
            self._df_view = df.reset_index()  # otherwise index 0 will be dropped
            self._index = 0
        self.show()
        return self

    def save_image(self):
        # self.grab_image(self.canvas).save(self.fn)
        # https://stackoverflow.com/questions/41940945/saving-canvas-from-tkinter-to-file?rq=1
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save(self.out_dir / self.fn.rsplit('/', 1)[1], 'jpeg')

    def reload(self):
        self._df_view = self.df

    def on_key(self, event):
        self.frame.focus_set()
        # print("pressed", repr(event.char))
        if event.char in [str(n) for n in range(10)]:
            self.reload()
            col_name = self.opt_label_col[0]
            # there might not be valid data
            self.filter(lambda x: x[col_name] // 10 == int(event.char))

        elif event.char is ' ':
            self.reload()
            self.show()
        elif event.char is 's':  # save capture
            self.save_image()

        elif event.char is 'h':  # hide score
            self.hide_scores = not self.hide_scores
            self.show()
        elif event.char is 'w':  # i
            self.opt_bbox_width = np.roll(self.opt_bbox_width, 1)
            self.show()
        else:
            pass
        # print(self.index)


    # https://stackoverflow.com/questions/9886274/how-can-i-convert-canvas-content-to-an-image
    # def grab_image(self, widget):
    #     x = self.window.winfo_rootx() + widget.winfo_x()
    #     y = self.window.winfo_rooty() + widget.winfo_y()
    #     x1 = x + widget.winfo_width()
    #     y1 = y + widget.winfo_height()
    #     return ImageGrab.grab().crop((x, y, x1, y1))
    #     # .save(filename)

    def callback(self, event):
        self.frame.focus_set()
        print("clicked at", event.x, event.y)
        print(self._df_view[self.fn_col][self.index])



class VidBrowser(Browser):
    def open_image(self, file):
        """

        :param file:
        :return: a PIL image
        """
        file = file/'image_00001.jpg'
        return Image.open(file)  # "../data/FLIVE/EE371R/cj23478+019.jpg"
        # if self.apply_img_proc: im = self.img_proc(im)

"""
WontFix
* support different backend: tkinter or matplotlib

Reference
=========

https://effbot.org/tkinterbook/tkinter-events-and-bindings.htm

Matplotlib backbone
===================

https://matplotlib.org/gallery/animation/image_slices_viewer.html


PySimpleGUI
============

PySimpleGUI is a wrapper for Tkinter and Qt (others on the way). The amount of code required to implement custom GUIs is much shorter using PySimpleGUI than if the same GUI were written directly using Tkinter or Qt.

sudo apt-get install python-tk
sudo apt-get install python3-tk

https://github.com/PySimpleGUI/PySimpleGUI

not working here, cannot switch images

Tkinter
========

sudo apt-get install python3.6-tk


wont support python 2
for browser only, support python 2

import sys
if sys.version_info[0] == 3:
    # for Python3
    from tkinter import *
    # print(TclVersion)
else:
    # for Python2
    from Tkinter import *

"""
