import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pandas as pd
import seaborn as sns

def add_label(fig, label_text, xpos, ypos, **kwargs):
    '''
    add labels to a figure canvas using the placeAxesOnGrid infrastructure

    fig: figure handle
    label_text : text of label, 
    xpos, ypos: floats from 0 to 1 defining where on the canvas the label should be
    kwargs: additional keyword arguments for matplotlib text()
    ''' 
    label_axis = placeAxesOnGrid(
        fig, 
        xspan=[xpos,xpos + 0.01],
        yspan=[ypos,ypos + 0.01],
    )
    label_axis.text(0,0,label_text,**kwargs)
    label_axis.axis('off')


def placeAxesOnGrid(
        fig,
        dim=[1, 1],
        xspan=[0, 1],
        yspan=[0, 1],
        wspace=None,
        hspace=None,
        sharex=False,
        sharey=False,
        frameon=True,
        
):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec
    DRO

    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively
        labels: a list of dictionaries of format [{label:LABELTEXT, x:XPOS, y:YPOS}]

    returns:
        subaxes handles
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100, 100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        dim[0],
        dim[1],
        subplot_spec=outer_grid[int(100 * yspan[0]):int(100 * yspan[1]), int(100 * xspan[0]):int(100 * xspan[1])],
        wspace=wspace,
        hspace=hspace
    )

    # NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0] * [dim[1] * [fig]]  # filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            if row > 0 and sharex == True:
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:
                share_y_with = inner_ax[row][0]
            else:
                share_y_with = None

            inner_ax[row][col] = plt.Subplot(
                fig,
                inner_grid[idx],
                sharex=share_x_with,
                sharey=share_y_with,
                frameon=frameon,
            )

            if row == dim[0] - 1 and sharex == True:
                inner_ax[row][col].xaxis.set_ticks_position('bottom')
            elif row < dim[0] and sharex == True:
                plt.setp(inner_ax[row][col].get_xtick)

            if col == 0 and sharey == True:
                inner_ax[row][col].yaxis.set_ticks_position('left')
            elif col > 0 and sharey == True:
                plt.setp(inner_ax[row][col].get_yticklabels(), visible=False)

            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax


def save_figure(fig, fname, formats=['.png'], transparent=False, dpi=300, **kwargs):
    mpl.rcParams['pdf.fonttype'] = 42 # makes text editable
    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])
    for f in formats:
        fig.savefig(
            fname + f,
            transparent=transparent,
            orientation='landscape',
            dpi=dpi
        )
