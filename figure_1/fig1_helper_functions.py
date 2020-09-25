from visual_behavior_research.projects.tbd.session import Session
import visual_behavior_research.utilities as vbu
import visual_behavior_research.plotting as vbp
import visual_behavior_research.projects.tbd.database_utilities as du
import visual_behavior_research.projects.tbd.utilities as tbdu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

import isx

def zscore_traces(traces):

    traces_zs = pd.DataFrame()
    traces_no_nan = traces.dropna(axis='index').copy()
    traces_zs['t'] = traces_no_nan['t']
    for column in [C for C in traces.columns if 't' not in C]:
        traces_zs[column] = stats.zscore(traces_no_nan[column])
    return traces_zs