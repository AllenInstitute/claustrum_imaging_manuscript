## note: the code contained in this file relies on internal allen institute drives and private repositories
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind,ttest_rel
from scipy import stats,signal
import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import platform
import os
import pandas as pd
import json

import helper_functions

from visual_behavior_research.projects.tbd.session import Session as tbd_Session

import visual_behavior_research.projects.tbd.utilities as tbdu
import visual_behavior_research.plotting as vbp
import visual_behavior_research.plotting.dro_plots as pf
import visual_behavior_research.utilities as vbu

from visual_behavior_research.projects.tbd.identify_repeat_rois import Repeat_ROI_Identifier

import isx
from visual_behavior_research.projects.tbd.Inscopix.pipeline import make_cell_map,plot_cell_map

from visual_behavior.translator.foraging import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe

def load_passive_session_from_disk(data_path,load_cached_traces_table=True):
    '''
    load passive stimulus session, add a column denoting the closest frame in the Inscopix video
    this depends on allen institute internal packages/databases
    '''
    session = tbd_Session(data_path, load_cached_traces_table=load_cached_traces_table)

    passive_stim_trials = session.behavior_core_data['trials']
    passive_stim_trials['nearest_F_frame'] = passive_stim_trials['startframe'].map(
            lambda x: helper_functions.find_nearest_index(session.sync_data['behavior_vsync'][x], session.sync_data['fluorescence_camera'])
        )

    return session

def load_active_session_from_disk(data_path,load_cached_traces_table=True):
    '''
    load active behavior session, add columns denoting the closest frame in the Inscopix video
    this depends on allen institute internal packages/databases
    '''
    cached_data = pd.read_pickle(os.path.join(data_path,'cached_data_2019.10.04.pkl'))
    session = tbd_Session(
        data_path, 
        load_cached_traces_table = True,
        behavior_pkl_filename = which_pkl(cached_data['filename_dict'])
    )
    
    trials = session.behavior_core_data['extended_trials']
    
    rewards = session.behavior_core_data['rewards']
    # rebase time on sync
    rewards['time'] = session.sync_data['behavior_vsync'][rewards['frame'].values]
    
    licks = session.behavior_core_data['licks']
    # rebase time on sync
    licks['time'] = session.sync_data['behavior_vsync'][licks['frame'].values]
    licks['nearest_F_frame'] = licks['frame'].map(lambda x:get_F_frame(int(x), session.sync_data))
    licks['time_since_last_lick'] = licks['time'] - licks['time'].shift()
    licks['nearest_reward']=licks['time'].map(lambda x:nearest_reward(x,rewards))
    
    visual_stimuli = session.behavior_core_data['visual_stimuli']
    # rebase time on sync
    visual_stimuli['time'] = session.sync_data['behavior_vsync'][visual_stimuli['frame'].values]
    visual_stimuli['nearest_lick'] = visual_stimuli['time'].map(lambda x: nearest_lick(x,licks))
    visual_stimuli['change'] = visual_stimuli['image_name'] != visual_stimuli['image_name'].shift()
    visual_stimuli['nearest_F_frame'] = visual_stimuli['frame'].map(
        lambda x: get_F_frame(int(x), session.sync_data)
    )
    session.event_dict = define_behavior_events(session)
    
    return session


def define_behavior_events(session):
    '''
    build an event dictionary for a given session denoting frame numbers for:
        hit: responses after stimulus changes
        miss: lack of response after stimulus changes
        false_alarm (fa): responses after stimulus non-changes
        correct_reject (cr): lack of response after stimulus non-changes
    '''
    event_dict = {}
    licks = session.behavior_core_data['licks']
    visual_stimuli = session.behavior_core_data['visual_stimuli']
    trials = session.behavior_core_data['extended_trials']
    rewards = session.behavior_core_data['rewards']
    
    hit_licks = licks[
        (licks['nearest_reward']<0.1)
        &(licks['time_since_last_lick']>2.25)
        &(~np.isnan(licks['nearest_F_frame']))
    ]
    
    event_dict['hit_events'] = []
    for idx,row in hit_licks.iterrows():
        lick_time = row['time']
        event_dict['hit_events'].append(visual_stimuli.query('time < @lick_time')['nearest_F_frame'].iloc[-1])

    fa_licks = licks[
        (licks['nearest_reward']>2)
        &(licks['time_since_last_lick']>2.25)
        &(~np.isnan(licks['nearest_F_frame']))
    ]
    
    event_dict['fa_events'] = []
    for idx,row in fa_licks.iterrows():
        lick_time = row['time']
        event_dict['fa_events'].append(visual_stimuli.query('time < @lick_time')['nearest_F_frame'].iloc[-1])

    event_dict['miss_events'] = trials.query('response_type == "MISS"')['change_frame'].map(
        lambda x:get_F_frame(int(x),session.sync_data)
    )
    event_dict['miss_events'] = event_dict['miss_events'][pd.notnull(event_dict['miss_events'])]

    cr_events = visual_stimuli.query('change == False and nearest_lick > 2.25')
    #get a random sample of correct rejections (non-change flashes without nearby licks)
    event_dict['cr_events'] = cr_events.sample(min(100,len(cr_events)),random_state=0)['nearest_F_frame'] 
    event_dict['cr_events'] = event_dict['cr_events'][pd.notnull(event_dict['cr_events'])].values
    
    # cast frames to int and drop nans
    for event_type in event_dict.keys():
        event_dict[event_type] = [int(frame) for frame in event_dict[event_type] if pd.notnull(frame)]

    # set the event_dict as an attribute of the session:
    return event_dict

def build_event_lists(trials,sync_data,):
    hit_trials = trials[
        (trials.trial_type=='go')
        &(trials.response==1)
        &(trials.auto_rewarded!=True)
    ]
    
    auto_rewarded_trials = trials[
        trials.auto_rewarded==True
    ]
    
    miss_trials = trials[
        (trials.trial_type=='go')
        &(trials.response==0)
        &(trials.auto_rewarded!=True)
    ]
    
    catch_trials = trials[
        (trials.trial_type=='catch')
        &(trials.auto_rewarded!=True)
    ]
    
    F_events={}

    #get change frames on hit trials
    F_events['hit_frames'] = hit_trials['change_frame'].map(lambda x: helper_functions.find_nearest_fluorescence_frame(x,sync_data))
    F_events['miss_frames'] = miss_trials['change_frame'].map(lambda x: helper_functions.find_nearest_fluorescence_frame(x,sync_data))
    F_events['autoreward_frames'] = auto_rewarded_trials['change_frame'].map(lambda x: helper_functions.find_nearest_fluorescence_frame(x,sync_data))
    
    F_events['first_hit_licks'] = hit_trials[
        hit_trials['number_of_licks']>0
    ]['lick_frames'].map(lambda x: helper_functions.find_nearest_fluorescence_frame(x[0],sync_data))
    
    F_events['fa_frames'] = catch_trials[
        catch_trials['response']==1
    ]['change_frame'].map(lambda x: helper_functions.find_nearest_fluorescence_frame(x,sync_data))
    
    F_events['cr_frames'] = catch_trials[
        catch_trials['response']==0
    ]['change_frame'].map(lambda x: helper_functions.find_nearest_fluorescence_frame(x,sync_data))
    
    F_events['first_catch_lick_frames'] = catch_trials[
        catch_trials['number_of_licks']>0
    ]['lick_frames'].map(lambda x: helper_functions.find_nearest_fluorescence_frame(x[0],sync_data))
    

    for event_type in F_events.keys():
        if len(F_events[event_type])> 0:
            F_events[event_type]=F_events[event_type][~np.isnan(F_events[event_type])]
        else:
            F_events[event_type] = []
    
    return F_events

def nearest_reward(lick_time,rewards):
    return abs(lick_time - rewards['time']).min()

def nearest_lick(t,licks):
    return abs(t - licks['time']).min()

def which_pkl(filename_dict):
    '''
    I saved multiple processed PKL files back to these directories :(
    The processed PKL files all have 'trial' or 'cachecd' somewhere in the filename
    Only return a file that does not have 'trial' or 'cached' in the filename
    '''
    if len(filename_dict['behavior_pkl']) == 1:
        return filename_dict['behavior_pkl'][0]
    else:
        for fn in filename_dict['behavior_pkl']:
            if 'trial' not in fn and 'cached' not in fn and 'sanity_check' not in fn:
                return fn


def get_gender(mouse_id):
    '''get mouse gender from internal labtracks database'''
    mouse_info = vbu.query_labtracks(mouse_id)
    return mouse_info['sex']


def get_genotype_shorthand(genotype):
    '''
    assigns genotype shorthand
    '''
    if 'Ai94' in genotype:
        return 'Gnb4-CreERT2;Ai94'
    elif 'Ai93' in genotype:
        return 'Gnb4-CreERT2;Ai93'
    elif 'Ai148' in genotype:
        return 'Gnb4-CreERT2;Ai148'
    elif 'AAV' in genotype:
        return 'Gnb4-Cre;AAB-GCaMPs'
    else:
        return 'unknown'


def save_cell_images(session, savepath, force_overwrite=True):
    '''
    each cell image is a 3-channel array, but all three channels are identical
    therefore, only save the 0-indexed channel to avoid redundancy on disk
    '''
    for cell_id in session.cell_images.keys():
        filename = os.path.join(savepath, '{}.npy'.format(cell_id))
        if not os.path.exists(filename) and not force_overwrite:
            np.save(
                filename,
                session.cell_images[cell_id][0] # the array is in a single-element list
            )

def get_F_frame(behavior_frame,sync_data,threshold=0.05):
    if behavior_frame < len(sync_data['behavior_vsync']):
        behavior_time = sync_data['behavior_vsync'][behavior_frame]
    else:
        return None
    F_frame = helper_functions.find_nearest_index(behavior_time, sync_data['fluorescence_camera'])
    
    #note: this deals with events that come during times when the camera wasn't recording
    if abs(sync_data['fluorescence_camera'][F_frame]-behavior_time) < threshold:
        return F_frame
    else:
        return None

def find_nearest_fluorescence_frame(behavior_frame,sync_data,temporal_downsample_factor=1.0,largest_allowable_difference=1./20,display_lag=0.035):
    behavior_event_time = sync_data['behavior_vsync'][int(behavior_frame)]+display_lag
    F_frame_times = sync_data['fluorescence_camera']
    nearest_F_frame = find_nearest_index(behavior_event_time, F_frame_times)/temporal_downsample_factor
    
    if abs(behavior_event_time-F_frame_times[int(nearest_F_frame)]) < largest_allowable_difference:
        return nearest_F_frame
    else:
        return np.nan