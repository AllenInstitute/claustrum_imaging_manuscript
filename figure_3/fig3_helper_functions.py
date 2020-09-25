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

from visual_behavior_research.projects.tbd.session import Session

import visual_behavior_research.projects.tbd.utilities as tbdu
from visual_behavior_research.projects.tbd.Inscopix.series import SeriesCellSet
import visual_behavior_research.plotting as vbp
import visual_behavior_research.plotting.dro_plots as pf
import visual_behavior_research.utilities as vbu

from visual_behavior_research.projects.tbd.identify_repeat_rois import Repeat_ROI_Identifier

import isx
from visual_behavior_research.projects.tbd.Inscopix.pipeline import make_cell_map,plot_cell_map

from visual_behavior.translator.foraging import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe

def find_nearest_fluorescence_frame(behavior_frame,sync_data,temporal_downsample_factor=1.0,largest_allowable_difference=1./20,display_lag=0.035):
    behavior_event_time = sync_data['behavior_vsync'][int(behavior_frame)]+display_lag
    F_frame_times = sync_data['fluorescence_camera']
    nearest_F_frame = tbdu.find_nearest_index(behavior_event_time,F_frame_times)/temporal_downsample_factor
    
    if abs(behavior_event_time-F_frame_times[int(nearest_F_frame)]) < largest_allowable_difference:
        return nearest_F_frame
    else:
        return np.nan

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
    
    ## Note: Return to this. Why categorize catch trials this way?
    catch_trials = trials[
        (trials.trial_type=='catch')
        &(trials.auto_rewarded!=True)
    ]
    
    F_events={}

    #get change frames on hit trials
    F_events['hit_frames'] = hit_trials['change_frame'].map(lambda x:find_nearest_fluorescence_frame(x,sync_data))
    F_events['miss_frames'] = miss_trials['change_frame'].map(lambda x:find_nearest_fluorescence_frame(x,sync_data))
    F_events['autoreward_frames'] = auto_rewarded_trials['change_frame'].map(lambda x:find_nearest_fluorescence_frame(x,sync_data))
    
    F_events['first_hit_licks'] = hit_trials[
        hit_trials['number_of_licks']>0
    ]['lick_frames'].map(lambda x:find_nearest_fluorescence_frame(x[0],sync_data))
    
    F_events['fa_frames'] = catch_trials[
        catch_trials['response']==1
    ]['change_frame'].map(lambda x:find_nearest_fluorescence_frame(x,sync_data))
    
    F_events['cr_frames'] = catch_trials[
        catch_trials['response']==0
    ]['change_frame'].map(lambda x:find_nearest_fluorescence_frame(x,sync_data))
    
    F_events['first_catch_lick_frames'] = catch_trials[
        catch_trials['number_of_licks']>0
    ]['lick_frames'].map(lambda x:find_nearest_fluorescence_frame(x[0],sync_data))
    

    for event_type in F_events.keys():
        if len(F_events[event_type])> 0:
            F_events[event_type]=F_events[event_type][~np.isnan(F_events[event_type])]
        else:
            F_events[event_type] = []
    
    return F_events

def which_pkl(filename_dict):
    '''
    I saved multiple processed PKL files back to these directories :(
    The processed PKL files all have 'trial' or 'cachecd' somewhere in the filename
    Only return a file that does not have 'trial' or 'cached' in the filename
    This feels like an awful way of dealing with this problem, but I don't see another obvious solution
    '''
    if len(filename_dict['behavior_pkl']) == 1:
        return filename_dict['behavior_pkl'][0]
    else:
        for fn in filename_dict['behavior_pkl']:
            if 'trial' not in fn and 'cached' not in fn and 'sanity_check' not in fn:
                return fn


def get_F_frame(behavior_frame,sync_data,threshold=0.05):
    if behavior_frame < len(sync_data['behavior_vsync']):
        behavior_time = sync_data['behavior_vsync'][behavior_frame]
    else:
        return None
    F_frame = tbdu.find_nearest_index(behavior_time,sync_data['fluorescence_camera'])
    
    #note: this deals with events that come during times when the camera wasn't recording
    if abs(sync_data['fluorescence_camera'][F_frame]-behavior_time) < threshold:
        return F_frame
    else:
        return None
    
def nearest_reward(lick_time,rewards):
    return abs(lick_time - rewards['time']).min()

def nearest_lick(t,licks):
    return abs(t - licks['time']).min()

def get_responses(session,cell_id,events,frame_before=100,frame_after=100):
    '''
    Uses the `event_triggered_average` function to get responses relative to a given event
    '''
    
    dat = tbdu.event_triggered_average(
            session.filtered_traces[cell_id].values,
            events=events,
            frame_before=frame_before,
            frame_after=frame_after,
            sampling_rate=20, #Hz
            norm_frames=50,
            output='f',
            progressbar=False
        )

    
    return dat

def build_average_response_df(data_dict, foldername,frame_before=150,frame_after=150):
    session = data_dict['session']
    
    dfs = {
        'hit':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
        'miss':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
        'fa':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
        'cr':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
    }
    for row,cell_id in enumerate(session.filtered_cell_IDs):
        
        for col,condition in enumerate(['hit','miss','fa','cr']):

            events = data_dict['{}_events'.format(condition)]

            dat = get_responses(session,cell_id,events)
            

def get_responsiveness_data(data_dict, foldername, window_size=1):
    '''
    build a dataframe of integrals of activity in a pre/post stimulus window for every stimulus
    '''

    responsiveness_data = []
    
    pre_stim_indices = None
    post_stim_indices = None
    
    session = data_dict['session']

    for row,cell_id in enumerate(session.filtered_cell_IDs):
        for col,condition in enumerate(['hit','miss','fa','cr']):

            events = data_dict['{}_events'.format(condition)]

            dat = get_responses(session,cell_id,events)

            if not pre_stim_indices:
                pre_stim_indices = np.where(np.logical_and(dat['t']>=-window_size,dat['t']<0))
                post_stim_indices = np.where(np.logical_and(dat['t']>0,dat['t']<=window_size))

            for idx in range(len(dat['all_traces'])):

                responsiveness_data.append({
                    'cell_id':cell_id,
                    'foldername':foldername,
                    'presentation_number':idx,
                    'condition':condition,
                    'pre_stim_integral':np.trapz(dat['all_traces'][idx][pre_stim_indices]),
                    'post_stim_integral':np.trapz(dat['all_traces'][idx][post_stim_indices]),
                    'pre_stim_mean':np.mean(dat['all_traces'][idx][pre_stim_indices]),
                    'post_stim_mean':np.mean(dat['all_traces'][idx][post_stim_indices]),
                })

    cols_to_return = [
        'cell_id',
        'foldername',
        'condition',
        'presentation_number',
        'pre_stim_integral',
        'post_stim_integral',
        'pre_stim_mean',
        'post_stim_mean'
    ]
    return pd.DataFrame(responsiveness_data)[cols_to_return]

def build_responsiveness_summary(responsiveness_data,window_size=1):
    '''
    build a dataframe of containing summary data of integrals of activity in a pre/post stimulus window for every cell
    '''
    responsiveness_summary = []
    foldername = responsiveness_data['foldername'].iloc[0]

    for row,cell_id in enumerate(responsiveness_data['cell_id'].unique()):
        cell_data = responsiveness_data[
            (responsiveness_data['cell_id']==cell_id)
        ]
        for col,condition in enumerate(['hit','miss','fa','cr']):

            res_ints = stats.ttest_ind(
                cell_data['pre_stim_integral'].dropna(),
                cell_data.query('condition == @condition')['post_stim_integral'].dropna()
            )
            
            res_means = stats.ttest_ind(
                cell_data['pre_stim_mean'].dropna(),
                cell_data.query('condition == @condition')['post_stim_mean'].dropna()
            )
            
            pre_stim_integral_mean = cell_data.query('condition == @condition')['pre_stim_integral'].mean()
            post_stim_integral_mean = cell_data.query('condition == @condition')['post_stim_integral'].mean()
            
            pre_stim_mean_all = cell_data.query('condition == @condition')['pre_stim_mean'].mean()
            post_stim_mean_all = cell_data.query('condition == @condition')['post_stim_mean'].mean()
            
            responsiveness_summary.append({
                'cell_id':cell_id,
                'foldername':foldername,
                'condition':condition,
                'number_of_trials':len(cell_data.query('condition == @condition')),
                'pre_stim_integral_mean':pre_stim_integral_mean,
                'post_stim_integral_mean':post_stim_integral_mean,
                'pre_stim_mean_all':pre_stim_mean_all,
                'post_stim_mean_all':post_stim_mean_all,
                'p-value_on_integrals':res_ints.pvalue,
                'statistic_on_integrals':res_ints.statistic,
                'direction_of_effect_on_integrals': np.sign(post_stim_integral_mean - pre_stim_integral_mean),
                'p-value_on_means':res_means.pvalue,
                'statistic_on_means':res_means.statistic,
                'direction_of_effect_on_means': np.sign(post_stim_mean_all - pre_stim_mean_all),
            })

    cols_to_return = [
        'cell_id', 
        'foldername',
        'condition', 
        'number_of_trials', 
        'pre_stim_integral_mean', 
        'post_stim_integral_mean', 
        'p-value_on_integrals', 
        'direction_of_effect_on_integrals',
        'pre_stim_mean_all',
        'post_stim_mean_all',
        'p-value_on_means', 
        'direction_of_effect_on_means',
    ]
    return pd.DataFrame(responsiveness_summary)[cols_to_return]


def plot_examples(to_plot, data, ax, frame_before=200, frame_after=200):
    
    colors = {
        'hit':'darkgreen',
        'fa':'firebrick',
        'cr':'DarkSlateBlue',
        'miss':'DarkOrange'
    }

    row=-1
    for idx, entry in to_plot.iterrows():
        row += 1
        foldername = entry['foldername']
        cell_id = entry['cell_id']

        for col,event in enumerate(['hit','fa','miss','cr']):
            data_dict = data[foldername]
            session = data_dict['session']
            events = data_dict['{}_events'.format(event)]
            dat = get_responses(session,cell_id,events,frame_before=frame_before,frame_after=frame_after)

            pf.plot_event_triggered_timeseries(dat,foreground_color=colors[event],ax=ax[row][col])
            ax[row][col].axvline(0,color='k',alpha=0.5,zorder=-np.inf,linewidth=3)
            ax[row][col].set_ylim(-5,15.5)

            ax[row][col].axis('off')

    for col,event_type in enumerate(['Hit\nTrials','False Alarm\nTrials','Miss\nTrials','Correct Rejection\nTrials']):
        ax[0][col].set_title(event_type,rotation=0,ha='center')

    ax[3][0].plot((0,0),(-4.5,0.5),color='k',linewidth=2)
    ax[3][0].plot((0,0+5),(-4.5,-4.5),color='k',linewidth=2)
    ax[3][0].text(1,-7.5,'2 s')
    ax[3][0].text(-1,-3,'5 SD',ha='right')

    sns.despine()
    plt.subplots_adjust(wspace=0.05)
    