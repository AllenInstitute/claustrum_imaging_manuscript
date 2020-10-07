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

def load_session(data_path,load_cached_traces_table=True):
    '''
    load session, add a column denoting the closest frame in the Inscopix video
    '''
    session = Session(data_path, load_cached_traces_table=load_cached_traces_table)

    passive_stim_trials = session.behavior_core_data['trials']
    passive_stim_trials['nearest_F_frame'] = passive_stim_trials['startframe'].map(
            lambda x:find_nearest_index(session.sync_data['fluorescence_camera'],session.sync_data['behavior_vsync'][x])
        )

    return session

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

def find_nearest_index(array,value):
    '''
    a convenience function the find the nearest index to a given value in an array
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx-1
    else:
        return idx
    
def find_nearest_value(array,value):
    '''
    a convenience function the find the nearest value to a given value in an array
    '''
    return array[find_nearest_index(array,value)]

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

def get_responses(session,cell_id,events,frame_before=100,frame_after=100, ):
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
            
def get_responsiveness_data(session, window_size=1, behavior_condition='active'):
    '''
    build a dataframe of integrals and means of activity in a pre/post stimulus window for every stimulus
    '''

    responsiveness_data = []
    
    pre_stim_indices = None
    post_stim_indices = None

    if behavior_condition == 'active':
        conditions = ['hit','miss','fa','cr']
    elif behavior_condition == 'passive':
        conditions = ['visual','auditory','auditory_and_visual']

    for row,cell_id in enumerate(session.filtered_cell_IDs):
        for col,stim_condition in enumerate(conditions):

            if behavior_condition == 'passive':
                passive_stim_trials = session.behavior_core_data['trials']
                df = passive_stim_trials[passive_stim_trials['stim_type']==stim_condition]
                events = df['nearest_F_frame']
            elif behavior_condition == 'active':
                events = session.event_dict['{}_events'.format(stim_condition)]

            dat = get_responses(session, cell_id, events)

            if not pre_stim_indices:
                pre_stim_indices = np.where(np.logical_and(dat['t']>=-window_size,dat['t']<0))
                post_stim_indices = np.where(np.logical_and(dat['t']>0,dat['t']<=window_size))

            for idx in range(len(dat['all_traces'])):

                responsiveness_data.append({
                    'cell_id':cell_id,
                    'presentation_number':idx,
                    'stim_condition':stim_condition,
                    'pre_stim_integral':np.trapz(dat['all_traces'][idx][pre_stim_indices]),
                    'post_stim_integral':np.trapz(dat['all_traces'][idx][post_stim_indices]),
                    'pre_stim_mean':np.mean(dat['all_traces'][idx][pre_stim_indices]),
                    'post_stim_mean':np.mean(dat['all_traces'][idx][post_stim_indices]),
                })

    cols_to_return = [
        'cell_id',
        'stim_condition',
        'presentation_number',
        'pre_stim_integral',
        'post_stim_integral',
        'pre_stim_mean',
        'post_stim_mean'
    ]
    return pd.DataFrame(responsiveness_data)[cols_to_return]

def build_responsiveness_summary(session,window_size=1):
    '''
    build a dataframe of containing summary data of integrals and means of activity in a pre/post stimulus window for every cell
    '''
    responsiveness_summary = []
    responsiveness_data = session.responsiveness_data
    stim_conditions = responsiveness_data['stim_condition'].unique()
    for row,cell_id in enumerate(session.filtered_cell_IDs):
        cell_data = responsiveness_data[
            (responsiveness_data['cell_id']==cell_id)
        ]
        for col,stim_condition in enumerate(stim_conditions):

            res_ints = stats.ttest_ind(
                cell_data['pre_stim_integral'].dropna(),
                cell_data.query('stim_condition == @stim_condition')['post_stim_integral'].dropna()
            )
            
            res_means = stats.ttest_ind(
                cell_data['pre_stim_mean'].dropna(),
                cell_data.query('stim_condition == @stim_condition')['post_stim_mean'].dropna()
            )
            
            pre_stim_integral_mean = cell_data['pre_stim_integral'].dropna().mean()
            post_stim_integral_mean = cell_data.query('stim_condition == @stim_condition')['post_stim_integral'].dropna().mean()
            
            pre_stim_mean_all = cell_data['pre_stim_mean'].dropna().mean()
            post_stim_mean_all = cell_data.query('stim_condition == @stim_condition')['post_stim_mean'].dropna().mean()
            
            responsiveness_summary.append({
                'cell_id':cell_id,
                'condition':stim_condition,
                'number_of_trials':len(cell_data.query('stim_condition == @stim_condition')),
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


def get_cells_with_significant_responses(session, significance_level=0.05, metric='p-value_on_means'):
    '''
    get cells from the `responsiveness_summary` dataframe that had a significant response to any stimulus condition
    '''
    rs = session.responsiveness_summary
    significant_responses = rs[rs[metric]<=significance_level].sort_values(by=metric)
    return significant_responses.cell_id.unique()


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

        tdf = []
        for col,event in enumerate(['hit','fa','miss','cr']):
            data_dict = data[foldername]
            session = data_dict['session']
            events = session.event_dict['{}_events'.format(event)] #data_dict['{}_events'.format(event)]
            dat = get_responses(session,cell_id,events,frame_before=frame_before,frame_after=frame_after)
            for ii in range(np.shape(dat['all_traces'])[0]):
                df = pd.DataFrame({'t':dat['t'],'z_scored_activity':dat['all_traces'][ii]})
                df['condition'] = event
                df['repeat_number'] = ii
                df['cell_id'] = cell_id
                df['experiment'] = foldername
                tdf.append(df)
        tdf = pd.concat(tdf)

        for col,event in enumerate(['hit','fa','miss','cr']):
            # pf.plot_event_triggered_timeseries(dat,foreground_color=colors[event],ax=ax[row][col])
            sns.lineplot(
                data = tdf.query('condition == "{}"'.format(event)),
                x='t',
                y='z_scored_activity',
                ax = ax[row][col],
                color=colors[event],
            )
            ax[row][col].axvline(0,color='k',alpha=0.5,zorder=-np.inf,linewidth=3)
            ax[row][col].set_ylim(-2.5,5)
            ax[row][col].set_xlim(-2,6)
            ax[row][col].axis('off')

    for col,event_type in enumerate(['Hit\nTrials','False Alarm\nTrials','Miss\nTrials','Correct Rejection\nTrials']):
        ax[0][col].set_title(event_type,rotation=0,ha='center')

    ax[row][0].plot((-1.9,-1.9),(-1.4,4.6),color='k',linewidth=2)
    ax[row][0].plot((-1.9,-1.9+2),(-1.4,-1.4),color='k',linewidth=2)
    ax[row][0].text(-1,-2.25,'2 s',ha='center',va='top')
    ax[row][0].text(-2.25,1,'5 SD',ha='right')

    sns.despine()
    plt.subplots_adjust(wspace=0.05)
    
def plot_cell(session,cell,timeseries_type='filtered'):
    '''
    plot cell image and timeseries for a given cell in a session
    '''
    fig = plt.figure(figsize=(15,3))

    ax = [
        plt.subplot2grid((1, 3), (0, 0)),
        plt.subplot2grid((1, 3), (0, 1), colspan=2)
    ]

    ax[0].imshow(session.series_cs.cell_images[cell][0],cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('cell image')
    
    if timeseries_type == 'filtered':
        t = session.filtered_traces['t']
        y = session.filtered_traces[cell]
        ylabel = 'z-scored fluorescence data'
    elif timeseries_type == 'raw':
        t = session.traces['t']
        y = session.traces[cell]
        ylabel = 'raw fluorescence (a.u.)'
        
    ax[1].plot(t,y,color='k')
    ax[1].set_title('Time series for cell {}, SNR = {:0.2f}, Peak Height (95th percentile) = {:0.2f}'.format(
        cell,
        session.metrics.loc[cell]['snr'],
        session.metrics.loc[cell]['peak_95th_percentile'],
    ))
    ax[1].set_xlabel('recording time (s)')
    ax[1].set_ylabel(ylabel)
#     ax[1].set_ylim(-5,1.1*y.max())
    ax[1].set_xlim(t.min(),t.max())
    fig.tight_layout()


def plot_cells_with_significant_responses(session, significance_level=0.05):
    '''
    makes a plot of a cells response to each condition
    '''
    cells_with_significant_responses = get_cells_with_significant_responses(session, significance_level=significance_level)
    for cell_id in cells_with_significant_responses:
        plot_cell_responses(session,cell_id)
        
def get_sign(v):
    if v>0:
        return 'positive'
    elif v<0:
        return 'negative'
        
def plot_cell_responses(session,cell_id,metric='p-value_on_means'):
    rs = session.responsiveness_summary
    fig,ax=plt.subplots(1,3,figsize=(12,5),sharey=True,sharex=True)
    for col, stim_condition in enumerate(['visual','auditory','auditory_and_visual']):
        row = rs.query("cell_id == '{}' and condition == '{}'".format(cell_id, stim_condition)).iloc[0]
        passive_stim_trials = session.behavior_core_data['trials']
        if type(stim_condition) == str:
            df = passive_stim_trials[passive_stim_trials['stim_type'] == stim_condition]
        elif type(stim_condition) == list:
            df = passive_stim_trials[passive_stim_trials['stim_type'].isin(stim_condition)]

        events = df['nearest_F_frame']
        dat = get_responses(session,cell_id,events)
        vbp.plot_event_triggered_timeseries(dat,ax=ax[col])
        ax[col].set_title('Cell {}, stimulus = {}\np = {:0.4f}, direction = {}'.format(
            row['cell_id'], 
            row['condition'], 
            row[metric], 
            get_sign(row['direction_of_effect_on_means'])
        ))
        ax[col].axvline(0,color='k',zorder=-1,linewidth=3,alpha=0.25)
        ax[col].set_xlabel('time from stimulus (s)')
        ax[col].set_xlim(dat['t'].min(),dat['t'].max())
    ax[0].set_ylabel('z-scored activity')
    fig.tight_layout()
    return fig,ax
        
def plot_significance_vs_window_size(mouse_id,cell_id,session,metric='mean'):
    '''
    recalculate significance as a function of window size. Plot results
    '''
    passive_stim_trials = session[mouse_id].behavior_core_data['trials']

    dat_all = get_responses(
        session[mouse_id],
        cell_id,
        passive_stim_trials.query("stim_type in ['visual','auditory','auditory_and_visual']")['nearest_F_frame']
    )
    dat_condition = get_responses(
        session[mouse_id],
        cell_id,
        passive_stim_trials.query("stim_type == 'auditory_and_visual'")['nearest_F_frame']
    )

    window_sizes = np.arange(0,5,0.01)
    p_vals = np.empty(len(window_sizes))
    for ii,window_size in enumerate(window_sizes):
        pre_stim_indices = np.where(np.logical_and(dat_all['t']>=-window_size,dat_all['t']<0))
        post_stim_indices = np.where(np.logical_and(dat_all['t']>0,dat_all['t']<=window_size))

        pre_stim_integrals = np.empty(len(dat_all['all_traces']))
        post_stim_integrals = np.empty(len(dat_condition['all_traces']))
        pre_stim_means = np.empty(len(dat_all['all_traces']))
        post_stim_means = np.empty(len(dat_condition['all_traces']))

        for idx in range(len(dat_all['all_traces'])):
            pre_stim_integrals[idx] = np.trapz(dat_all['all_traces'][idx][pre_stim_indices])
            pre_stim_means[idx] = np.mean(dat_all['all_traces'][idx][pre_stim_indices])

        for idx in range(len(dat_condition['all_traces'])):
            post_stim_integrals[idx] = np.trapz(dat_condition['all_traces'][idx][post_stim_indices])
            post_stim_means[idx] = np.mean(dat_condition['all_traces'][idx][post_stim_indices])

        if metric == 'mean':
            res = stats.ttest_ind(pre_stim_means,post_stim_means)
        elif metric == 'integral':
            res = stats.ttest_ind(pre_stim_integrals,post_stim_integrals)
        p_vals[ii] = res.pvalue

    fig,ax=plt.subplots(figsize=(15,3))
    ax.plot(window_sizes,p_vals)
    ax.plot(window_sizes[p_vals<=0.05],p_vals[p_vals<=0.05],'.r')
    ax.set_xlabel('window size used to calculate responsiveness')
    ax.set_ylabel('p-value of responsiveness measure')
    ax.set_title('significance vs. window size for cell {}\n(points below 0.05 in red)'.format(cell_id))
    fig.tight_layout()
    return fig,ax


def single_cell_heatmap(session, cell_id, events, ax=None, cbar=True, title=''):
    '''
    make a heatmap for a single cell where each line is a trial
    '''
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(10,4),sharex=True,sharey=True)


    dat = get_responses(
        session,
        cell_id,
        events
    )
    sns.heatmap(
        dat['all_traces'],
        ax=ax,
        vmin=-5,
        vmax=5,
        cbar=cbar,
        cmap='seismic',
        cbar_kws = {'label':'z-scored activity'},
    )
    ax.set_title(title)
    ax.set_xticklabels([dat['t'][int(i)] for i in ax.get_xticks()]);
    ax.axvline(len(dat['t'])/2,color='white')