<<<<<<< HEAD
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

class Session(object):
    '''
    reconstructs a session object from saved data in github repo
    A session object contains data from a single recording session as attributes
    only necessary data for analyses are stored to github
    '''
    def __init__(self, mouse, session_type):
        self.data_path = os.path.join(os.path.split(os.getcwd())[0],'data')
        self.session_type_folder = os.path.join(self.data_path,'{}_sessions'.format(session_type))
        self.mouse_folder = os.path.join(self.session_type_folder, mouse)
        self.cell_folder = os.path.join(self.mouse_folder, 'cell_images')
        self.session_type = session_type
        self.mouse_id = mouse
        
        self.trials = pd.read_csv(os.path.join(self.mouse_folder, 'trials.csv'))
        if session_type == 'active':
            self.licks = pd.read_csv(os.path.join(self.mouse_folder, 'licks.csv'))
            self.rewards = pd.read_csv(os.path.join(self.mouse_folder, 'rewards.csv'))
            self.visual_stimuli = pd.read_csv(os.path.join(self.mouse_folder, 'visual_stimuli.csv'))
            self.event_dict = load_json(os.path.join(self.mouse_folder, 'events.json'))
            
        self.filtered_cell_ids = pd.read_csv(os.path.join(self.mouse_folder, 'filtered_cell_ids.csv'))['cell_id'].values
        self.filtered_cell_count = len(self.filtered_cell_ids)
        
        self.traces = pd.read_csv(os.path.join(self.mouse_folder, 'traces.csv'))
        self.filtered_traces = pd.read_csv(os.path.join(self.mouse_folder, 'filtered_traces.csv'))
        self.metrics = pd.read_csv(os.path.join(self.mouse_folder, 'metrics.csv')).set_index('cellName')
        self.cell_images = load_cell_images(os.path.join(self.mouse_folder, 'cell_images'))
        

def load_session(mouse_id, session_type):
    '''
    load a session as a Session object
    '''
    return Session(mouse_id, session_type)

def load_passive_session_from_disk(data_path,load_cached_traces_table=True):
    '''
    load passive stimulus session, add a column denoting the closest frame in the Inscopix video
    this depends on allen institute internal packages/databases
    '''
    session = tbd_Session(data_path, load_cached_traces_table=load_cached_traces_table)

    passive_stim_trials = session.behavior_core_data['trials']
    passive_stim_trials['nearest_F_frame'] = passive_stim_trials['startframe'].map(
            lambda x:find_nearest_index(session.sync_data['fluorescence_camera'],session.sync_data['behavior_vsync'][x])
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
            
def load_cell_images(loadpath):
    '''
    load each cell image for a given session
    '''
    cell_images = {}
    for fn in os.listdir(loadpath):
        filename = os.path.join(loadpath, fn)
        cell_id = fn.split('.npy')[0]
        cell_images[cell_id] = np.load(filename)
    return cell_images
    
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
    
def get_gender(mouse_id):
    '''get mouse gender from internal labtracks database'''
    mouse_info = vbu.query_labtracks(mouse_id)
    return mouse_info['sex']

def make_folder(path):
    '''make a folder if it does not already exist'''
    if not os.path.exists(path):
        os.mkdir(path)
        
def save_json(dictionary, filename):
    with open(filename, "w") as outfile:  
        json.dump(dictionary, outfile)
        
def load_json(filename):
    with open(filename, "r") as infile:  
        dictionary = json.load(infile)
    return dictionary

def find_nearest_index(value, array):
    '''
    a convenience function the find the nearest index to a given value in an array
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return idx-1
    else:
        return idx
    
def find_nearest_value(value, array):
    '''
    a convenience function the find the nearest value to a given value in an array
    '''
    return array[find_nearest_index(value, array)]

def find_nearest_fluorescence_frame(behavior_frame,sync_data,temporal_downsample_factor=1.0,largest_allowable_difference=1./20,display_lag=0.035):
    behavior_event_time = sync_data['behavior_vsync'][int(behavior_frame)]+display_lag
    F_frame_times = sync_data['fluorescence_camera']
    nearest_F_frame = find_nearest_index(behavior_event_time,F_frame_times)/temporal_downsample_factor
    
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
    F_frame = find_nearest_index(behavior_time,sync_data['fluorescence_camera'])
    
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
    
    dat = event_triggered_average(
            session.filtered_traces[cell_id].values,
            events=events,
            frame_before=frame_before,
            frame_after=frame_after,
            sampling_rate=20, #Hz
            norm_frames=50,
            output='f',
        )

    
    return dat

def event_triggered_average(data, events, frame_before, frame_after, norm_frames=50, 
                            sampling_rate=100, output='F'):
    '''Extracts average activity in an ROI relative to event frames'''
    events = list(events)
    # preallocate an array for all of the timeseries
    all_traces = np.empty((len(events), int(frame_before+frame_after)))*np.NAN

    # create a time vector based on the requested frames and sampling rate
    t = np.linspace(-frame_before,
                    frame_after-1,
                    frame_before+frame_after)/sampling_rate

    # for each event, calculate the timeseries of the df values inside the mask
    for E in events:
        try:
            if np.isnan(E) == False:
                E = int(E)
                first_frame = int(E-frame_before)
                last_frame = int(E+frame_after)

                F = data[np.maximum(first_frame, 0):np.minimum(
                    last_frame, np.shape(data)[0])]

                # get F0, defined as the average of N frames before the event
                first_norm_frame = int(np.max((E-norm_frames, 0)))
                F0 = np.mean(data[first_norm_frame:E])

                # calculate df
                if output.lower() == 'df':
                    response = F - F0
                elif output.lower() == 'f':
                    response = F
                elif output.lower() == 'dff':
                    response = (F - F0)/F0

                masked_data = response

                # put the timeseries in the proper row of the preallocated array
                # Pad with nans if the range of requested frames goes beyond the bounds of the movie
                if first_frame < 0:
                    pad = np.empty(abs(E-frame_before))*np.nan
                    all_traces[events.index(E), :] = np.hstack(
                        (pad, masked_data.flatten()))
                elif last_frame > np.shape(data)[0]:
                    pad = np.empty(abs(last_frame-np.shape(data)[0]))*np.nan
                    all_traces[events.index(E), :] = np.hstack(
                        (masked_data.flatten(), pad))
                else:
                    all_traces[events.index(E), :] = masked_data[:np.shape(
                        all_traces)[1]].flatten()

        except Exception as e:
            print("failed on {}".format(E))
            print(e)
    return {'t': t,
            'trace_mean': np.nanmean(all_traces, axis=0),
            'trace_std': np.nanstd(all_traces, axis=0),
            'trace_median': np.median(all_traces, axis=0),
            'all_traces': all_traces}

def build_average_response_df(data_dict, foldername,frame_before=150,frame_after=150):
    session = data_dict['session']
    
    dfs = {
        'hit':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
        'miss':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
        'fa':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
        'cr':pd.DataFrame({'t':np.linspace(-frame_before*0.05,(frame_after-1)*0.05,frame_before+frame_after)}),
    }
    for row,cell_id in enumerate(session.filtered_cell_ids):
        
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

    for row,cell_id in enumerate(session.filtered_cell_ids):
        for col,stim_condition in enumerate(conditions):

            if behavior_condition == 'passive':
                passive_stim_trials = session.trials
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

def build_responsiveness_summary(session,window_size=1.5):
    '''
    build a dataframe of containing summary data of integrals and means of activity in a pre/post stimulus window for every cell
    '''
    responsiveness_summary = []
    responsiveness_data = session.responsiveness_data
    stim_conditions = responsiveness_data['stim_condition'].unique()
    for row,cell_id in enumerate(session.filtered_cell_ids):
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
                'cell_id': cell_id,
                'mouse_id': session.mouse_id,
                'condition': stim_condition,
                'number_of_trials': len(cell_data.query('stim_condition == @stim_condition')),
                'pre_stim_integral_mean': pre_stim_integral_mean,
                'post_stim_integral_mean': post_stim_integral_mean,
                'pre_stim_mean_all': pre_stim_mean_all,
                'post_stim_mean_all': post_stim_mean_all,
                'p-value_on_integrals': res_ints.pvalue,
                'statistic_on_integrals': res_ints.statistic,
                'direction_of_effect_on_integrals': np.sign(post_stim_integral_mean - pre_stim_integral_mean),
                'p-value_on_means': res_means.pvalue,
                'statistic_on_means': res_means.statistic,
                'direction_of_effect_on_means': np.sign(post_stim_mean_all - pre_stim_mean_all),
            })

    cols_to_return = [
        'cell_id', 
        'mouse_id',
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


def plot_examples(to_plot, session_dict, ax, frame_before=200, frame_after=200, xlim=(-2,6), ylim=(-2,3)):
    '''
    plot an array of example cells in all four active behavior conditions
    '''
    
    colors = {
        'hit':'darkgreen',
        'fa':'firebrick',
        'cr':'DarkSlateBlue',
        'miss':'DarkOrange'
    }

    row=-1
    for idx, entry in to_plot.iterrows():
        row += 1
        cell_id = entry['cell_id']
        mouse_id = entry['mouse_id']
        tdf = []
        session = session_dict[mouse_id]
        for col,event in enumerate(['hit','fa','miss','cr']):
            
            events = session.event_dict['{}_events'.format(event)] #data_dict['{}_events'.format(event)]
            dat = get_responses(session,cell_id,events,frame_before=frame_before,frame_after=frame_after)
            for ii in range(np.shape(dat['all_traces'])[0]):
                df = pd.DataFrame({'t':dat['t'],'z_scored_activity':dat['all_traces'][ii]})
                df['condition'] = event
                df['repeat_number'] = ii
                df['cell_id'] = cell_id
                df['mouse_id'] = mouse_id
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
            ax[row][col].set_ylim(ylim[0], ylim[1])
            ax[row][col].set_xlim(xlim[0], xlim[1])
            ax[row][col].axis('off')

    for col,event_type in enumerate(['Hit\nTrials','False Alarm\nTrials','Miss\nTrials','Correct Rejection\nTrials']):
        ax[0][col].set_title(event_type,rotation=0,ha='center')

    scalebar_x_corner = -1.9
    scalebar_y_corner = -0.9
    scalebar_height = 5
    scalebar_width = 2
    ax[row][0].plot(
        (scalebar_x_corner,scalebar_x_corner),
        (scalebar_y_corner,scalebar_y_corner+scalebar_height),
        color='k',
        linewidth=2
    )
    ax[row][0].plot(
        (scalebar_x_corner,scalebar_x_corner+scalebar_width),
        (scalebar_y_corner,scalebar_y_corner),
        color='k',
        linewidth=2
    )
    ax[row][0].text(-1,-1.25,'2 s',ha='center',va='top')
    ax[row][0].text(-1.25,1,'5 SD',ha='right')

    sns.despine()
    plt.subplots_adjust(wspace=0.05)
    
    
def plot_event_triggered_timeseries(data_dict, ax=None, background_color='gray', foreground_color='black', 
                                    background_alpha=0.25, foreground_linewidth=3, 
                                    ylabel="", xlabel="",title="", title_fontsize=18):
    '''
    assumes a data dictionary of the format output by utilities.event_triggered_average
    '''
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    for trace in data_dict['all_traces']:
        ax.plot(data_dict['t'], trace,
                color=background_color, alpha=background_alpha)
    ax.plot(data_dict['t'], data_dict['trace_mean'],
            color=foreground_color, alpha=1, linewidth=foreground_linewidth)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=title_fontsize)

    if 'fig' in dir():
        return fig, ax
    else:
        return ax
    
def plot_cell(session,cell,timeseries_type='filtered'):
    '''
    plot cell image and timeseries for a given cell in a session
    '''
    fig = plt.figure(figsize=(15,3))

    ax = [
        plt.subplot2grid((1, 3), (0, 0)),
        plt.subplot2grid((1, 3), (0, 1), colspan=2)
    ]

    ax[0].imshow(session.cell_images[cell],cmap='gray')
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
        passive_stim_trials = session.trials
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
    passive_stim_trials = session[mouse_id].trials

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
=======
version https://git-lfs.github.com/spec/v1
oid sha256:f57ec2c20051cd517edff249f181b613b6afabd36b78c11b8492c09e60677148
size 32694
>>>>>>> 9c72ba05f8b42e226ff09ae6e7e351ebd21bb98e
