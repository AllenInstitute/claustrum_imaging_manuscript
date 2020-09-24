# a collection of functions in support of passive stimulus analysis

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

def get_responses(session,cell_id,stim_condition,frame_before=100,frame_after=100):
    '''
    Uses the `event_triggered_average` function to get responses relative to a given event
    '''
    passive_stim_trials = session.behavior_core_data['trials']
    if type(stim_condition) == str:
        df = passive_stim_trials[passive_stim_trials['stim_type']==stim_condition]
    elif type(stim_condition) == list:
        df = passive_stim_trials[passive_stim_trials['stim_type'].isin(stim_condition)]
    
    dat = tbdu.event_triggered_average(
            session.filtered_traces[cell_id].values,
            events=df['nearest_F_frame'],
            frame_before=frame_before,
            frame_after=frame_after,
            sampling_rate=20, #Hz
            norm_frames=50,
            output='f',
            progressbar=False
        )
    
    return dat

def get_responsiveness_data(session,window_size=1):
    '''
    build a dataframe of integrals of activity in a pre/post stimulus window for every stimulus
    '''

    responsiveness_data = []
    
    pre_stim_indices = None
    post_stim_indices = None

    for row,cell_id in enumerate(session.filtered_cell_IDs):
        for col,stim_condition in enumerate(['visual','auditory','auditory_and_visual']):

            dat = get_responses(session,cell_id,stim_condition)

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
    build a dataframe of containing summary data of integrals of activity in a pre/post stimulus window for every cell
    '''
    responsiveness_summary = []
    responsiveness_data = session.responsiveness_data
    for row,cell_id in enumerate(session.filtered_cell_IDs):
        cell_data = responsiveness_data[
            (responsiveness_data['cell_id']==cell_id)
        ]
        for col,stim_condition in enumerate(['visual','auditory','auditory_and_visual']):

            res_ints = stats.ttest_ind(
                cell_data['pre_stim_integral'],
                cell_data.query('stim_condition == @stim_condition')['post_stim_integral']
            )
            
            res_means = stats.ttest_ind(
                cell_data['pre_stim_mean'],
                cell_data.query('stim_condition == @stim_condition')['post_stim_mean']
            )
            
            pre_stim_integral_mean = cell_data['pre_stim_integral'].mean()
            post_stim_integral_mean = cell_data.query('stim_condition == @stim_condition')['post_stim_integral'].mean()
            
            pre_stim_mean_all = cell_data['pre_stim_mean'].mean()
            post_stim_mean_all = cell_data.query('stim_condition == @stim_condition')['post_stim_mean'].mean()
            
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

    
def get_cells_with_significant_responses(session, significance_level=0.05, metric='p-value_on_integrals'):
    '''
    get cells from the `responsiveness_summary` dataframe that had a significant response to any stimulus condition
    '''
    rs = session.responsiveness_summary
    significant_responses = rs[rs[metric]<=significance_level].sort_values(by=metric)
    return significant_responses.cell_id.unique()

## plotting functions

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
        
def plot_cell_responses(session,cell_id,metric='p-value_on_integrals'):
    rs = session.responsiveness_summary
    fig,ax=plt.subplots(1,3,figsize=(12,5),sharey=True,sharex=True)
    for col,condition in enumerate(['visual','auditory','auditory_and_visual']):
        row = rs.query("cell_id == '{}' and condition == '{}'".format(cell_id,condition)).iloc[0]
        dat = get_responses(session,cell_id,condition)
        vbp.plot_event_triggered_timeseries(dat,ax=ax[col])
        ax[col].set_title('Cell {}, stimulus = {}\np = {:0.4f}, direction = {}'.format(
            row['cell_id'], 
            row['condition'], 
            row[metric], 
            get_sign(row['direction_of_effect_on_integrals'])
        ))
        ax[col].axvline(0,color='k',zorder=-1,linewidth=3,alpha=0.25)
        ax[col].set_xlabel('time from stimulus (s)')
        ax[col].set_xlim(dat['t'].min(),dat['t'].max())
    ax[0].set_ylabel('z-scored activity')
    fig.tight_layout()
    return fig,ax
        
def plot_significance_vs_window_size(mouse_id,cell_id,session):
    '''
    recalculate significance as a function of window size. Plot results
    '''
    dat_all = get_responses(session[mouse_id],cell_id,['visual','auditory','auditory_and_visual'])
    dat_condition = get_responses(session[mouse_id],cell_id,'auditory_and_visual')

    window_sizes = np.arange(0,5,0.01)
    p_vals = np.empty(len(window_sizes))
    for ii,window_size in enumerate(window_sizes):
        pre_stim_indices = np.where(np.logical_and(dat_all['t']>=-window_size,dat_all['t']<0))
        post_stim_indices = np.where(np.logical_and(dat_all['t']>0,dat_all['t']<=window_size))

        pre_stim_integrals = np.empty(len(dat_all['all_traces']))
        post_stim_integrals = np.empty(len(dat_condition['all_traces']))

        for idx in range(len(dat_all['all_traces'])):
            pre_stim_integrals[idx] = np.trapz(dat_all['all_traces'][idx][pre_stim_indices])

        for idx in range(len(dat_condition['all_traces'])):
            post_stim_integrals[idx] = np.trapz(dat_condition['all_traces'][idx][post_stim_indices])

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