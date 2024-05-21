import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy.sparse
from typing import Tuple
from numpy.typing import ArrayLike


def vector_sum(radii: ArrayLike, angles: ArrayLike) -> Tuple[float, float]:
    """Given an 1D array of vector lengths and an 1D array of corresponding radian angles,
    return the vectorial sum as a tuple (radius, radian angle).

    Parameters
    ----------
    radii : np.array
        An 1D array of vector lengths.
    angles : np.array
        An 1D array of angles (radian).
    Returns
    -------
    tuple(float, float)
        The radius and angle of the sum vector.
    Raises
    -------
    ValueError: if input parameters are invalid
    """
    if len(angles) != len(radii):  # passed arrays of different lenghts, or both empty
        raise ValueError(f"Shape of angles and radii do not match.")
    elif len(angles) == 0:
        raise ValueError(f"Empty input arrays detected.")
    x = np.sum(radii*np.cos(angles))  # sum up x components
    y = np.sum(radii*np.sin(angles))
    angle = np.arctan2(y, x)
    radius = np.sqrt(x**2 + y**2)
    return (radius, angle)


def event_numbers(data, threshold, max_distance):
    peaks, _ = find_peaks(data, height=threshold, distance=max_distance)
    return data.iloc[peaks]


def make_firing_rate_maps(data, num_rounds, num_units, num_bins):
    # Initialize a 3-dimensional array to store firing rate maps for each cell and round
    firing_rate_maps = np.zeros((int(num_units), num_rounds, num_bins))

    for cell in range(num_units):
        # FIXME: what if, for example, rounds 0, 1, 3, 4 are included, and round 2 was filtered out?
        # FIXME: can speed up if filtering for round happens first. Right now, same filtering for round is done for each cell
        for round_num in range(1, num_rounds + 1):
            # Filter data for the current round and cell
            round_data = data[data['Rounds'] == round_num]

            # Calculate histogram for the current cell and round with specified number of bins
            hist, _ = np.histogram(
                round_data['Distance'], bins=num_bins, weights=round_data[cell])

            # Calculate histogram of count of activations in each bin
            count_hist, _ = np.histogram(round_data['Distance'], bins=num_bins)

            # Calculate average firing rate map for the current cell and round
            avg_firing_rate_map = np.divide(
                hist, count_hist, out=np.zeros_like(hist), where=(count_hist != 0))
            avg_firing_rate_map[np.isnan(avg_firing_rate_map)] = 0

            # Store the average firing rate map in the firing_rate_maps array
            firing_rate_maps[cell, round_num - 1] = avg_firing_rate_map

    return firing_rate_maps


def firing_rate_map(unit_traces: np.array, lv_rounds: np.array, lv_distance: np.array, n_bins: int) -> np.array:
    """Calculates the spatial firing rate map for an arbitrary trace, defined as the average of the trace over a spatial segment (spatial bin). 
    Parameters
    ----------
    unit_traces : np.array(shape=(n_cells, n_frames))
        A numpy array of 1D arbitrary traces for each neuron. Example: temporal components (raw or z-score).
    lv_rounds : np.array(shape=(n_frames,), dtype=np.int16)
        1D numpy array that marks the number of finished rounds for each frame
    lv_distance : np.array(shape=(n_frames,), dtype=np.float64)
        1D numpy array of the distance per round quantity.
    n_bins : int
        the number of spatial bins to calculate
    Returns
    -------
    np.array(shape=(n_components, n_rounds, n_bins))
        A 3D array that contains for each component, for each round, the firing rate corresponding to each spatial bin.
    """
    # make sure filtering was done correctly, i.e. frames of traces match with frames of loco data
    assert unit_traces.shape[1] == len(lv_rounds)
    assert len(lv_rounds) == len(lv_distance)

    n_units = unit_traces.shape[0]

    # get all unique rounds included in the data
    rounds_to_include = np.unique(lv_rounds)
    n_rounds = len(rounds_to_include)
    firing_rate_maps = np.zeros(
        shape=(n_units, n_rounds, n_bins), dtype=np.float64)
    for i_round, round in enumerate(rounds_to_include):
        frames_current_round = lv_rounds == round
        distance_current_round = lv_distance[frames_current_round]
        traces_current_round = unit_traces[:, frames_current_round]
        for i_unit in range(n_units):
            hist, _ = np.histogram(
                distance_current_round, bins=n_bins, weights=traces_current_round[i_unit])
            count_hist, _ = np.histogram(distance_current_round, bins=n_bins)
            avg_firing_rate_map = np.divide(
                hist, count_hist, out=np.zeros_like(hist), where=(count_hist != 0), dtype=firing_rate_maps.dtype)
            avg_firing_rate_map[np.isnan(avg_firing_rate_map)] = 0
            firing_rate_maps[i_unit, i_round, :] = avg_firing_rate_map
    return firing_rate_maps


def get_running_frames(speed_trace: ArrayLike, sampling_rate: float = 15., min_duration: float = 1, merge_threshold: float = 0.5, min_peak_speed: float = 5.) -> ArrayLike:
    """
    Classify frames as locomotion or not, based on Danielson et al. 2016. Merge frames of locomotion when 
    break between them is short enough; filter on minimum length and minimum peak speed.
    Parameters
    ----------
    speed_trace : np.array(shape=(n_frames,))
    The locomotion speed in units cm/s 
    sampling_rate: float
    The sampling rate (Hz) of the locomotion trace.
    min_duration: float
    Minimum duration (s) of a loco segment to be accepted.
    merge_threshold: float
    Merge two locomotion intervals separated by less than merge_threshold (s) stillness
    min_peak_speed: float
    Minimum peak speed (cm/s) required to classify as locomotion
    Returns
    -------
    np.array(shape=(n_frames,))
        A boolean array that is True at indices that fulfill the internal logic to classify as locomotion, False otherwise. 
    """
    min_duration_frames = round(min_duration*sampling_rate)
    merge_threshold_frames = round(merge_threshold*sampling_rate)
    speed_binary = speed_trace > 0.
    # 1. merge loco intervals
    # for each frame, check if end or beginning of a loco cluster.
    # if end, set i_end_last_loco to current frame
    # if beginning, check how many frames elapsed since last end of a loco cluster.
    # If less than <0.5s equivalent, fill the gap with 1s
    # check i_end_last_loco > 0 to avoid filling beginning of recording with 1 if first loco starts <0.5s
    i_end_last_loco = -1
    for i_frame in range(1, len(speed_binary)):
        # detect if end of loco
        # now at rest and last frame loco
        if (not speed_binary[i_frame]) and speed_binary[i_frame-1]:
            i_end_last_loco = i_frame - 1
        # detect if beginning of loco
        # loco now and at rest last frame
        elif speed_binary[i_frame] and not speed_binary[i_frame-1]:
            # check if gap short enough to merge
            # for example, [1, 0, 0, 0, 1] -> gap is 3 frames = 4 - (0 + 1)
            if i_end_last_loco > 0 and (i_frame - i_end_last_loco) < merge_threshold_frames:
                speed_binary[i_end_last_loco+1:i_frame] = 1
    # 2. filter by minimum duration and minimum peak speed
    # loop over frames, detect beginning and end of loco, check if passes threshold
    i_begin = -1
    i_end = -1
    for i_frame in range(1, len(speed_binary)):
        if speed_binary[i_frame] and not speed_binary[i_frame-1]:
            i_begin = i_frame
        elif not speed_binary[i_frame] and speed_binary[i_frame-1]:
            i_end = i_frame-1
            # found end, do the filtering now
            is_long = (i_end - i_begin - 1) >= min_duration_frames
            is_fast = np.max(speed_trace[i_begin:i_end+1]) >= min_peak_speed
            # check if any of the criteria not fulfilled
            if (not is_long) or (not is_fast):
                speed_binary[i_begin:i_end+1] = 0
    return speed_binary


# taking the zscore flurorescence. finding the peaks in it and making a binary panda frame out of it.
# meaning with 0 and 1. zero is events and 1 is events

# def make_binary(data,peak_threshold=3,peak_distance=10):
#         #the data file will be the the firing rate map of every cell
#         data_binarized=np.zeros_like(data) #making a copy of the original file
#         for unit in range(data.shape[0]):
#             peaks_all_data=peak_local_max(data[unit],threshold_abs=peak_threshold,min_distance=peak_distance,exclude_border=False)
#             data_binarized[unit][peaks_all_data[:, 0], peaks_all_data[:, 1]] = 1
#         return data_binarized


def make_binary(firing_rate_map, peak_threshold=3, peak_distance=10):
    """Returns a binary array with 1 where the firing rate map contains peaks

    Parameters
    ----------
    firing_rate_map : np.array(shape=(n_units, n_rounds, n_bins)) or np.array(shape=(n_rounds, n_bins))
        The firing rate map data.
    peak_threshold : int, optional
        The threshold value a bin value should pass to be classified as peak (same unit as firing_rate_map), by default 3
    peak_distance : int, optional
        The required minimum distance between two peaks, by default 10

    Returns
    -------
    np.array(shape=firing_rate_map.shape) 
        The binary data filled with 1 where a peak was detected, 0 otherwise 
    """
    data_binarized = np.zeros_like(
        firing_rate_map)  # making a copy of the original file
    # iterating through cells
    if len(firing_rate_map.shape) == 3:  # (n_units, n_rounds, n_bins)
        for unit in range(firing_rate_map.shape[0]):
            for round in range(firing_rate_map.shape[1]):
                data_per_round = firing_rate_map[unit][round]
                peaks, _ = find_peaks(
                    data_per_round, height=peak_threshold, distance=peak_distance)
                data_binarized[unit][round][peaks] = 1
    elif len(firing_rate_map.shape) == 2:  # single unit data: shape = (n_rounds, n_bins)
        for round in range(firing_rate_map.shape[0]):
            data_per_round = firing_rate_map[round]
            peaks, _ = find_peaks(
                data_per_round, height=peak_threshold, distance=peak_distance)
            data_binarized[round][peaks] = 1
    else:
        raise ValueError(
            f"Invalid input shape: make_binary() takes 2D or 3D numpy array as firing_rate_map argument, received array with shape {firing_rate_map.shape}")
    return data_binarized


def adding_parameters(zscore_fluo_pd, raw_fluo_pd, param_file):

    # panda frame for time
    time_hdf = h5py.File(param_file)['inferred']['belt_dict']['tsscn']
    time_hdf = pd.DataFrame(time_hdf)
    time_hdf.columns = ['Time (ms)']
    # panda frame for distance
    distance_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['distance']
    distance_hdf = pd.DataFrame(distance_hdf)
    distance_hdf.columns = ['Distance']
    # panda frame for speed
    speed_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['speed']
    speed_hdf = pd.DataFrame(speed_hdf)
    speed_hdf.columns = ['Speed']
    # panda frame for number of rounds
    rounds_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['rounds']
    rounds_hdf = pd.DataFrame(rounds_hdf)
    rounds_hdf.columns = ['Rounds']
    rounds_hdf = rounds_hdf.astype(int)
    # panda frame for running(yes or no running)
    running_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['running']
    running_hdf = pd.DataFrame(running_hdf)
    running_hdf.columns = ['Running']
    running_hdf = running_hdf.astype(int)

    #####################################################################################################################

    # adding all the parameters in one panda frame for z score and raw data
    zscore_fluo_pd = pd.concat([zscore_fluo_pd, time_hdf, distance_hdf,
                               speed_hdf, rounds_hdf, running_hdf], axis=1, ignore_index=True)
    raw_fluo_pd = pd.concat([raw_fluo_pd, time_hdf, distance_hdf,
                            speed_hdf, rounds_hdf, running_hdf], axis=1, ignore_index=True)
    # Create a mapping dictionary for column renaming
    rename_mapping = {old_col: new_col for old_col, new_col in zip(
        zscore_fluo_pd.columns[-5:], ['Time (ms)', 'Distance', 'Speed', 'Rounds', 'Running'])}
    # Rename the columns
    zscore_fluo_pd = zscore_fluo_pd.rename(columns=rename_mapping)
    raw_fluo_pd = raw_fluo_pd.rename(columns=rename_mapping)

    return zscore_fluo_pd, raw_fluo_pd


def read_spatial(A_data, A_indices, A_indptr, A_shape, n_components, resolution, unflatten: bool = False) -> np.array:
    """Given the numpy arrays data, indices, indptr, shape, read the sparse encoded spatial component data and
    reshape it into (n_components, resolution_x, resolution_y)

    Parameters
    ----------
    A_data : np.array
        The data field of the sparse encoding
    A_indices : np.array
        The indices field of the sparse encoding
    A_indptr : np.array
        The indptr field of the sparse encoding
    A_shape : np.array
        The shape field of the sparse encoding
    n_components : int
        the number of components in the CaImAn data
    resolution : tuple(int, int), or [int, int], or np.array(shape=(2,), dtype=dtype("int32"))
        the resolution of the 2p recording. It should be read out from CaImAn dims.
    unflatten : bool
        default: False. If True, the individual spatial components will be converted into 2d arrays. If False,
        left as 1d/flat numpy arrays.
    Returns
    -------
    np.array of shape (n_components, resolution_x * resolution_y ) if unflatten=False, else (n_components, *resolution)
        The dense matrix form of the spatial components.
    """
    spatial = scipy.sparse.csc.csc_matrix(
        (A_data, A_indices, A_indptr), shape=A_shape).todense()  # returns array with dimensions (flat resolution, n_components)
    spatial = np.array(spatial)  # change type to numpy array
    # (262144 -> 512x512, i.e. "unflatten" along imaging resolution)
    spatial = np.swapaxes(spatial, 0, 1)
    if unflatten:
        # TODO: need to test if x and y are in correct order (for asymmetric resolution).
        spatial = np.reshape(spatial, (n_components, *resolution))
    return spatial


def filter_event_count(binary_events: np.array, n_events_threshold: int) -> np.array:
    """Get the indices of units in binary_events that showed events > n_events_threshold. 
    Parameters
    ----------
    binary_events : np.array(shape=(n_units, n_rounds_used, n_bins))
        a 3D numpy array containing binary firing count per cell per round per spatial bin
    n_events_threshold : int
        The number of firing events over the whole data one cell must strictly exceed to be accepted. 

    Returns
    -------
    np.array(shape=(variable, ))
    a 1D numpy array containing the indices of binary_events first axis that pass the threshold. 

    """
    # sum binary events over all rounds and all bins for each cell, compare to threshold
    # boolean array of shape (n_units, )
    cells_with_many_events = np.sum(
        binary_events, axis=(1, 2)) > n_events_threshold
    return np.where(cells_with_many_events)


def cell_morphology(dataset):
    # opening the hpf5 file
    with h5py.File(dataset, "r") as hdf:
        # defining the spatial parameters
        A_data = hdf['estimates']['A']['data']
        A_indices = hdf['estimates']['A']['indices']
        A_indptr = hdf['estimates']['A']['indptr']
        A_shape = hdf['estimates']['A']['shape']
        # number of neurons
        n_neurons = len(hdf['estimates']['C'])
        spatial = scipy.sparse.csc.csc_matrix(
            (A_data, A_indices, A_indptr), shape=A_shape).todense()
        spatial = np.array(spatial)  # change type to numpy array
        # spatial = np.reshape(spatial[:,cell_number], (512, 512)) # (262144 -> 512x512, i.e. "unflatten")
        return spatial


# def ks_test_analysis(data,data_avg=None,n_shuffles=None,num_rounds=None,num_bins=None):
#     shuffled_ks=[] #array where I will put the ks distances where I will compare the shuffled data with my first shuffling
#     baseline=data.copy()

#     #shuffle 1 for the baseline
#     for i in range(num_rounds):
#         shuf=random.randint(1,150)
#         baseline[i]=np.roll(baseline[i],shuf)
#     baseline_avg=np.mean(baseline,axis=0)

#     baseline_ks,_=kstest(data_avg,baseline_avg)


#     # now I will shuffle many times and then compare

#     for n in range(1,shuffling_times):
#         data_shuffle=data.copy()
#         for i in range(num_rounds):
#             shuf=random.randint(1,150)
#             data_shuffle[i]=np.roll(data_shuffle[i],shuf)


#         data_shuffle=np.mean(data_shuffle,axis=0)
#         ks_shuffle,p_value_=kstest(baseline_avg,data_shuffle)
#         shuffled_ks.append(ks_shuffle)
