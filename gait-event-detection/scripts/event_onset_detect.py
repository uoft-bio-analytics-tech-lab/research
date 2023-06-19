import numpy as np
from scipy import signal

def get_mean_freq(x, fs):
    f, Pxx_den = signal.periodogram( x, fs )                                                    
    Pxx_den = np.reshape( Pxx_den, (1,-1) )
    width = np.tile( f[1]-f[0], (1, Pxx_den.shape[1]) )
    f = np.reshape( f, (1, -1) )
    P = Pxx_den * width
    pwr = np.sum(P)

    mean_freq = np.dot( P, f.T ) / pwr

    return mean_freq

def event_onset_detect(acc, fs):

    # 0. Generate time data
    time = np.linspace(1, len(acc), len(acc)) / fs
    
    # 1. Get acceleration norm
    acc_norm = np.sqrt(
        np.sum(
            acc**2,
            axis=1)
        )
    
    # 2. Filter acceleration
    filter_freq = get_mean_freq(acc_norm, fs)
    norm_freq = filter_freq / (fs/2)
    b_filt, a_filt = signal.butter(2, norm_freq)
    acc_norm_filt = signal.filtfilt(b_filt, a_filt, acc_norm)
    
    # 3. Get peak projection point
    id_proj = np.argmax(acc_norm_filt)
    # Get vector connecting signal start to peak projection point
    vector = np.array(
        [[time[0], acc_norm_filt[0]],
         time[id_proj], acc_norm_filt[id_proj]]
        )
    
    # 4. Get orthogonal projections from start to projection
    proj_sz = []
    for i in range(id_proj):
        # Get query point
        query_point = np.array( [time[i], acc_norm_filt[i]] )
        # Get vector endpoints
        p0 = vector[0,:]
        p1 = vector[1,:]
        # Get projection from query point to vector
        a = np.array(
            [-1*query_point[0] * (p1[0] - p0[0]) - query_point[1] * (p1[1] - p0[1]),
             -1*p0[1] * (p1[0] - p0[0]) + p0[0] * (p1[1] - p0[1])]
            )
        b = np.array(
            [[p1[0] - p0[0], p1[1] - p0[1]],
             [p0[1] - p1[1], p1[0] - p0[0]]]
            )
        proj_point = -1 * np.linalg.lstsq(b, a)
        # Check projection point orientation
        if proj_point[1] - query_point[1] > 0:
            # Get projection vector size
            proj_sz.append(
                np.sqrt(
                    (proj_point[0] - query_point[0])**2
                    + (proj_point[1] - query_point[1])**2
                    )
                )
        else:
            proj_sz.append( 0 )
    
    # 5. Get timepoint of longest projection vector
    id_onset = np.argmax( proj_sz )

    return id_onset