# Functions to compute the distance (similarity) of two seismic sections

# External dependencies
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dtaidistance import dtw
import pywt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import directed_hausdorff
from scipy.signal import correlate2d
from sklearn.metrics.pairwise import cosine_similarity
import pylops
from pylops.utils.wavelets import ricker


# CONSTANTS

# Grid constants
nx, nz = 100, 100  # Grid size in nx (n, offset) and z (depth) directions
x = np.linspace(0, nx-1, nx)
z = np.linspace(0, -(nx-1), nz)
x_grid, z_grid = np.meshgrid(x, z)

# Impedance values
salt_impedance = 1
rock_impedance = 0

# Wavelet
nt0 = 51
dt0 = 0.002
t0 = np.arange(nt0) * dt0
ntwav = 101
wav, twav, wavc = ricker(t0[: ntwav // 2 + 1], 20)

# PyLops dense operator
PPop_dense = pylops.avo.poststack.PoststackLinearModelling(
    wav / 2, nt0=nz, spatdims=nx, explicit=True
)
# PyLops lop operator
PPop = pylops.avo.poststack.PoststackLinearModelling(wav / 2, nt0=nz, spatdims=nx)


# FUNTIONS

# SSIM Index
def compute_ssim(simulated_data, observed_data):

    # Data range
    data_range = observed_data.max() - observed_data.min()
    # SSIM
    ssim_value, _ = ssim(simulated_data, observed_data, data_range=data_range, full=True)   

    return ssim_value


# MAE
def compute_mae(simulated_data, observed_data):  

    return np.mean(np.abs(simulated_data - observed_data))


# Normalized Root Mean Squared
def compute_nrmse(simulated_data, observed_data):

    # Root mean squared error
    rmse = np.sqrt(np.mean((simulated_data - observed_data)**2))
    # Data range for normalization
    data_range = observed_data.max() - observed_data.min()   

    return rmse / data_range


# Correlation
def compute_correlation(simulated_data, observed_data):   

    return np.corrcoef(simulated_data.flatten(), observed_data.flatten())[0, 1]


# Dynamic Time Warping
def compute_dtw(simulated_data, observed_data):

    # Initialization
    distances = []

    # DTW and concatenation with distances
    for row_sim, row_obs in zip(simulated_data, observed_data):
        distance = dtw.distance(row_sim, row_obs)
        distances.append(distance)    

    return np.mean(distances)


# Energy Difference
def compute_energy_difference(simulated_data, observed_data):

    # Sum of seismic sections
    energy_sim = np.sum(simulated_data**2)
    energy_obs = np.sum(observed_data**2)    

    return np.abs(energy_sim - energy_obs) / energy_obs


# Earth Mover's Distance (Wasserstein distance)
def compute_emd(simulated_data, observed_data):  

    return wasserstein_distance(simulated_data.flatten(), observed_data.flatten())


# Cross-correlation
def compute_cross_correlation(simulated_data, observed_data):

    # Cross-correlation with correlated2D
    cross_corr = correlate2d(simulated_data, observed_data, mode="valid")       

    return np.max(cross_corr)  # Peak correlation value


# Pearson Correlation
def compute_pearson_correlation(simulated_data, observed_data):

    # Initialization
    correlation = np.zeros(nx)

    for i in range (nx):        

        # Arrays
        array1 = simulated_data[:, i]
        array2 = observed_data[:, i]

        # Mean of each array
        mean1 = np.mean(array1)
        mean2 = np.mean(array2)

        # Numerator: covariance between the arrays
        covariance = np.sum((array1 - mean1) * (array2 - mean2))

        # Denominator: product of standard deviations
        std1 = np.sqrt(np.sum((array1 - mean1)**2))
        std2 = np.sqrt(np.sum((array2 - mean2)**2))

        # Pearson correlation coefficient
        correlation[i] = covariance / (std1 * std2)

    return np.exp(-correlation / 2.0)


# Element wise multiplication
def compute_element_wise_multiplication(simulated_data, observed_data):

    # Initialization
    el_w_mult = np.zeros(nx)

    for i in range (nx):        

        # Arrays
        array1 = simulated_data[:, i]
        array2 = observed_data[:, i]

        # Multiplication
        el_w_mult[i] = np.sum(np.multiply(array1, array2))

    return np.sum(el_w_mult) / nx

    
# Wavelet transform
def compute_wavelet_transform(simulated_data, observed_data):

    # Sections
    image1 = simulated_data
    image2 = observed_data

    # Wavelet transform
    coeffs1 = pywt.dwt2(image1, 'haar')
    coeffs2 = pywt.dwt2(image2, 'haar')

    # Approximation and detail coefficients (transformed part)
    cA1, (cH1, cV1, cD1) = coeffs1
    cA2, (cH2, cV2, cD2) = coeffs2

    # SSIM on the wavelet coefficients (approximation and details separately)
    ssim_cA, _ = ssim(cA1, cA2, full=True, data_range=1.0)
    ssim_cH, _ = ssim(cH1, cH2, full=True, data_range=1.0)
    ssim_cV, _ = ssim(cV1, cV2, full=True, data_range=1.0)
    ssim_cD, _ = ssim(cD1, cD2, full=True, data_range=1.0)

    return (1 - ssim_cA * ssim_cH * ssim_cV * ssim_cD)


# Difference and total sum
def compute_difference_and_total_sum(simulated_data, observed_data):

    return np.sum(simulated_data - observed_data)


# Cosine Similarity
def compute_cosine_similarity_seismic(simulated_data, observed_data):

    # Flattening of seismic sections
    seismic1_flat = simulated_data.flatten().reshape(1, -1)
    seismic2_flat = observed_data.flatten().reshape(1, -1)
    
    return cosine_similarity(seismic1_flat, seismic2_flat)[0, 0]


# Semblance
def compute_semblance(observed_data, seismic_section):

    # Initialization
    window_size = 10
    semblance = np.zeros((nz, nx*2))

    for i in range (0, nx):
        semblance[:, 2*i] = observed_data[:, i]
        semblance[:, 2*i+1] = seismic_section[:, i]

    half_window = window_size // 2

    # Semblance for each trace and time sample
    for trace_idx in range(0, nx*2):
        for time_idx in range(0, nz):
            # Define the bounds of the moving window
            start_trace = max(0, trace_idx - half_window)
            end_trace = min(nx*2, trace_idx + half_window + 1)
            start_time = max(0, time_idx - half_window)
            end_time = min(nz, time_idx + half_window + 1)

            # Windowed data
            #window_data = seismic_section[start_trace:end_trace, start_time:end_time]
            window_data = semblance[:, start_trace:end_trace][start_time:end_time, :]
            
            # Semblance
            num = (np.sum(np.sum(window_data, axis=1)))**2
            denom = window_data.shape[1]*window_data.shape[0] * np.sum((np.sum(window_data**2, axis=1)))
            
            if denom == 0:
                semblance[time_idx, trace_idx] = 0
            else:
                semblance[time_idx, trace_idx] = num / denom

            semblance[time_idx, trace_idx] = num / denom

            semblance_sum = np.sum(np.sum(semblance))

    return semblance, semblance_sum


# Combined Semblance
def compute_combined_semblance(observed_data, seismic_section):

    # Initialization
    window_size = 10
    semblance = np.zeros((nz, nx*2))

    # Combined sections from the two given sections
    for i in range (0, nx):
        semblance[:, 2*i] = observed_data[:, i]
        semblance[:, 2*i+1] = seismic_section[:, i]

    half_window = window_size // 2

    # Semblance for each trace and time sample
    for trace_idx in range(0, nx*2):
        for time_idx in range(0, nz):
            # Bounds of moving window
            start_trace = max(0, trace_idx - half_window)
            end_trace = min(nx*2, trace_idx + half_window + 1)
            start_time = max(0, time_idx - half_window)
            end_time = min(nz, time_idx + half_window + 1)

            # Windowed data
            window_data = semblance[:, start_trace:end_trace][start_time:end_time, :]
            
            # Semblance
            num = (np.sum(np.sum(window_data, axis=1)))**2
            denom = window_data.shape[1]*window_data.shape[0] * np.sum((np.sum(window_data**2, axis=1)))
            
            #if denom == 0:
            #    semblance[time_idx, trace_idx] = 0
            #else:
            #    semblance[time_idx, trace_idx] = num / denom

            semblance[time_idx, trace_idx] = num / denom

            semblance_sum = np.sum(np.sum(semblance))

    return semblance, semblance_sum


# Wavelet Distance (with Wassersten distance)
def compute_wavelet_distance(observed_data, simulated_data):

    # Initialization
    bins=500
    wavelet="db1"
    levels=50
    total_distance = 0
  
    # Decomposeition of observed and simulated data into wavelet coefficients
    coeffs_f = pywt.wavedec2(observed_data, wavelet=wavelet, level=levels)
    coeffs_g = pywt.wavedec2(simulated_data, wavelet=wavelet, level=levels)

    # Loop through levels and sub-bands
    for level in range(len(coeffs_f)):

        if level == 0:
            # Approximation coefficients at the coarsest level
            coeff_f = coeffs_f[level]
            coeff_g = coeffs_g[level]
            
            # Histogram of coefficients
            hist_f, _ = np.histogram(coeff_f, bins=bins, range=(coeff_f.min(), coeff_f.max()), density=True)
            hist_g, _ = np.histogram(coeff_g, bins=bins, range=(coeff_g.min(), coeff_g.max()), density=True)

            #plt.plot(hist_f)
            #plt.plot(hist_g)
            
            # Statistical distance with Wasserstein distance)
            d_sc = wasserstein_distance(hist_f, hist_g)
            
            # Accumulated distance (approximation level gets higher weight)
            total_distance += d_sc * 2

        else:
            # Detailed coefficients: Horizontal, Vertical, Diagonal
            for sub_band in range(3):
                coeff_f = coeffs_f[level][sub_band]
                coeff_g = coeffs_g[level][sub_band]
                
                # Histogram of coefficients
                hist_f, _ = np.histogram(coeff_f, bins=bins, range=(coeff_f.min(), coeff_f.max()), density=True)
                hist_g, _ = np.histogram(coeff_g, bins=bins, range=(coeff_g.min(), coeff_g.max()), density=True)
                
                #plt.plot(hist_f)
                #plt.plot(hist_g)

                # Statistical distance with Wasserstein distance
                d_sc = wasserstein_distance(hist_f, hist_g)
                
                # Accumulated distance
                total_distance += d_sc
    
    return total_distance


# Hausdorff Distance
def compute_hausdorff (observed_data, simulated_data):

    # Convertion of images to coordinate sets
    observed_coords = np.array(np.nonzero(observed_data)).T  # Transpose to get (row, col) pairs
    simulated_coords = np.array(np.nonzero(simulated_data)).T  # Transpose to get (row, col) pairs

    # Computation of directed Hausdorff distances
    forward_distance = directed_hausdorff(observed_coords, simulated_coords)[0]
    backward_distance = directed_hausdorff(simulated_coords, observed_coords)[0]

    # Hausdorff distance is the maximum of forward and backward distances
    hausdorff_distance = max(forward_distance, backward_distance)

    return hausdorff_distance