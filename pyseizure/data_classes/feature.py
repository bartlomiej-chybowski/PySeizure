from enum import Enum


class Feature(Enum):
    MEAN = 'mean'
    VARIANCE = 'variance'
    SKEWNESS = 'skewness'
    KURTOSIS = 'kurtosis'
    INTERQUARTILE_RANGE = 'interquartile_range'
    MIN = 'min'
    MAX = 'max'
    HJORTH_COMPLEXITY = 'hjorth_complexity'
    HJORTH_MOBILITY = 'hjorth_mobility'
    PETROSIAN_FRACTAL_DIMENSION = 'petrosian_fractal_dimension'
    POWER_SPECTRAL_DENSITY = 'power_spectral_density'           # FF
    POWER_SPECTRAL_CENTROID = 'power_spectral_centroid'         # FF
    SIGNAL_MONOTONY = 'signal_monotony'                         # FF
    SIGNAL_TO_NOISE = 'signal_to_noise'                         # FF
    SPIKE_COUNT = 'spike_count'
    COASTLINE = 'coastline'                                     # TF, FF
    INTERMITTENCY = 'intermittency'
    VOLTAGE_AUC = 'voltage_auc'
    SPIKINESS = 'spikiness'
    STANDARD_DEVIATION = 'standard_deviation'
    ZERO_CROSSING = 'zero_crossing'
    PEAK_TO_PEAK = 'peak_to_peak'
    ABSOLUTE_AREA_UNDER_SIGNAL = 'absolute_area_under_signal'
    TOTAL_SIGNAL_ENERGY = 'total_signal_energy'
    ENERGY_PERCENTAGE = 'energy_percentage'                     # FF
    DISCRETE_WAVELET_TRANSFORM = 'discrete_wavelet_transform'   # FF
    CROSS_CORRELATION_MAX_COEF = 'cross_correlation_max_coef'   # CF
    COHERENCE = 'coherence'                                     # CF
    IMAGINARY_COHERENCE = 'imaginary_coherence'                 # CF
    PHASE_SLOPE_INDEX = 'phase_slope_index'                     # CF
    ECCENTRICITY = 'eccentricity'                               # GF
    CLUSTERING_COEFFICIENT = 'clustering_coefficient'           # GF
    BETWEENNESS_CENTRALITY = 'betweenness_centrality'           # GF
    LOCAL_EFFICIENCY = 'local_efficiency'                       # GF
    GLOBAL_EFFICIENCY = 'global_efficiency'                     # GF
    DIAMETER = 'diameter'                                       # GF
    RADIUS = 'radius'                                           # GF
    CHARACTERISTIC_PATH = 'characteristic_path'                 # GF
