import itertools
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns

from General_Ising_cy_connection import IsingModel
import plotting

plt.rc('text', usetex=True)
sns.set_style("ticks")
sns.set_palette('colorblind')  # Options: deep, muted, pastel, bright, dark, colorblind

SIMULATION_FOLDER = "./simulation_runs"
SAVE_LOCATION = "./analysis_images"


def data_analysis(data, save=False, show_plots=True, exact_ising=True):
    """If save=True, save plots."""
    energies = {}
    energy_correlations = {}
    magnetizations = {}
    magnetization_correlations = {}
    cluster_fractions = {}
    heat_capacities = {}
    magnetizabilities = {}
    binder_cumulants = {}
    (lattice_size, bond_energy, thermalization_sweeps, measurement_sweeps,
     temperature, correction_factor, correction_factor_error) = data[0]
    measurements = data[1]
    heat_cap_jack, heat_cap_error_jack, _ = jackknife(measurements['energy bins'],
                                                      measurements['energy sq bins'], 8, heat_capacity,
                                                      temperature=temperature, no_of_sites=lattice_size ** 2)
    heat_cap_boot, heat_cap_error_boot = bootstrap(measurements['energy bins'], measurements['energy sq bins'],
                                                   1000, heat_capacity, temperature=temperature,
                                                   no_of_sites=lattice_size ** 2)
    heat_cap = np.mean([heat_cap_jack, heat_cap_boot])
    heat_cap_error = max(heat_cap_error_jack, heat_cap_error_boot)

    # Magnetization.
    chi_jack, chi_error_jack, _ = jackknife(measurements['m bins'], measurements['mag sq bins'], 8,
                                            susceptibility, temperature=temperature,
                                            no_of_sites=lattice_size ** 2)
    chi_boot, chi_error_boot = bootstrap(measurements['m bins'], measurements['mag sq bins'], 1000,
                                         susceptibility, temperature=temperature, no_of_sites=lattice_size ** 2)
    chi = np.mean([chi_jack, chi_boot])
    chi_error = max(chi_error_jack, chi_error_boot)

    binder_jack, binder_error_jack, _ = jackknife(measurements['mag sq bins'], measurements['mag 4th bins'], 8,
                                                  binder_cumulant)
    binder_boot, binder_error_boot = bootstrap(measurements['mag sq bins'], measurements['mag 4th bins'], 1000,
                                               binder_cumulant)
    binder = np.mean([binder_jack, binder_boot])
    binder_error = max(binder_error_jack, binder_error_boot)

    energies.setdefault(lattice_size, []).append(
        (temperature, measurements['energy'], measurements['energy error']))
    energy_correlations.setdefault(lattice_size, []).append((temperature, measurements['energy correlation']))
    magnetizations.setdefault(lattice_size, []).append(
        (temperature, measurements['m'], measurements['m error']))
    cluster_fractions.setdefault(lattice_size, []).append(
        (temperature, correction_factor, correction_factor_error))
    heat_capacities.setdefault(lattice_size, []).append((temperature, heat_cap, heat_cap_error))
    magnetizabilities.setdefault(lattice_size, []).append((temperature, chi, chi_error))
    binder_cumulants.setdefault(lattice_size, []).append((temperature, binder, binder_error))
    magnetization_correlations.setdefault(lattice_size, []).append((temperature, measurements['m correlation']))

    # Find the critical temperature.
    # print(f'Binder_cumulants: {binder_cumulants}')
    # critical_temperature, critical_temperature_error = find_binder_intersection(binder_cumulants)

    if exact_ising:
        exact_heat = IsingModel.exact_heat_capacity(bond_energy, 0, 10 * np.absolute(bond_energy))
        exact_energy = IsingModel.exact_internal_energy(bond_energy, 0, 10 * np.absolute(bond_energy))
        exact_magnetization = IsingModel.exact_magnetization(bond_energy, 0, 10 * np.absolute(bond_energy))
    else:
        exact_heat = None
        exact_energy = None
        exact_magnetization = None

    if show_plots:
        plotting.plot_quantity_range(energies, "Energy per Site", exact=exact_energy, save=save)
        plotting.plot_quantity_range(cluster_fractions, "Mean Cluster Size as Fraction of Lattice", save=save)
        plotting.plot_quantity_range(magnetizations, "Absolute Magnetization per Site", exact=exact_magnetization,
                                     save=save)
        plotting.plot_quantity_range(heat_capacities, "Heat Capacity per Site", exact=exact_heat, save=save)
        plotting.plot_quantity_range(magnetizabilities, "Susceptibility per Site", save=save)
        plotting.plot_quantity_range(binder_cumulants, "Binder Cumulant", save=save)
        plotting.plot_correlation_time_range(energy_correlations, "Energy per Site", save=save)
        plotting.plot_correlation_time_range(magnetization_correlations, "Absolute Magnetization", save=save)

    # return critical_temperature, critical_temperature_error, magnetizabilities, magnetizations, heat_capacities
    return magnetizabilities, magnetizations, heat_capacities

def bootstrap(data1, data2, no_of_resamples, operation, **kwargs):
    """Calculate error using the bootstrap method."""
    resamples = np.empty(no_of_resamples)
    for k in range(no_of_resamples):
        random_picks1 = np.random.choice(data1, len(data1))
        random_picks2 = np.random.choice(data2, len(data2))
        resamples.put(k, operation(random_picks1, random_picks2, kwargs))

    error = calculate_error(resamples)
    return np.mean(resamples), error


def calculate_error(data):
    """Calculate the error on a data set."""
    return np.std(data) / np.sqrt(len(data))


def binning(data, title):
    """
    Calculate autocorrelation time, mean and error for a quantity using the binning method.
    The bins become uncorrelated when the error approaches a constant.
    These uncorrelated bins can be used for jackknife resampling.
    """
    original_length = len(data)
    errors = []
    errors.append((original_length, calculate_error(data)))
    while len(data) > 128:
        data = np.asarray([(a + b) / 2 for a, b in zip(data[::2], data[1::2])])
        errors.append((len(data), calculate_error(data)))
    autocorrelation_time = 0.5 * ((errors[-1][1] / errors[0][1]) ** 2 - 1)
    if np.isnan(autocorrelation_time) or autocorrelation_time <= 0:
        autocorrelation_time = 1
    return np.mean(data), errors[-1][1], autocorrelation_time, data


def jackknife(data1, data2, no_of_bins, operation, **kwargs):
    """
    Calculate errors using jackknife resampling.
    no_of_bins should divide len(data)
    """
    data_length = len(data1)
    all_bin_estimate = operation(data1, data2, kwargs)
    calculated_values = []
    split_data1 = np.split(data1, no_of_bins)
    split_data2 = np.split(data2, no_of_bins)
    # From https://stackoverflow.com/questions/28056195/
    mask = np.arange(1, no_of_bins) - np.tri(no_of_bins, no_of_bins - 1, k=-1, dtype=bool)
    leave_one_out1 = np.asarray(split_data1)[mask]
    leave_one_out2 = np.asarray(split_data2)[mask]
    for m1, m2 in zip(leave_one_out1, leave_one_out2):
        value = operation(np.concatenate(m1), np.concatenate(m2), kwargs)
        calculated_values.append(value)
    mean = np.sum(calculated_values) / no_of_bins
    standard_error = np.sqrt((1 - 1 / data_length) * (np.sum(np.asarray(calculated_values) ** 2 - mean ** 2)))
    bias = (no_of_bins - 1) * (mean - all_bin_estimate)
    if bias >= 0.5 * standard_error and bias != 0:
        print("Bias is large for {0}: error is {1}, bias is {2} ".format(operation, standard_error, bias))
    return all_bin_estimate, standard_error, bias


def heat_capacity(energy_data, energy_sq_data, kwargs):
    """
    Calculate the heat capacity for a given energy data set and temperature.
    Multiply by the number of sites, because the data has been normalised to the number of sites.
    """
    return kwargs['no_of_sites'] / kwargs['temperature'] ** 2 * (np.mean(energy_sq_data) - np.mean(energy_data) ** 2)


def magnetization(state, **kwargs):
    """Calculate the magnetization."""
    return 1 / kwargs['no_of_sites'] * np.mean(state)


def susceptibility(magnetization_data, magnetization_sq_data, kwargs):
    """Calculate the susceptibility."""
    return kwargs['no_of_sites'] / kwargs['temperature'] * (
                np.mean(magnetization_sq_data) - np.mean(magnetization_data) ** 2)


def binder_cumulant(magnetization_sq_data, magnetization_4th_data, kwargs):
    """Calculate the binder cumulant of 4th grade."""
    return 1 - np.mean(magnetization_4th_data) / (3 * np.mean(magnetization_sq_data) ** 2)


def find_binder_intersection(data):
    """Find the intersection of Binder cumulant data series for different lattice sizes."""
    intersections = []
    keys = sorted(data.keys())
    for key in keys[:-1]:
        data1 = data[key]
        data2 = data[keys[keys.index(key) + 1]]
        for z in range(len(data1) - 1):
            intersection_error = []
            for e in [-1, 0, 1]:
                p1a = data1[z]
                p1b = data1[z + 1]
                dx1 = p1b[0] - p1a[0]
                dy1 = p1b[1] + e * p1b[2] - (p1a[1] + e * p1a[2])
                dydx1 = dy1 / dx1
                c1 = -dydx1 * p1a[0] + p1a[1]

                p2a = data2[z]
                p2b = data2[z + 1]
                dx2 = p2b[0] - p2a[0]
                dy2 = p2b[1] + e * p2b[2] - (p2a[1] + e * p2a[2])
                dydx2 = dy2 / dx2
                c2 = -dydx2 * p2a[0] + p2a[1]

                det = -dydx1 + dydx2
                if det == 0:
                    continue
                intersection_x = (c1 - c2) / det
                intersection_y = (dydx2 * c1 - dydx1 * c2) / det
                # The intersection should lie within the area of interest.
                if p1a[0] <= intersection_x <= p1b[0]:
                    intersection_error.append((intersection_x, intersection_y))

            if intersection_error:
                x_intersection = np.mean([i[0] for i in intersection_error])
                x_intersection_error = calculate_error([i[0] for i in intersection_error])
                y_intersection = np.mean([i[1] for i in intersection_error])
                y_intersection_error = calculate_error([i[1] for i in intersection_error])
                intersections.append(((x_intersection, x_intersection_error), (y_intersection, y_intersection_error)))
    if intersections:
        critical_temperature = np.mean([p[0][0] for p in intersections])
        critical_temperature_error = calculate_error([p[0][0] for p in intersections])
    else:
        critical_temperature = None
        critical_temperature_error = None
    # print([i[0] for i in intersections])
    print("Critical temperature is {0} +/- {1}".format(critical_temperature, critical_temperature_error))

    return critical_temperature, critical_temperature_error
