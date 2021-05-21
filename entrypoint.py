import matplotlib.pyplot as plt
import numpy as np
import analysis
import General_Ising_cy_connection
import General_Potts_cy_connection
from SAW_generator import plot

SIMULATION_FOLDER = "./simulation_runs"
plt.rcParams.update({'errorbar.capsize': 2})


def single_temperature_simulation(model, algorithm, lattice_size, lattice_x, lattice_y, bond_energy, temperature,
                                  thermalization_sweeps, measurement_sweeps):
    total_no_of_sweeps = thermalization_sweeps + measurement_sweeps
    print(f"Lattice Size {lattice_size}, Temperature {temperature}")
    if model == 'Ising':
        model = General_Ising_cy_connection.IsingModel(lattice_size, lattice_x, lattice_y, bond_energy, temperature, total_no_of_sweeps)
    elif model == 'Potts':
        model = General_Potts_cy_connection.PottsModel(lattice_size, lattice_x, lattice_y, bond_energy, temperature, total_no_of_sweeps)
    else:
        raise Exception("Invalid model")
    if algorithm == "metropolis":
        model.metropolis()
        correlation_correction = 1
        correlation_correction_error = 0
    elif algorithm == "wolff":
        cluster_sizes = model.wolff()
        correlation_correction = np.mean(cluster_sizes) / model.number_of_sites
        correlation_correction_error = analysis.calculate_error(cluster_sizes) / model.number_of_sites
    else:
        raise Exception("Invalid algorithm")

    equilibrated_energy = model.energy_history[thermalization_sweeps:] / model.number_of_sites
    equilibrated_magnetization = model.magnetization_history[thermalization_sweeps:] / model.number_of_sites

    energy, energy_error, energy_correlation, energy_bins = analysis.binning(equilibrated_energy, "Energy per Site")
    energy_sq, energy_sq_error, _, energy_sq_bins = analysis.binning(equilibrated_energy ** 2,
                                                                     "Energy Squared per Site")
    energy_4th, energy_4th_error, _, energy_4th_bins = analysis.binning(equilibrated_energy ** 4, "Energy^4 per Site")

    abs_magnetization, abs_magnetization_error, abs_magnetization_correlation, abs_magnetization_bins = analysis.binning(
        np.absolute(equilibrated_magnetization), "<|M|>/N")
    magnetization, magnetization_error, _, magnetization_bins = analysis.binning(model.lattice,
                                                                                 "Magnetization per Site")
    magnetization_squared, magnetization_squared_error, _, magnetization_squared_bins = analysis.binning(
        equilibrated_magnetization ** 2, "Magnetization Squared per site")
    magnetization_4th, magnetization_4th_error, _, magnetization_4th_bins = analysis.binning(
        equilibrated_magnetization ** 4, "Magnetization^4 per Site")

    data = ((lattice_size, bond_energy, thermalization_sweeps, measurement_sweeps,
             temperature, correlation_correction, correlation_correction_error),
            {"energy": energy, "energy error": energy_error,
             "energy bins": energy_bins, "energy correlation": energy_correlation * correlation_correction,
             "energy sq": energy_sq, "energy sq error": energy_sq_error, "energy sq bins": energy_sq_bins,
             "energy 4th": energy_4th, "energy 4th error": energy_4th_error, "energy 4th bins": energy_4th_bins,
             "m": abs_magnetization, "m error": abs_magnetization_error,
             "m correlation": abs_magnetization_correlation * correlation_correction, "m bins": abs_magnetization_bins,
             "mag": magnetization, "mag error": magnetization_error, "mag bins": magnetization_bins,
             "mag sq": magnetization_squared, "mag sq error": magnetization_squared_error,
             "mag sq bins": magnetization_squared_bins,
             "mag fourth": magnetization_4th, "mag 4th error": magnetization_4th_error,
             "mag 4th bins": magnetization_4th_bins})

    return data


def run(model, algorithm, lattice_sizes, x_coords, y_coords, bond_energy,
        thermalization_sweeps, measurement_sweeps, lower, upper, step=0.01):
    """Run a given model over a range of temperature."""
    for k in lattice_sizes:
        num_of_samples = round(((upper - lower) / step) + 1)
        energies = []
        energies_error = []
        magnetizations = []
        magnetizations_error = []
        for t in np.linspace(lower, upper, num_of_samples):
            data = single_temperature_simulation(model, algorithm, k, x_coords, y_coords, bond_energy, t, thermalization_sweeps,
                                                 measurement_sweeps)

            energies.append(data[1]['energy'])
            energies_error.append(data[1]['energy error'])
            magnetizations.append(data[1]['mag sq'])
            magnetizations_error.append(data[1]['mag sq error'])

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.errorbar(np.linspace(lower, upper, num_of_samples), energies, yerr=energies_error, color="#A60628")
        ax.set_xlabel("Temperature", fontsize=20)
        ax.set_ylabel("Energy", fontsize=20)
        ax.grid()
        ax.set_title(f"Energy to temperature ({k})")
        # ax.legend(loc=2)
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.errorbar(np.linspace(lower, upper, num_of_samples), np.power(magnetizations, 2), yerr=magnetizations_error,
                     ecolor='#2196F3', color="#A60628")
        ax.set_xlabel("Temperature", fontsize=20)
        ax.set_ylabel("Magnetization", fontsize=20)
        ax.grid()
        ax.set_title(f"Magnetization to temperature ({k})")
        # ax.legend(loc=2)
        plt.show()

    print("Done.")


if __name__ == '__main__':
    lattice_xs, lattice_ys, lattice_size = plot(500)
    run('Potts', 'wolff', [lattice_size], lattice_xs, lattice_ys, 1, 2 ** 12, 2 ** 8, 1, 3)
