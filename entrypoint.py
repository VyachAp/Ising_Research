import pickle
import time
import numpy as np
import analysis
import General_Ising_cy_connection

SIMULATION_FOLDER = "./simulation_runs"


def single_temperature_simulation(algorithm, lattice_size, bond_energy, temperature,
                                  thermalization_sweeps, measurement_sweeps):
    total_no_of_sweeps = thermalization_sweeps + measurement_sweeps
    print(f"Lattice Size {lattice_size}, Temperature {temperature}")
    model = General_Ising_cy_connection.IsingModel(lattice_size, bond_energy, temperature, total_no_of_sweeps)
    if algorithm == "metropolis":
        model.metropolis()
        correlation_correction = 1
        correlation_correction_error = 0
    elif algorithm == "wolff":
        cluster_sizes = model.wolff()
        correlation_correction = np.mean(cluster_sizes) / model.number_of_sites
        correlation_correction_error = analysis.calculate_error(cluster_sizes) / model.number_of_sites
    else:
        raise Exception("Invalid algorithm.")

    equilibrated_energy = model.energy_log[thermalization_sweeps:] / model.number_of_sites
    equilibrated_magnetization = model.magnetization_log[thermalization_sweeps:] / model.number_of_sites

    energy, energy_error, energy_correlation, energy_bins = analysis.binning(equilibrated_energy, "Energy per Site")
    energy_sq, energy_sq_error, _, energy_sq_bins = analysis.binning(equilibrated_energy ** 2, "Energy Squared per Site")

    abs_magnetization, abs_magnetization_error, abs_magnetization_correlation, abs_magnetization_bins = analysis.binning(np.absolute(equilibrated_magnetization), "<|M|>/N")
    magnetization, magnetization_error, _, magnetization_bins = analysis.binning(equilibrated_magnetization, "Magnetization per Site")
    magnetization_squared, magnetization_squared_error, _, magnetization_squared_bins = analysis.binning(equilibrated_magnetization ** 2, "Magnetization Squared per site")
    magnetization_4th, magnetization_4th_error, _, magnetization_4th_bins = analysis.binning(equilibrated_magnetization ** 4, "Magnetization^4 per Site")

    data = ((lattice_size, bond_energy, thermalization_sweeps, measurement_sweeps,
             temperature, correlation_correction, correlation_correction_error),
            {"energy": energy, "energy error": energy_error, "energy correlation": energy_correlation * correlation_correction,
             "energy sq": energy_sq, "energy sq error": energy_sq_error,
             "m": abs_magnetization, "m error": abs_magnetization_error, "m correlation": abs_magnetization_correlation * correlation_correction, "m bins": abs_magnetization_bins,
             "mag": magnetization, "mag error": magnetization_error,"mag sq bins": magnetization_squared_bins,
             "mag 4th": magnetization_4th, "mag 4th error": magnetization_4th_error})

    print("Energy per site: {0} +/- {1}".format(energy, energy_error))
    print("\n")

    return data


def run(algorithm, lattice_sizes, bond_energy,
        thermalization_sweeps, measurement_sweeps, lower, upper, step=0.2, save=True):
    """Run a given model over a range of temperature."""
    for k in lattice_sizes:
        simulations = []
        num_of_samples = round(((upper - lower) / step) + 1)
        for t in np.linspace(lower, upper, num_of_samples):
            data = single_temperature_simulation(algorithm, k, bond_energy, t, thermalization_sweeps,
                                                 measurement_sweeps)
            simulations.append(data)

        if save:
            with open("{0}/{1}_{2}_{3}_{4}_{5}_[{6}-{7}].pickle".format(SIMULATION_FOLDER,
                                                                        time.strftime("%Y%m%d_%H%M%S",
                                                                                      time.localtime(time.time())),
                                                                        algorithm, k, measurement_sweeps, lower, upper,
                                                                        step), 'wb+') as f:
                print(simulations)
                pickle.dump(simulations, f, pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == '__main__':
    run('wolff', (10, 10), 1, 5000, 16384, 2, 3)
