"""
Copyright (c) 2021. Revolv Space.

This file was adapted by TUDATPY EXAMPLE APPLICATION: Perturbed Satellite Orbit.

It simulates the orbit of two satellites (marco and olek) for 2 months.

The two satellites are 3U cubesats (5kg each), but one of them has 3x the drag surface area
of the other (satellite side + 2 solar panels of the same size = 3 * 30cm * 10cm).

Dynamical model:
- SH Earth (2, 0)
- aerodynamic
- point mass gravity of Sun and Moon
- Solar Radiation Pressure

At the end of the script, there are some legacy plots from the original tudatpy example.

I added two plots:
1. the linear regression of the semi-major axis (NOTE: this is different from the altitude) to see the difference in
decay of both satellites
2. the angular separation between the satellite (this is not the difference in true anomaly, because we don't know
how much the orbital plane changes, therefore I computed the angular separation as the angle between the two position
vectors, see script).

TODO:
- create generic function that returns the linear regression of the data
(rationale: there are a lot of periodic variations in the output variables, e.g. in the Kepler elements, but we are
not interested in those periodic variations: we want to know the long term trend)
- plot and assess difference in other Kepler elements (e.g., inclination and RAAN for the orbital plane)
"""

###############################################################################
# IMPORT STATEMENTS ###########################################################
###############################################################################
import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import propagation
from matplotlib import pyplot as plt


def main():
    # Load spice kernels.
    spice_interface.load_standard_kernels()

    # Set simulation start and end epochs.
    simulation_start_epoch = 0.0
    simulation_end_epoch = constants.JULIAN_DAY * 60.0

    ###########################################################################
    # CREATE ENVIRONMENT ######################################################
    ###########################################################################

    # Define string names for bodies to be created from default.
    bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle objects.
    bodies.create_empty_body("olek")
    bodies.create_empty_body("marco")

    bodies.get("olek").mass = 5.0
    bodies.get("marco").mass = 5.0

    # Create aerodynamic coefficient interface settings, and add to vehicle
    reference_area = 0.3 * 0.1
    drag_coefficient = 1.2
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0, 0]
    )
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, "olek", aero_coefficient_settings)
    # Second sat
    reference_area = 3 * 0.3 * 0.1
    drag_coefficient = 1.2
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0, 0]
    )
    environment_setup.add_aerodynamic_coefficient_interface(
        bodies, "marco", aero_coefficient_settings)

    # Create radiation pressure settings, and add to vehicle
    reference_area_radiation = 0.3 * 0.1
    radiation_pressure_coefficient = 1.2
    occulting_bodies = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )
    environment_setup.add_radiation_pressure_interface(
        bodies, "olek", radiation_pressure_settings)
    environment_setup.add_radiation_pressure_interface(
        bodies, "marco", radiation_pressure_settings)

    ###########################################################################
    # CREATE ACCELERATIONS ####################################################
    ###########################################################################

    # Define bodies that are propagated.
    bodies_to_propagate = ["olek", "marco"]

    # Define central bodies.
    central_bodies = ["Earth", "Earth"]

    # Define accelerations acting on Delfi-C3 by Sun and Earth.
    accelerations_settings_delfi_c3 = dict(
        Sun=[
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Earth=[
            propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
            propagation_setup.acceleration.aerodynamic()
        ],
        Moon=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
    )

    # Create global accelerations settings dictionary.
    acceleration_settings = {"olek": accelerations_settings_delfi_c3,
                             "marco": accelerations_settings_delfi_c3}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    ###########################################################################
    # CREATE PROPAGATION SETTINGS #############################################
    ###########################################################################

    # Set initial conditions for the Asterix satellite that will be
    # propagated in this simulation. The initial conditions are given in
    # Kepler elements and later on converted to Cartesian elements.
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
        gravitational_parameter=earth_gravitational_parameter,
        semi_major_axis=6378.0E3 + 500.0E3,
        eccentricity=0.0,
        inclination=np.deg2rad(97.4),
        argument_of_periapsis=np.deg2rad(235.7),
        longitude_of_ascending_node=np.deg2rad(23.4),
        true_anomaly=np.deg2rad(139.87)
    )
    initial_states = np.concatenate((initial_state, initial_state))

    # Define list of dependent variables to save.
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.total_acceleration("olek"),
        propagation_setup.dependent_variable.keplerian_state("olek", "Earth"),
        propagation_setup.dependent_variable.latitude("olek", "Earth"),
        propagation_setup.dependent_variable.longitude("olek", "Earth"),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "olek", "Sun"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.point_mass_gravity_type, "olek", "Moon"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.spherical_harmonic_gravity_type, "olek", "Earth"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.aerodynamic_type, "olek", "Earth"
        ),
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.cannonball_radiation_pressure_type, "olek", "Sun"
        ),
        propagation_setup.dependent_variable.keplerian_state("marco", "Earth"),
    ]


    # Create propagation settings.
    termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_states,
        termination_condition,
        output_variables=dependent_variables_to_save
    )
    # Create numerical integrator settings.
    initial_step_size = 10.0
    maximum_step_size = 100.0
    minimum_step_size = 1.0
    tolerance = 1.0E-10
    integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        initial_step_size,
        propagation_setup.integrator.RKCoefficientSets.rkf_78,
        minimum_step_size,
        maximum_step_size,
        tolerance,
        tolerance)

    ###########################################################################
    # PROPAGATE ORBIT #########################################################
    ###########################################################################

    # Create simulation object and propagate dynamics.
    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies, integrator_settings, propagator_settings)
    states = dynamics_simulator.state_history
    dependent_variables = dynamics_simulator.dependent_variable_history

    ###########################################################################
    # PLOT RESULTS    #########################################################
    ###########################################################################

    # By use of the dependent variable history, we can infer some interesting
    #  insight into the role that the various acceleration types play during
    #  the propagation, how the perturbers affect the kepler elements and what
    #  the ground track of Delfi-C3 would look like!

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    time = dependent_variables.keys()
    time_hours = [t / 3600 for t in time]

    states_list = np.vstack(list(states.values()))
    dependent_variable_list = np.vstack(list(dependent_variables.values()))

    # Plot Kepler elements as a function of time
    kepler_elements = dependent_variable_list[:, 3:9]
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 17))
    fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

    # Semi-major Axis
    semi_major_axis = [element / 1000 for element in kepler_elements[:, 0]]
    ax1.plot(time_hours, semi_major_axis)
    ax1.set_ylabel('Semi-major axis [km]')

    # Eccentricity
    eccentricity = kepler_elements[:, 1]
    ax2.plot(time_hours, eccentricity)
    ax2.set_ylabel('Eccentricity [-]')

    # Inclination
    inclination = [np.rad2deg(element) for element in kepler_elements[:, 2]]
    ax3.plot(time_hours, inclination)
    ax3.set_ylabel('Inclination [deg]')

    # Argument of Periapsis
    argument_of_periapsis = [np.rad2deg(element) for element in kepler_elements[:, 3]]
    ax4.plot(time_hours, argument_of_periapsis)
    ax4.set_ylabel('Argument of Periapsis [deg]')

    # Right Ascension of the Ascending Node
    raan = [np.rad2deg(element) for element in kepler_elements[:, 4]]
    ax5.plot(time_hours, raan)
    ax5.set_ylabel('RAAN [deg]')

    # True Anomaly
    true_anomaly = [np.rad2deg(element) for element in kepler_elements[:, 5]]
    ax6.scatter(time_hours, true_anomaly, s=1)
    ax6.set_ylabel('True Anomaly [deg]')
    ax6.set_yticks(np.arange(0, 361, step=60))

    for ax in fig.get_axes():
        ax.set_xlabel('Time [hr]')
        ax.set_xlim([min(time_hours), max(time_hours)])
        ax.grid()
    plt.figure(figsize=(17, 5))

    # Plot accelerations as a function of time

    # Point Mass Gravity Acceleration Sun
    acceleration_norm_pm_sun = dependent_variable_list[:, 11]
    plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')

    # Point Mass Gravity Acceleration Moon
    acceleration_norm_pm_moon = dependent_variable_list[:, 12]
    plt.plot(time_hours, acceleration_norm_pm_moon, label='PM Moon')

    # Spherical Harmonic Gravity Acceleration Earth
    acceleration_norm_sh_earth = dependent_variable_list[:, 13]
    plt.plot(time_hours, acceleration_norm_sh_earth, label='SH Earth')

    # Aerodynamic Acceleration Earth
    acceleration_norm_aero_earth = dependent_variable_list[:, 14]
    plt.plot(time_hours, acceleration_norm_aero_earth, label='Aerodynamic Earth')

    # Cannonball Radiation Pressure Acceleration Sun
    acceleration_norm_rp_sun = dependent_variable_list[:, 15]
    plt.plot(time_hours, acceleration_norm_rp_sun, label='Radiation Pressure Sun')

    plt.grid()

    plt.title("Accelerations norms on Delfi-C3, distinguished by type and origin, over the course of propagation.")
    plt.xlim([min(time_hours), max(time_hours)])
    plt.xlabel('Time [hr]')
    plt.ylabel('Acceleration Norm [m/s$^2$]')

    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.yscale('log')

    ##### REVOLV PLOTS ####
    from scipy import interpolate
    # Extract second satellite's keplerian elements
    sma_second = [sma / 1000 for sma in dependent_variable_list[:, 16]]
    # Compute average and trend of semi-major axis
    # Create time vector to evaluate values
    time_plot = np.linspace(simulation_start_epoch, simulation_end_epoch, 500)
    # In days
    time_days = [sec / 3600 / 24 for sec in time_plot]
    # Interpolate values
    interpolated_sma_func = interpolate.interp1d(list(time), semi_major_axis)
    ls = np.polyfit(list(time), semi_major_axis, 1)
    offset = ls[1]
    slope = ls[0]
    ls_2 = np.polyfit(list(time), sma_second, 1)
    offset_2 = ls_2[1]
    slope_2 = ls_2[0]
    # Create array of values
    trend = [offset + slope * el for el in list(time_plot)]
    trend_2 = [offset_2 + slope_2 * el for el in list(time_plot)]
    interpolated_sma = [interpolated_sma_func(el) for el in list(time_plot)]
    # Plot
    fig, ax = plt.subplots()
    ax.plot(time_days, interpolated_sma, label="Interpolated data", color="b")
    ax.plot(time_days, trend, label="Trend 1", color="r")
    ax.plot(time_days, trend_2, label="Trend 2", color="k")

    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Trend of semi-major axis [km]")
    ax.legend()


    # Compute angular separation between satellites
    states_1 = states_list[:, :6]
    states_2 = states_list[:, 6:]
    angular_sep = []
    for i in range(states_1.shape[0]):
        # Get scalar product
        scalar_product = np.dot(states_1[i, :3], states_2[i, :3])
        # Get costheta
        cos_theta = scalar_product / (np.linalg.norm(states_1[i, :3]) * np.linalg.norm(states_2[i, :3]))
        if cos_theta < -1.0:
            cos_theta = -1.0
        elif cos_theta > 1.0:
            cos_theta = 1.0
        # Get angle
        angle = np.arccos(cos_theta)
        angular_sep.append(angle)

    angular_sep = np.array(angular_sep)
    # Plot this quantity over time
    # Interpolate values
    interpolated_ang_func = interpolate.interp1d(list(time), angular_sep)
    ls = np.polyfit(list(time), angular_sep, 1)
    offset = ls[1]
    slope = ls[0]
    # Create array of values
    trend_ang = [offset + slope * el for el in list(time_plot)]
    interpolated_ang = [interpolated_ang_func(el) for el in list(time_plot)]
    # Plot
    fig, ax = plt.subplots()
    ax.plot(time_days, np.rad2deg(interpolated_ang), label="Interpolated data", color="b")
    ax.plot(time_days, np.rad2deg(trend_ang), label="Trend", color="r")

    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Trend of angular separation [deg]")
    plt.show()

    # Final statement (not required, though good practice in a __main__).
    return 0


if __name__ == "__main__":
    main()