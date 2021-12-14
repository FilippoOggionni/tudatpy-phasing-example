"""
Copyright (c) 2021. Revolv Space.

This file was adapted by TUDATPY EXAMPLE APPLICATION: Perturbed Satellite Orbit.

This script is the same as two_satellite_phasing.py, so refer to that description.

The only difference here is the DynamicPanelRotation class. Besides the constructor, this class has a method that
computes the drag acceleration due to the solar panels ONLY. This function is called at each integration step,
so all quantities used by the computation are retrieved from "within" the simulation (they change dynamically).
This scheme is useful to change the rotation of the panel at each time step (and, as an immediate consequence, the
drag surface). NOTE: for now, the rotation / area is set randomly within the [0, max_panel_area) interval.
This function is used to create a custom, self-exerted acceleration in the acceleration dict.

The script runs, but it should be validated.

TODO:
- start with a midnight-midday Sun-synchronous orbit as a baseline (this can be set by changing the RAAN accordingly)
- create methods inside DynamicPanelRotation to set the rotation angle of the panel with different techniques:
    1. sun pointing:
        A. retrieve the state of the Sun and of the satellite
        B. compute vector connecting Sun and satellite: this is the direction of maximum solar power
        C. compute the angle between direction computed in B and velocity direction (this is easy - simple cosine
        - for a midnight-midday orbit, but more complicated for a generic case because a projection is needed)
        D. compute drag accordingly
    2. ...

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


class DynamicPanelRotation:

    def __init__(self, bodies, panel_area, drag_coefficient, vehicle_name):

        self.bodies = bodies
        self.panel_area = panel_area
        self.drag_coefficient = drag_coefficient
        self.vehicle_name = vehicle_name

    def compute_drag_from_panel(self, time):

        # Get density
        density = self.bodies.get(self.vehicle_name).flight_conditions.density
        # Get velocity
        velocity_vector = self.bodies.get(self.vehicle_name).velocity
        # Get mass
        mass = self.bodies.get(self.vehicle_name).mass
        # Set the rotation angle of the panel (currently random)
        random_number = np.random.rand(1)
        # Get reference area
        area = random_number * self.panel_area
        # Compute drag for one panel only
        drag_one_panel = 0.5 * density * (np.linalg.norm(velocity_vector)) ** 2 * area / mass * self.drag_coefficient
        # Get normalized direction of drag
        drag_direction = - velocity_vector / np.linalg.norm(velocity_vector)
        # Compute drag vector from 2 panels
        drag_acceleration = 2 * drag_one_panel * drag_direction
        return drag_acceleration


def main():
    # Load spice kernels.
    spice_interface.load_standard_kernels()

    # Set simulation start and end epochs.
    simulation_start_epoch = 0.0
    simulation_end_epoch = constants.JULIAN_DAY * 1.

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

    # Create object with custom acceleration
    panel_area = 0.3 * 0.1
    drag_coefficient = 1.2
    panel_rotation_1 = DynamicPanelRotation(bodies, panel_area, drag_coefficient, "olek")
    panel_rotation_2 = DynamicPanelRotation(bodies, panel_area, drag_coefficient, "marco")

    # Define accelerations acting on Delfi-C3 by Sun and Earth.
    accelerations_settings_1 = dict(
        olek=[
            propagation_setup.acceleration.custom(panel_rotation_1.compute_drag_from_panel)
        ],
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

    accelerations_settings_2 = dict(
        marco=[
            propagation_setup.acceleration.custom(panel_rotation_2.compute_drag_from_panel)
        ],
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
    acceleration_settings = {"olek": accelerations_settings_1,
                             "marco": accelerations_settings_2}

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

    # Plot total acceleration as function of time
    total_acceleration_norm = np.linalg.norm(dependent_variable_list[:, 0:3], axis=1)
    plt.figure(figsize=(17, 5))
    plt.title("Total acceleration norm on Delfi-C3 over the course of propagation.")
    plt.plot(time_hours, total_acceleration_norm)
    plt.xlabel('Time [hr]')
    plt.ylabel('Total Acceleration [m/s$^2$]')
    plt.xlim([min(time_hours), max(time_hours)])
    plt.grid()

    # Plot ground track for a period of 3 hours
    latitude = dependent_variable_list[:, 9]
    longitude = dependent_variable_list[:, 10]
    hours = 3
    subset = int(len(time) / 24 * hours)
    latitude = np.rad2deg(latitude[0: subset])
    longitude = np.rad2deg(longitude[0: subset])
    plt.figure(figsize=(17, 5))
    plt.title("3 hour ground track of Delfi-C3")
    plt.scatter(longitude, latitude, s=1)
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.xlim([min(longitude), max(longitude)])
    plt.yticks(np.arange(-90, 91, step=45))
    plt.grid()

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