import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


def calculate_interpolation_error(sim_df_orig):
    """_
    Performs cross-validation on the simulation data to estimate the error
    introduced by the interpolation process.

    Args:
        sim_df_orig (pd.DataFrame): The original, high-resolution simulation dataframe.
    """
    print("\n--- Cross-Validating Interpolation Error ---")
    sim_df = sim_df_orig.copy()
    if 'time' in sim_df.columns and 't' not in sim_df.columns:
        sim_df = sim_df.rename(columns={'time': 't'})

    # The simulation is much denser than the experiment. We'll hold out
    # some simulation points, interpolate across the gaps, and see how
    # well the interpolated values match the held-out ground truth.
    # The experimental data is ~16x sparser, so holding out every 10th
    # point is a reasonable proxy for the interpolation task.
    holdout_step = 10

    # Create the ground truth validation set
    ground_truth_df = sim_df.iloc[::holdout_step].copy()

    # Create the training set by dropping the validation points
    training_df = sim_df.drop(ground_truth_df.index)

    # To predict values at the validation timestamps, we reindex the
    # training data to include all original timestamps. This creates NaNs
    # at the validation points, which we can then interpolate.
    combined_index = sim_df['t'].sort_values().unique()
    reindexed_training_df = training_df.set_index('t').reindex(combined_index)

    # Interpolate to fill the gaps
    interpolated_df = reindexed_training_df.interpolate(method='index')

    # Get the predicted values at the holdout timestamps
    predicted_df = interpolated_df.loc[ground_truth_df['t']]

    # The interpolation can't create values at the start or end of the
    # dataset, which results in NaNs. We need to drop these rows from
    # both the ground truth and the predictions to ensure a fair comparison.
    predicted_df = predicted_df.dropna()

    # Filter the ground_truth_df to only include timestamps that
    # were successfully interpolated.
    ground_truth_df = ground_truth_df[ground_truth_df['t'].isin(
        predicted_df.index)]

    if predicted_df.empty or ground_truth_df.empty:
        print("Could not perform interpolation validation. Not enough data points.")
        return np.nan, np.nan

    # Calculate the error between predicted and ground truth
    error_bob1 = np.sqrt((ground_truth_df['x1'].values - predicted_df['x1'].values)**2 + (
        ground_truth_df['y1'].values - predicted_df['y1'].values)**2)
    error_bob2 = np.sqrt((ground_truth_df['x2'].values - predicted_df['x2'].values)**2 + (
        ground_truth_df['y2'].values - predicted_df['y2'].values)**2)

    rmse_bob1 = np.sqrt(np.mean(error_bob1**2))
    rmse_bob2 = np.sqrt(np.mean(error_bob2**2))

    print("Method: Held out every 10th simulation point and interpolated its value.")
    print(f"Estimated Interpolation RMSE for Bob 1: {rmse_bob1:.6f} meters")
    print(f"Estimated Interpolation RMSE for Bob 2: {rmse_bob2:.6f} meters")
    print("This is an estimate of the error introduced by the timestamp synchronization process itself.")
    print("------------------------------------------")
    return rmse_bob1, rmse_bob2


def merge_computation_times(rmse_filepath, times_filepath):
    """
    Merges computation times into the main RMSE results file.

    This function reads the existing RMSE data and the computation time log,
    then performs a left merge to add the 'Computation Time (s)' column.

    Args:
        rmse_filepath (str): Path to the RMSE_data.csv file.
        times_filepath (str): Path to the computation_times.csv file.
    """
    try:
        if not os.path.exists(rmse_filepath) or not os.path.exists(times_filepath):
            print(
                "Warning: Cannot merge computation times. One or both files are missing.")
            return

        print("\n--- Merging Computation Times into RMSE Data ---")
        rmse_df = pd.read_csv(rmse_filepath)
        times_df = pd.read_csv(times_filepath)

        # Prepare the key for merging. The RMSE file has an '-aligned' suffix
        # that needs to be removed to match the timing log.
        rmse_df['merge_key'] = rmse_df['File Name'].str.replace(
            '-aligned.csv', '.csv')
        times_df['merge_key'] = times_df['File Name']

        # Perform a left merge to add the computation time to the RMSE data
        merged_df = pd.merge(
            rmse_df,
            times_df[['merge_key', 'Computation Time (s)']],
            on='merge_key',
            how='left'
        )

        # Clean up and save
        merged_df.drop(columns=['merge_key'], inplace=True)
        merged_df.to_csv(rmse_filepath, index=False, float_format='%.6f')
        print(f"Successfully merged computation times into: {rmse_filepath}")

    except Exception as e:
        print(f"An error occurred during computation time merge: {e}")


def calculate_and_plot_error(simulation_filepath, experimental_filepath, max_time=None, plot_error=False, validate_interpolation=False, plot_rmse_intervals=False, interval_duration=5.0, analyze_stabilization=False, log_results_to_csv=False):
    """

    Calculates and optionally plots the Root Mean Squared Error (RMSE) between
    experimental and simulated double pendulum data.

    Args:
        simulation_filepath (str): Path to the simulation data CSV.
        experimental_filepath (str): Path to the experimental data CSV.
        max_time (float, optional): The maximum time in seconds to analyze.
        plot_error (bool): If True, generates a plot of the error over time.
        validate_interpolation (bool): If True, runs a cross-validation check on interpolation error.
        plot_rmse_intervals (bool): If True and plot_error is True, plots RMSE over time intervals.
        interval_duration (float): The duration for each RMSE interval.
        analyze_stabilization (bool): If True, runs the error stabilization analysis.
        log_results_to_csv (bool): If True, appends the results to a CSV file.
    """
    try:
        # --- 1. Load Data ---
        print(f"Loading simulation data from: {simulation_filepath}")
        sim_df = pd.read_csv(simulation_filepath)
        print(f"Loading experimental data from: {experimental_filepath}")
        exp_df = pd.read_csv(experimental_filepath)

        # Standardize time column names
        if 'time' in sim_df.columns and 't' not in sim_df.columns:
            sim_df = sim_df.rename(columns={'time': 't'})
        if 'time' in exp_df.columns and 't' not in exp_df.columns:
            exp_df = exp_df.rename(columns={'time': 't'})

        # --- Report on loaded data ---
        print(
            f"\nSimulation data time range: {sim_df['t'].min():.2f}s to {sim_df['t'].max():.2f}s ({len(sim_df)} points)")
        print(
            f"Experimental data time range: {exp_df['t'].min():.2f}s to {exp_df['t'].max():.2f}s ({len(exp_df)} points)")

        # Filter data based on the time option
        if max_time is not None:
            print(f"\nFiltering data to analyze up to {max_time} seconds.")
            sim_df = sim_df[sim_df['t'] <= max_time].copy()
            exp_df = exp_df[exp_df['t'] <= max_time].copy()
            if sim_df.empty or exp_df.empty:
                print(
                    "One or both dataframes are empty after filtering. Cannot proceed.")
                return

        # --- Optional: Cross-Validate Interpolation Error ---
        interp_rmse_bob1 = np.nan
        interp_rmse_bob2 = np.nan
        if validate_interpolation:
            interp_rmse_bob1, interp_rmse_bob2 = calculate_interpolation_error(
                sim_df)

        # --- 3. Synchronize Timestamps ---
        # We will use the experimental timestamps as the reference and interpolate
        # the simulation data to match. This is a robust way to compare datasets
        # with different time steps.
        print("\nSynchronizing timestamps by finding the overlapping time range...")
        sim_df.set_index('t', inplace=True)
        exp_df.set_index('t', inplace=True)

        # Ensure indices are sorted for correct interpolation
        sim_df.sort_index(inplace=True)
        exp_df.sort_index(inplace=True)

        # Manually add suffixes before joining to ensure all columns are correctly labeled.
        # This prevents issues where columns without a name conflict (like x0, y0)
        # do not get the suffix applied by the join operation.
        exp_df = exp_df.add_suffix('_exp')
        sim_df = sim_df.add_suffix('_sim')

        # Combine dataframes, aligning on the index (time)
        # An outer join is used to find the common time range of both datasets.
        # This creates NaNs where one dataset has timestamps the other doesn't.
        combined_df = exp_df.join(sim_df, how='outer')

        # Interpolate the simulation data to fill the NaNs
        # This creates a synchronized dataset at all unique timestamps.
        combined_df.interpolate(method='index', inplace=True)

        # Drop any remaining rows with NaNs (e.g., at the very start or end)
        # This effectively limits the comparison to the intersecting time range.
        combined_df.dropna(inplace=True)

        if combined_df.empty:
            print(
                "Error: After synchronization, the datasets have no overlapping time points.")
            return

        print(
            f"Successfully synchronized data. Comparing {len(combined_df)} points from {combined_df.index.min():.2f}s to {combined_df.index.max():.2f}s.")

        # --- 4. Calculate Errors ---
        # Calculate the Euclidean distance error for each bob at each time step
        error_bob1 = np.sqrt(
            (combined_df['x1_exp'] - combined_df['x1_sim'])**2 +
            (combined_df['y1_exp'] - combined_df['y1_sim'])**2
        )
        error_bob2 = np.sqrt(
            (combined_df['x2_exp'] - combined_df['x2_sim'])**2 +
            (combined_df['y2_exp'] - combined_df['y2_sim'])**2
        )

        # Calculate the Root Mean Squared Error (RMSE) for each bob
        rmse_bob1 = np.sqrt(np.mean(error_bob1**2))
        rmse_bob2 = np.sqrt(np.mean(error_bob2**2))

        # Calculate a combined RMSE for the whole system
        total_squared_error = np.concatenate([error_bob1**2, error_bob2**2])
        rmse_combined = np.sqrt(np.mean(total_squared_error))

        # --- Calculate Angular Errors ---
        # Calculate the angle of the first bob relative to the simulation's pivot point
        # for both datasets to ensure a consistent reference frame.
        # The y-axis is inverted (-(y - y_pivot)) to match the standard mathematical
        # convention where theta=0 is along the positive x-axis.
        combined_df['theta1_exp'] = np.arctan2(
            combined_df['x1_exp'] - combined_df['x0_sim'], -(combined_df['y1_exp'] - combined_df['y0_sim']))
        combined_df['theta1_sim'] = np.arctan2(
            combined_df['x1_sim'] - combined_df['x0_sim'], -(combined_df['y1_sim'] - combined_df['y0_sim']))

        error_theta1 = combined_df['theta1_exp'] - combined_df['theta1_sim']
        rmse_theta1 = np.sqrt(np.mean(error_theta1**2))

        # --- Find Error Stabilization Point (only if plotting raw error) ---
        stabilization_time = np.nan
        if analyze_stabilization:
            print("\n--- Analyzing Error Stabilization ---")
            # Define the error threshold for stabilization
            stabilization_threshold_m = 0.1

            # Find the first time point where the raw error for Bob 2 exceeds the threshold.
            # This is a more direct and robust definition of the divergence point.
            divergence_points = error_bob2[error_bob2 >
                                           stabilization_threshold_m]

            if not divergence_points.empty:
                # The stabilization_time is the timestamp of the first such point.
                stabilization_time = divergence_points.index[0]
                print(
                    f"Divergence point found at {stabilization_time:.2f} seconds.")
                print(
                    f"(Criteria: First time point where raw error > {stabilization_threshold_m}m)")
            else:
                print(
                    f"No point found where raw error exceeds {stabilization_threshold_m}m. The simulation remained closely aligned.")

        # --- 5. Output Results ---
        print("\n--- Error Analysis Results ---")
        print(f"RMSE for Bob 1: {rmse_bob1:.6f} meters")
        print(f"RMSE for Bob 2: {rmse_bob2:.6f} meters")
        print(f"Combined RMSE:  {rmse_combined:.6f} meters")
        print(f"Angular RMSE for Theta 1: {rmse_theta1:.6f} radians")

        # Initialize stabilization metrics to NaN
        rmse_bob2_pre_stabilization = np.nan
        rmse_bob2_post_stabilization = np.nan

        # Calculate and print pre-stabilization RMSE if a point was found
        if not np.isnan(stabilization_time):
            pre_stabilization_error_slice = error_bob2.loc[:stabilization_time]
            if not pre_stabilization_error_slice.empty:
                rmse_bob2_pre_stabilization = np.sqrt(
                    np.mean(pre_stabilization_error_slice**2))
                print(
                    f"RMSE for Bob 2 (pre-stabilization): {rmse_bob2_pre_stabilization:.6f} meters (up to {stabilization_time:.2f}s)")
            post_stabilization_error_slice = error_bob2.loc[stabilization_time:]
            if not post_stabilization_error_slice.empty:
                rmse_bob2_post_stabilization = np.sqrt(
                    np.mean(post_stabilization_error_slice**2))
                print(
                    f"RMSE for Bob 2 (post-stabilization): {rmse_bob2_post_stabilization:.6f} meters (from {stabilization_time:.2f}s)")

        print("------------------------------")

        # --- 6. Save Results to CSV ---
        if log_results_to_csv:
            try:
                # Define the target directory and ensure it exists.
                output_dir = "/Users/jacob/Double_pendulum/Data/Error_Analysis_Data"
                results_csv_path = os.path.join(output_dir, "RMSE_data.csv")

                results_data = {
                    "File Name": os.path.basename(simulation_filepath),
                    "RMSE Bob 1 (m)": rmse_bob1,
                    "RMSE Bob 2 (m)": rmse_bob2,
                    "Combined RMSE (m)": rmse_combined,
                    "Angular RMSE Bob 1 (rad)": rmse_theta1,
                    "Time before divergence": stabilization_time,
                    "RMSE Bob 2 pre-divergence": rmse_bob2_pre_stabilization,
                    "RMSE Bob 2 post-divergence": rmse_bob2_post_stabilization,
                    "Interpolation Error Bob 1": interp_rmse_bob1,  # Keep these last
                    "Interpolation Error Bob 2": interp_rmse_bob2
                }

                new_row_df = pd.DataFrame([results_data])
                file_exists = os.path.exists(results_csv_path)
                new_row_df.to_csv(results_csv_path, mode='a',
                                  header=not file_exists, index=False, float_format='%.6f')
                print(f"Results appended to {results_csv_path}")
            except Exception as e:
                print(f"Warning: Could not save results to CSV. Error: {e}")

        # --- 7. Plot Error Over Time (Optional) ---
        if plot_error:
            if plot_rmse_intervals:
                # --- Calculate and Plot Rolling RMSE ---
                print(
                    f"\nCalculating and plotting RMSE over {interval_duration}s intervals...")
                end_time = combined_df.index.max()

                rmse_interval_data = []
                # Setup the plot outside the loop
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
                ax.set_title(
                    f'Bob 2 RMSE ({interval_duration}s Intervals) vs. Time')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('RMSE (meters)')
                ax.grid(True)

                has_plotted_anything = False

                # Iterate through time intervals from t=0 to cover the full range.
                current_time = 0.0
                while current_time < end_time:
                    t_start = current_time
                    t_end = current_time + interval_duration

                    # Select the squared errors within the current time interval [t_start, t_end)
                    mask = (error_bob2.index >= t_start) & (
                        error_bob2.index < t_end)
                    interval_squared_errors = error_bob2[mask]**2

                    if not interval_squared_errors.empty:
                        # Calculate RMSE for this interval
                        interval_rmse = np.sqrt(
                            np.mean(interval_squared_errors))
                        rmse_interval_data.append(
                            (t_start, t_end, interval_rmse))

                        # Plot a horizontal line for the interval's RMSE value.
                        # Add a label only for the first line segment to avoid clutter in the legend.
                        label = 'Bob 2 Interval RMSE' if not has_plotted_anything else ""
                        ax.plot([t_start, t_end], [interval_rmse, interval_rmse],
                                color='green', linestyle='-', label=label)
                        has_plotted_anything = True

                    current_time = t_end

                if rmse_interval_data:
                    print("\n--- RMSE per Interval ---")
                    for start, end, rmse in rmse_interval_data:
                        print(
                            f"Time [{start:.2f}s - {end:.2f}s]: RMSE = {rmse:.6f} meters")
                    print("-------------------------")

                if has_plotted_anything:
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
                else:
                    print("Warning: Not enough data to generate an interval RMSE plot.")
            else:
                # --- Plot Raw Positional Error ---
                print("\nGenerating raw error plot...")
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

                ax.plot(combined_df.index, error_bob2,
                        label='Bob 2 Positional Error', color='blue', alpha=0.8)

                # Plot the stabilization point if it was found
                if not np.isnan(stabilization_time):
                    ax.axvline(x=stabilization_time, color='gold', linestyle='--', linewidth=2.5,
                               label=f'Stabilization Point ({stabilization_time:.2f}s)')

                ax.set_title('Bob 2 Positional Error vs. Time')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Error (meters)')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """Prompts the user for options and runs the error calculation."""
    base_dir = '/Users/jacob/Double pendulum/Data'
    default_sim_file = os.path.join(
        base_dir, 'Simulation_data/rk4_10.00s.csv')
    default_exp_file = os.path.join(
        base_dir, 'Experimental-Data-Processed', '30-processed.csv')

    print("--- Pendulum Error Calculation Setup ---")

    # Prompt for simulation file
    sim_prompt = f"Enter path to simulation CSV [default: {default_sim_file}]: "
    simulation_filepath = input(sim_prompt).strip() or default_sim_file

    # Prompt for experimental file
    exp_prompt = f"Enter path to experimental CSV [default: {default_exp_file}]: "
    experimental_filepath = input(exp_prompt).strip() or default_exp_file

    # Prompt for max_time
    max_time_str = input(
        "Enter maximum time to analyze (in seconds) [default: all]: ").strip()
    max_time = None
    if max_time_str:
        try:
            max_time = float(max_time_str)
        except ValueError:
            print(
                f"Invalid input for time '{max_time_str}'. Analyzing all data.")
            max_time = None

    # Prompt for options
    plot_error = input(
        "Generate a plot of error over time? (y/n) [default: n]: ").lower().strip() == 'y'
    plot_rmse_intervals = False
    interval_duration = 5.0  # Define for use in the prompt below
    if plot_error:
        plot_rmse_intervals = input(
            f"Plot RMSE over {interval_duration}s instead of raw error? (y/n) [default: n]: ").lower().strip() == 'y'
    # Always run the stabilization analysis, but still allow user to validate interpolation.
    analyze_stabilization = True
    validate_interpolation = input(
        "Validate interpolation error? (y/n) [default: n]: ").lower().strip() == 'y'
    log_results_to_csv = input(
        "Log results to RMSE_data.csv? (y/n) [default: n]: ").lower().strip() == 'y'

    print("-" * 40)

    # Check if files exist before attempting to run
    if not os.path.exists(simulation_filepath):
        print(f"Error: Simulation file not found at: {simulation_filepath}")
    elif not os.path.exists(experimental_filepath):
        print(
            f"Error: Experimental data file not found at: {experimental_filepath}")
    else:
        calculate_and_plot_error(
            simulation_filepath,
            experimental_filepath,
            max_time=max_time,
            plot_error=plot_error,
            validate_interpolation=validate_interpolation,
            plot_rmse_intervals=plot_rmse_intervals,
            interval_duration=interval_duration,
            analyze_stabilization=analyze_stabilization,
            log_results_to_csv=log_results_to_csv
        )


if __name__ == '__main__':
    main()
