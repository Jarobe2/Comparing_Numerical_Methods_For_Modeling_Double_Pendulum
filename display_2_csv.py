#To align experimental and simulated data sets

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the dataframe by renaming the time column to 't'
    and sorting by time.
    """
    # The simulation file might use 'time', while others might use 't'.
    # We standardize on 't' for internal processing.
    if 'time' in df.columns and 't' not in df.columns:
        df.rename(columns={'time': 't'}, inplace=True)

    # Ensure data is sorted by time to correctly identify start/end points.
    if 't' in df.columns:
        df.sort_values(by='t', inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def plot_comparison(simulation_filepath, experimental_filepath, max_time=None, examine=False, align_start=False, save_aligned=False, show_plot=True):
    """
    Loads, optionally examines, and plots simulation and experimental data
    for the double pendulum on the same graph, centered on a common origin.

    Args:
        simulation_filepath (str): Path to the simulation data CSV.
        experimental_filepath (str): Path to the experimental data CSV.
        max_time (float, optional): The maximum time in seconds to plot.
                                    If None, all data is plotted.
        examine (bool): If True, prints the first few rows of each dataset.
        align_start (bool): If True, aligns the simulation's start to the experimental start.
        save_aligned (bool): If True and align_start is True, saves the aligned data.
        show_plot (bool): If True, displays the plot. Set to False for batch processing.
    """
    try:
        # Load the datasets using pandas
        print(f"Loading simulation data from: {simulation_filepath}")
        sim_df = pd.read_csv(simulation_filepath)
        print(f"Loading experimental data from: {experimental_filepath}")
        exp_df = pd.read_csv(experimental_filepath)

        # Standardize column names and sort by time for both dataframes
        sim_df = _standardize_dataframe(sim_df)
        exp_df = _standardize_dataframe(exp_df)

        if examine:
            print("\n--- Examining Simulation Data (first 5 rows) ---")
            print(sim_df.head())
            print("\n--- Examining Experimental Data (first 5 rows) ---")
            print(exp_df.head())
            print("-" * 50)

        # --- Optional: Align starting positions ---
        # This shifts the entire simulation dataset so that its starting
        # position for bob 1 matches the experimental data's start.
        if align_start:
            # Get the initial positions (t=0) for both datasets' first bob
            exp_start_x1 = exp_df['x1'].iloc[0]
            exp_start_y1 = exp_df['y1'].iloc[0]
            sim_start_x1 = sim_df['x1'].iloc[0]
            sim_start_y1 = sim_df['y1'].iloc[0]

            # Calculate the offset needed
            offset_x = exp_start_x1 - sim_start_x1
            offset_y = exp_start_y1 - sim_start_y1

            print(f"\nAligning simulation data to experimental start point.")
            print(
                f"Calculated offset (x, y): ({offset_x:.4f}, {offset_y:.4f})")

            # Apply the same fixed offset to all simulation coordinates (including pivot)
            cols_to_shift_x = [
                col for col in ['x0', 'x1', 'x2'] if col in sim_df.columns]
            cols_to_shift_y = [
                col for col in ['y0', 'y1', 'y2'] if col in sim_df.columns]
            sim_df[cols_to_shift_x] = sim_df[cols_to_shift_x] + offset_x
            sim_df[cols_to_shift_y] = sim_df[cols_to_shift_y] + offset_y

            # --- Save the aligned data to a new file if requested ---
            if save_aligned:
                try:
                    # Define the target directory and ensure it exists.
                    output_dir = "/Users/jacob/Double_pendulum/Data/Simulation_Data_Processed"
                    os.makedirs(output_dir, exist_ok=True)

                    # Create the new filename by adding '-aligned' to the base name.
                    original_filename = os.path.basename(simulation_filepath)
                    base, ext = os.path.splitext(original_filename)

                    # To prevent creating names like '...-aligned-aligned.csv',
                    # we first remove any existing '-aligned' suffix before adding it.
                    base = base.removesuffix('-aligned')
                    aligned_filename = f"{base}-aligned{ext}"
                    aligned_filepath = os.path.join(
                        output_dir, aligned_filename)

                    # Create a copy and revert the time column name for consistency
                    df_to_save = sim_df.copy()
                    if 't' in df_to_save.columns:
                        df_to_save.rename(columns={'t': 'time'}, inplace=True)

                    # Save to CSV, preserving precision
                    df_to_save.to_csv(aligned_filepath,
                                      index=False, float_format='%.6f')
                    print(
                        f"Aligned simulation data saved to: {aligned_filepath}")
                except Exception as e:
                    print(
                        f"\nWarning: Could not save aligned data file. Error: {e}")

        # --- Determine the time range for plotting ---
        # The plot should only cover the time range available in the experimental data.
        exp_end_time = exp_df['t'].max()
        analysis_end_time = exp_end_time

        # If the user specified a max_time, use it if it's less than the experimental duration.
        if max_time is not None:
            analysis_end_time = min(exp_end_time, max_time)
            print(
                f"\nUser-defined max time is {max_time}s. Plotting up to {analysis_end_time:.2f}s.")
        else:
            print(
                f"\nPlotting up to the end of experimental data at {analysis_end_time:.2f}s.")

        sim_df = sim_df[sim_df['t'] <= analysis_end_time].copy()
        exp_df = exp_df[exp_df['t'] <= analysis_end_time].copy()

        if sim_df.empty or exp_df.empty:
            print("One or both dataframes are empty after filtering. Cannot plot.")
            return

        # Print the number of rows being plotted
        print(f"Plotting {len(sim_df)} rows from simulation file.")
        print(f"Plotting {len(exp_df)} rows from experimental file.")

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        # Plot Experimental Data (Solid Lines)
        ax.plot(exp_df['x1'], exp_df['y1'], color='red',
                linestyle='-', label='Experimental Bob 1')
        ax.plot(exp_df['x2'], exp_df['y2'], color='blue',
                linestyle='-', label='Experimental Bob 2')

        # Plot Modeled Data (Dashed Lines)
        ax.plot(sim_df['x1'], sim_df['y1'], color='salmon',
                linestyle='--', label='Modeled Bob 1')
        ax.plot(sim_df['x2'], sim_df['y2'], color='skyblue',
                linestyle='--', label='Modeled Bob 2')

        # --- Mark Start and End Points ---
        # Use green for start points and orange for end points.

        # Experimental Start/End Points
        ax.plot(exp_df['x1'].iloc[0], exp_df['y1'].iloc[0], 'o', color='limegreen',
                markersize=8, label='Start Exp Bob 1', zorder=5, markeredgecolor='k')
        ax.plot(exp_df['x2'].iloc[0], exp_df['y2'].iloc[0], 's', color='limegreen',
                markersize=8, label='Start Exp Bob 2', zorder=5, markeredgecolor='k')
        ax.plot(exp_df['x1'].iloc[-1], exp_df['y1'].iloc[-1], 'o', color='orange',
                markersize=8, label='End Exp Bob 1', zorder=5, markeredgecolor='k')
        ax.plot(exp_df['x2'].iloc[-1], exp_df['y2'].iloc[-1], 's', color='orange',
                markersize=8, label='End Exp Bob 2', zorder=5, markeredgecolor='k')

        # Modeled Start/End Points
        ax.plot(sim_df['x1'].iloc[0], sim_df['y1'].iloc[0], 'X', color='limegreen',
                markersize=8, label='Start Sim Bob 1', zorder=5, markeredgecolor='k')
        ax.plot(sim_df['x2'].iloc[0], sim_df['y2'].iloc[0], 'P', color='limegreen',
                markersize=8, label='Start Sim Bob 2', zorder=5, markeredgecolor='k')
        ax.plot(sim_df['x1'].iloc[-1], sim_df['y1'].iloc[-1], 'X', color='orange',
                markersize=8, label='End Sim Bob 1', zorder=5, markeredgecolor='k')
        ax.plot(sim_df['x2'].iloc[-1], sim_df['y2'].iloc[-1], 'P', color='orange',
                markersize=8, label='End Sim Bob 2', zorder=5, markeredgecolor='k')

        # --- Plot Formatting ---
        ax.set_title(
            'Comparison of Experimental and Modeled Double Pendulum Paths')
        ax.set_xlabel('X-position (meters)')
        ax.set_ylabel('Y-position (meters)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True)

        if show_plot:
            print("\nDisplaying plot. Close the plot window to exit.")
            plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find a file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """Prompts the user for options and runs the comparison plot."""
    base_dir = '/Users/jacob/Double_pendulum/'
    default_sim_file = os.path.join(
        base_dir, 'Simulation_data/rk4_10.00s.csv')

    print("--- Pendulum Data Comparison Plotter Setup ---")

    # Prompt for simulation file
    sim_prompt = f"Enter path to simulation CSV [default: {default_sim_file}]: "
    simulation_filepath = input(sim_prompt).strip() or default_sim_file

    # Prompt for the experimental file
    default_exp_dir = os.path.join(base_dir, 'Error_Analysis_Data')
    exp_prompt = (
        f"Enter path to processed experimental CSV [searches in {default_exp_dir}]: ")
    experimental_filepath = input(exp_prompt).strip() or ""

    # Prompt for max_time
    max_time_str = input(
        "Enter maximum time to plot (in seconds) [default: all]: ").strip()
    max_time = None
    if max_time_str:
        try:
            max_time = float(max_time_str)
        except ValueError:
            print(
                f"Invalid input for time '{max_time_str}'. Plotting all data.")
            max_time = None

    # Prompt for boolean options
    examine = input(
        "Examine data (print first 5 rows)? (y/n) [default: n]: ").lower().strip() == 'y'
    align_start = input(
        "Align start positions before plotting? (y/n) [default: n]: ").lower().strip() == 'y'

    save_aligned = False
    if align_start:
        save_aligned = input(
            "Save the aligned simulation data to a new CSV file? (y/n) [default: n]: ").lower().strip() == 'y'

    print("-" * 40)

    # If a relative path was given for the experimental file, join it with the default directory.
    if experimental_filepath and not os.path.isabs(experimental_filepath):
        experimental_filepath = os.path.join(
            default_exp_dir, experimental_filepath)

    # Check if files exist before attempting to plot
    if not os.path.exists(simulation_filepath):
        print(f"Error: Simulation file not found at: {simulation_filepath}")
    elif not experimental_filepath or not os.path.exists(experimental_filepath):
        print(
            f"Error: Experimental data file not found at: {experimental_filepath}")
    else:
        plot_comparison(
            simulation_filepath,
            experimental_filepath,
            max_time,
            examine,
            align_start,
            save_aligned=save_aligned,
            show_plot=True  # Always show plot in interactive mode
        )


if __name__ == '__main__':
    main()
