import pandas as pd
import os
import io


def assign_angle_group(df):
    """
    Assigns an 'Initial Angle' to each row based on its index, assuming the
    data is sorted and grouped by angle in sets of 3 (one for each solver).
    """
    # Define the mapping from row index groups to angles
    angle_map = {
        0: 30,   # Rows 0-3
        1: 60,   # Rows 4-7
        2: 90,   # Rows 8-11
        3: 120,  # Rows 12-15
        4: 150,  # Rows 16-19
    }

    # With 3 solvers, integer division by 3 maps row indices (0,1,2) to group 0,
    # (3,4,5) to group 1, and so on.
    group_id = df.index // 3

    # Map the calculated group ID to the corresponding angle
    df['Initial Angle (deg)'] = group_id.map(angle_map)
    return df


def summarize_rmse_data(input_path, output_path):
    """
    Reads the detailed RMSE data and pivots it into a summary table
    for easier comparison of different simulation methods.

    Args:
        input_path (str): Path to the source RMSE_data.csv file.
        output_path (str): Path to save the summarized CSV file.
    """
    try:
        print(f"Reading detailed RMSE data from: {input_path}")
        df = pd.read_csv(input_path)

        if df.empty:
            print("The input file is empty. No summary to generate.")
            return

        # --- Assign Initial Angle to each row ---
        df = assign_angle_group(df)

        # Use an in-memory string buffer to build the final CSV content
        output_buffer = io.StringIO()

        print("\n--- Comparison Summary by Initial Angle ---")

        # Group by the newly assigned angle and process each group
        for angle, group in df.groupby('Initial Angle (deg)'):
            # --- Create a 'Solver' column from the 'File Name' ---
            # This extracts the method name (e.g., 'bs', 'GL8') from filenames
            # like 'bs_172.00s-aligned.csv'. It also handles the special case
            # of 'bs_adaptive_free'. This logic is now more robust.
            group['Solver'] = group['File Name'].str.split('_').str[0]
            group.loc[group['File Name'].str.contains(
                'bs_adaptive_free'), 'Solver'] = 'bs_adaptive_free'

            # --- Pivot the DataFrame for this group ---
            # Set the index to be the solver method, then transpose the table
            # so that solvers become columns and metrics become rows.
            summary_df = group.set_index('Solver').transpose()

            # Clean up the resulting table
            rows_to_drop = ['File Name', 'Initial Angle (deg)']
            summary_df = summary_df.drop(
                [row for row in rows_to_drop if row in summary_df.index])
            summary_df.index.name = 'Metric'

            # --- Print to console ---
            header = f"--- Initial Angle: {angle} degrees ---"
            print(header)
            print(summary_df)
            print("-" * (len(header)))

            # --- Write to buffer for CSV output ---
            output_buffer.write(f"Initial Angle (deg),{angle}\n")
            summary_df.to_csv(output_buffer, float_format='%.6f')
            output_buffer.write("\n")  # Add a blank line for separation

        # Write the complete buffer content to the output file
        with open(output_path, 'w') as f:
            f.write(output_buffer.getvalue())

        print(f"\nSuccessfully created summary file: {output_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # Define the input and output file paths
    base_dir = "/Users/jacob/Double_pendulum/Data/Error_Analysis_Data"
    input_csv = os.path.join(base_dir, "RMSE_data.csv")
    output_csv = os.path.join(base_dir, "RMSE_comparison_summary.csv")

    # Check if the source file exists before running
    if not os.path.exists(input_csv):
        print(f"Error: Source file not found at '{input_csv}'.")
        print("Please run 'calculate_error.py' first to generate the data.")
    else:
        summarize_rmse_data(input_csv, output_csv)
