import csv
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


def read_interleaved_coordinates(filepath, x_col='x', y_col='y', max_rows=None):
    """
    Reads interleaved x and y coordinates for two paths from a CSV file.
    The file is expected to have rows alternating between path 1 and path 2.

    Args:
        filepath (str): The path to the CSV file.
        x_col (str): The name of the column containing x-coordinates.
        y_col (str): The name of the column containing y-coordinates.
        max_rows (int, optional): The maximum number of data rows to read.
                                  If None, all rows are read. Defaults to None.

    Returns:
        tuple: A tuple containing four lists (x1, y1, x2, y2),
               or (None, None, None, None) if an error occurs.
    """
    x1_coords, y1_coords = [], []
    x2_coords, y2_coords = [], []

    try:
        with open(filepath, mode='r', newline='') as csvfile:
            # Use DictReader to easily access columns by name
            reader = csv.DictReader(csvfile)

            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break

                try:
                    x = float(row[x_col])
                    y = float(row[y_col])
                    if i % 2 == 0:  # Bob 1 on even rows (0, 2, 4...)
                        x1_coords.append(x)
                        y1_coords.append(y)
                    else:  # Bob 2 on odd rows (1, 3, 5...)
                        x2_coords.append(x)
                        y2_coords.append(y)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping malformed row {row}. Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None, None

    return x1_coords, y1_coords, x2_coords, y2_coords


def plot_multiple_paths(paths_data, title='Paths from CSV', save_path=None, unit='pixels'):
    """
    Plots one or more paths on the same figure.

    Args:
        paths_data (list): A list of dictionaries, where each dict contains
                           'x_coords', 'y_coords', and 'label'.
        title (str): The title for the plot.
        unit (str): The unit of measurement for the axes (e.g., 'pixels', 'meters').
        save_path (str, optional): If provided, saves the plot to this file path.
    """
    if not paths_data:
        print("Cannot plot: No path data provided.")
        return

    # Check if any path has actual data to prevent errors with empty lists
    if not any(path.get('x_coords') for path in paths_data):
        print("Cannot plot: All path data is empty.")
        return

    plt.figure(figsize=(10, 8))
    colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'plum']
    legend_elements = []

    for i, path in enumerate(paths_data):
        x_coords = path.get('x_coords')
        y_coords = path.get('y_coords')
        label = path.get('label', f'Path {i+1}')
        color = colors[i % len(colors)]

        if not x_coords or len(x_coords) != len(y_coords):
            print(f"Warning: Skipping invalid path data for '{label}'.")
            continue

        # Plot the line connecting the points
        line, = plt.plot(x_coords, y_coords, color=color,
                         linestyle='-', linewidth=2, label=label)
        legend_elements.append(line)

        # Highlight the start and end points
        plt.plot(x_coords[0], y_coords[0], 'o', color='green', markersize=10)
        plt.plot(x_coords[-1], y_coords[-1], 'o', color='red', markersize=10)

    # Adding labels and title for clarity
    plt.title(title, fontsize=16)
    plt.xlabel(f'X-Coordinate ({unit})', fontsize=12)
    plt.ylabel(f'Y-Coordinate ({unit})', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axis('equal')

    # --- Set specific grid spacing when using meters ---
    if unit == 'meters':
        ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.02))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))

    # Add a consolidated legend for paths and markers
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', label='Start Point',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='End Point',
               markerfacecolor='red', markersize=10)
    ])
    plt.legend(handles=legend_elements)

    # Invert the y-axis to match the coordinate system of most video/image
    # formats where the origin is at the top-left corner.
    plt.gca().invert_yaxis()

    # Save or display the plot
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot successfully saved to '{save_path}'")
        except Exception as e:
            print(f"\nError saving plot: {e}")
    else:
        plt.show()


def main():
    """
    Main function to run the script.
    """
    # --- Step 1: Get user input for file path and column names ---
    default_path = "/Users/jacob/Double pendulum/Double Pendulum Positions 1.csv"
    csv_filepath_input = input(
        f"Enter path to CSV file (or press Enter for default: {default_path}): ")
    csv_filepath = csv_filepath_input or default_path

    if not os.path.exists(csv_filepath):
        print(
            f"\nError: The file '{csv_filepath}' was not found. Please check the path and try again.")
        return

    x_col_name = input(
        "Enter the name of the X-coordinate column (default: 'x'): ") or 'x'
    y_col_name = input(
        "Enter the name of the Y-coordinate column (default: 'y'): ") or 'y'
    path_labels = ['Pendulum 1 Path', 'Pendulum 2 Path']

    # --- NEW: Get scale factor from user ---
    pixels_per_meter = None
    unit = 'pixels'
    try:
        scale_input = input(
            "\nEnter the number of pixels per meter (or press Enter to plot in pixels): "
        )
        if scale_input:
            pixels_per_meter = float(scale_input)
            if pixels_per_meter <= 0:
                print("Pixels per meter must be positive. Plotting in pixels.")
                pixels_per_meter = None
            else:
                unit = 'meters'
    except ValueError:
        print("Invalid input. Plotting in pixels.")

    # --- Step 2: Ask user for the number of rows to read ---
    num_rows_to_read = None
    try:
        rows_input = input(
            "Enter the number of rows to read (or press Enter to read all): "
        )
        if rows_input:
            num_rows_to_read = int(rows_input)
            if num_rows_to_read <= 0:
                print("Number of rows must be positive. Reading all rows.")
                num_rows_to_read = None
    except ValueError:
        print("Invalid input. Reading all rows.")

    if num_rows_to_read:
        print(
            f"\nReading first {num_rows_to_read} rows from '{csv_filepath}'...")
    else:
        print(f"\nReading all time-series data from '{csv_filepath}'...")

    # --- Step 3: Read interleaved coordinates ---
    x1, y1, x2, y2 = read_interleaved_coordinates(
        csv_filepath, x_col=x_col_name, y_col=y_col_name, max_rows=num_rows_to_read
    )

    # --- NEW: Convert coordinates if scale is provided ---
    if pixels_per_meter and x1 is not None:
        print(
            f"\nConverting coordinates from pixels to meters (1m = {pixels_per_meter}px)...")
        x1 = [p / pixels_per_meter for p in x1]
        y1 = [p / pixels_per_meter for p in y1]
        x2 = [p / pixels_per_meter for p in x2]
        y2 = [p / pixels_per_meter for p in y2]

    # --- Step 4: Plot the paths if any data was read successfully ---
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        print(f"  - Found {len(x1)} points for Pendulum 1.")
        print(f"  - Found {len(x2)} points for Pendulum 2.")

        if not x1 and not x2:
            print(
                "\nWarning: No data points were read. Check column names and file content.")
            return

        paths_to_draw = [
            {'x_coords': x1, 'y_coords': y1, 'label': path_labels[0]},
            {'x_coords': x2, 'y_coords': y2, 'label': path_labels[1]},
        ]

        print("\nData read successfully. Preparing plot...")

        # Ask user if they want to save the plot
        save_input = input("Save plot to a file? (y/n, default: n): ").lower()
        save_path = None
        if save_input == 'y':
            save_path = "pendulum_trajectory.png"
            print(f"Plot will be saved as '{save_path}'")

        plot_multiple_paths(
            paths_to_draw, title='Double Pendulum Trajectory', save_path=save_path, unit=unit)
    else:
        print("\nFailed to read any valid coordinate data. Exiting.")


if __name__ == "__main__":
    main()
