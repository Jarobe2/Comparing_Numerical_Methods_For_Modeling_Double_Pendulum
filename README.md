# Comparing_Numerical_Methods_For_Modeling_Double_Pendulum
Comparing Bulirsch-Stoer, GL8, and DOP853 for modeling the double pendulum. This work was done for an IB physics Extended Essay.

This project provides a comprehensive framework for simulating a double pendulum, comparing the simulation results against real-world experimental data, and evaluating the performance of different numerical integration methods.

The core objective is to quantify the accuracy, stability, and efficiency of three solvers:
1.  **DOP853**: A high-order Dormand-Prince method from the SciPy library.
2.  **GL8**: A custom implementation of an 8th-order Gauss-Legendre implicit Runge-Kutta method.
3.  **bs_adaptive_free**: A custom implementation of the Bulirsch-Stoer method with adaptive time stepping.

## Data Acquisition and Processing Workflow

The project follows a two-part workflow: first, acquiring and processing the experimental and simulated data, and second, running the analysis pipeline.

### Part 1: Data Acquisition

#### 1. Experimental Data Acquisition

The experimental data is sourced from video recordings of a physical double pendulum. The process is as follows:

1.  **Record a Video**: Film the double pendulum in motion.
2.  **Track Points**: Use a point-tracking model like **[CoTracker](https://github.com/facebookresearch/co-tracker)** (included in the `Cotracker/` directory) or my low cost version(Comparing-Iterative-Methods-for-the-Double-Pendulum/Cotracker/Low_cost_version) to extract the pixel coordinates of the two bobs for each frame of the video. This will typically produce a raw CSV file.
3.  **Convert and Process**: Use the `Cotracker/Pixels to Meters Converter.py` script to:
    *   Read the raw, interleaved pixel coordinates from the tracking software's output.
    *   Convert the coordinates from pixels to meters using a known scale factor (pixels per meter).
    *   The output is a processed CSV file (e.g., `30-processed.csv`) with coordinates in meters, ready for analysis.

#### 2. Simulation Data Generation

The simulated data is generated using the `run_batch_simulations.py` script.

*   This script runs the `Double_Pendulum_Model.py` for various initial angles (30, 60, 90, 120, 150 degrees) using each of the three numerical solvers.
*   It saves the raw simulation output (time, position coordinates) to CSV files in `Simulation_data/`.
*   It also logs the computation time for each simulation to `Simulation_data/computation_times.csv`.

### Part 2: Analysis Pipeline

The analysis is performed through a series of sequential steps, each handled by a dedicated batch script. The process flows as follows:

1.  **Align Simulation Data (`run_batch_alignment.py`)**
    *   Takes the raw simulation data and aligns its starting position with the corresponding experimental data file. This corrects for any offset in the physical setup or recording.
    *   Saves the newly aligned data to `Simulation_Data_Processed/` with an `-aligned.csv` suffix.

2.  **Calculate Errors (`run_batch_error_analysis.py`)**
    *   Compares each aligned simulation file against its corresponding experimental data.
    *   Synchronizes the datasets by interpolating the high-resolution simulation data onto the experimental timestamps.
    *   Calculates key performance metrics:
        *   Positional Root Mean Squared Error (RMSE) for each bob.
        *   Angular RMSE for the first bob.
        *   "Time before divergence," which marks the point where the simulation significantly deviates from the experiment.
    *   Appends all results to a master `Error_Analysis_Data/RMSE_data.csv` file.
    *   Merges the computation times into the final `RMSE_data.csv`.

3.  **Analyze and Score Results**
    *   **`summarize_rmse.py`**: Pivots the `RMSE_data.csv` into a human-readable summary table (`RMSE_comparison_summary.csv`), grouping results by initial angle for easy comparison.
    *   **`calculate_solver_scores.py`**: Reads the final `RMSE_data.csv` and calculates a normalized performance score (out of 10) for each solver based on weighted metrics for accuracy, stability, and efficiency. It outputs the final ranking to the console and saves it to `Final_Scores.csv`.

## How to Run the Full Analysis

To replicate the analysis from start to finish, run the batch scripts from your terminal in the following order:

```sh
# 1. Generate all simulation data
python run_batch_simulations.py

# 2. Align all simulation data with experimental data
python run_batch_alignment.py

# 3. Calculate error metrics for all aligned pairs
python run_batch_error_analysis.py

# 4. Generate the final performance scores and summary tables
python calculate_solver_scores.py
python summarize_rmse.py
```

## File Descriptions

### Core Simulation

*   **`Double_Pendulum_Model.py`**: The heart of the simulation.
    *   Defines the physical parameters and equations of motion for the double pendulum.
    *   Contains the implementations for the `GL8` and `bs_adaptive_free` solvers, and a wrapper for SciPy's `DOP853`.
    *   Can be run interactively to generate and plot a single simulation.

### Batch Processing Scripts

*   **`run_batch_simulations.py`**: Automates running `Double_Pendulum_Model.py` for a predefined set of initial conditions and solvers.
*   **`run_batch_alignment.py`**: Automates running `display_2_csv.py` to align all simulation files.
*   **`run_batch_error_analysis.py`**: Automates running `calculate_error.py` to perform error analysis on all aligned files.

### Analysis and Utility Scripts

*   **`display_2_csv.py`**: A utility for visualizing the path of a simulation against experimental data. Its primary function in the batch workflow is to perform the spatial alignment of the datasets.
*   **`calculate_error.py`**: The main error analysis engine. It handles timestamp synchronization, interpolation, and calculation of RMSE and divergence metrics. Can be run interactively for a single file pair.
*   **`summarize_rmse.py`**: A reporting script that transforms the raw `RMSE_data.csv` into a more readable, pivoted format.
*   **`calculate_solver_scores.py`**: The final analysis script that ranks the solvers based on a weighted scoring system across accuracy, stability, and efficiency.

### Experimental Data Utilities

*   **`Cotracker/Pixels to Meters Converter.py`**: A utility script for pre-processing raw experimental data. It reads coordinate data from a CSV file (e.g., from video tracking software), converts the units from pixels to meters, and can plot the resulting trajectory. This is used to prepare the experimental data before it is compared with simulation results.

## Data Directories

The scripts expect a specific directory structure. The base directory is hardcoded as `/Users/jacob/Double_pendulum/`.

*   **`Data/Experimental_Data_Processed/`**: Should contain the processed experimental data files (e.g., `30-processed.csv`).
*   **`Simulation_data/`**: Output directory for raw simulations from `run_batch_simulations.py`.
*   **`Data/Simulation_Data_Processed/`**: Output directory for aligned data from `run_batch_alignment.py`.
*   **`Data/Error_Analysis_Data/`**: Output directory for all analysis results (`RMSE_data.csv`, `RMSE_comparison_summary.csv`, `Final_Scores.csv`).

## Contributing

Contributions to this project are welcome. Please adhere to the standards outlined in the `CODE_OF_CONDUCT.md` to ensure a welcoming and inclusive environment for everyone.

*   **`Cotracker/CODE_OF_CONDUCT.md`**: Sets the standards for behavior and interaction within the community.

## Dependencies

*   Python 3
*   pandas
*   numpy
*   matplotlib
*   scipy
