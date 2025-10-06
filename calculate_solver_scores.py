import pandas as pd
import os


def calculate_scores(df, accuracy_weight=0.25, stability_weight=0.25, angular_accuracy_weight=0.25, efficiency_weight=0.25,
                     positional_acc_sub_weight=0.5, angular_acc_sub_weight=0.5,
                     time_sub_weight=0.4, pre_div_sub_weight=0.4, post_div_sub_weight=0.2):
    """
    Calculates normalized scores for accuracy and stability, then computes a
    weighted overall score for each solver.

    Args:
        df (pd.DataFrame): DataFrame with performance metrics.
        accuracy_weight (float): The weight for the combined accuracy score.
        accuracy_weight (float): The weight for the accuracy score.
        stability_weight (float): The weight for the stability score.
        angular_accuracy_weight (float): The weight for the angular accuracy score.
        efficiency_weight (float): The weight for the computation time score.
        time_sub_weight (float): Sub-weight for time before divergence within the stability score.
        pre_div_sub_weight (float): Sub-weight for pre-divergence RMSE within the stability score.
        post_div_sub_weight (float): Sub-weight for post-divergence RMSE within the stability score.

    Returns:
        pd.DataFrame: A DataFrame with the final averaged scores for each solver.
    """
    scores = []

    # Group data by the experimental run (angle) to score solvers against each other
    for angle, group in df.groupby('Initial Angle (deg)'):
        # --- Accuracy Score (lower Combined RMSE is better) ---
        min_rmse = group['Combined RMSE (m)'].min()
        max_rmse = group['Combined RMSE (m)'].max()  # Positional accuracy
        # Avoid division by zero if all solvers have the same RMSE
        if max_rmse == min_rmse:
            group['Accuracy Score'] = 10.0
        else:
            # Normalize so lowest RMSE gets 10, highest gets 0
            group['Accuracy Score'] = 10 * (1 - (group['Combined RMSE (m)'] - min_rmse) /
                                            (max_rmse - min_rmse))

        # --- Angular Accuracy Score (lower Angular RMSE is better) ---
        min_angular_rmse = group['Angular RMSE Bob 1 (rad)'].min()
        max_angular_rmse = group['Angular RMSE Bob 1 (rad)'].max()
        if max_angular_rmse == min_angular_rmse:
            group['Angular Accuracy Score'] = 10.0
        else:
            # Normalize so lowest angular RMSE gets 10, highest gets 0
            group['Angular Accuracy Score'] = 10 * (1 - (group['Angular RMSE Bob 1 (rad)'] - min_angular_rmse) /
                                                    (max_angular_rmse - min_angular_rmse))

        # --- Composite Stability Score ---
        # This score is a weighted average of three factors:
        # 1. Time before divergence (higher is better)
        # 2. Pre-divergence RMSE (lower is better)
        # 3. Post-divergence RMSE (lower is better, indicates less severe failure)

        # 1. Time Score
        min_time = group['Time before divergence'].min()
        max_time = group['Time before divergence'].max()
        time_score = 10.0 if max_time == min_time else 10 * \
            (group['Time before divergence'] -
             min_time) / (max_time - min_time)

        # 2. Pre-Divergence Accuracy Score
        min_pre_rmse = group['RMSE Bob 2 pre-divergence'].min()
        max_pre_rmse = group['RMSE Bob 2 pre-divergence'].max()
        pre_div_score = 10.0 if max_pre_rmse == min_pre_rmse else 10 * \
            (1 - (group['RMSE Bob 2 pre-divergence'] - min_pre_rmse) /
             (max_pre_rmse - min_pre_rmse))

        # 3. Post-Divergence Severity Score
        min_post_rmse = group['RMSE Bob 2 post-divergence'].min()
        max_post_rmse = group['RMSE Bob 2 post-divergence'].max()
        post_div_score = 10.0 if max_post_rmse == min_post_rmse else 10 * \
            (1 - (group['RMSE Bob 2 post-divergence'] - min_post_rmse) /
             (max_post_rmse - min_post_rmse))

        # Combine sub-scores into the final Stability Score
        group['Stability Score'] = (time_score * time_sub_weight +
                                    pre_div_score * pre_div_sub_weight +
                                    post_div_score * post_div_sub_weight)

        # --- Efficiency Score (lower Computation Time is better) ---
        min_comp_time = group['Computation Time (s)'].min()
        max_comp_time = group['Computation Time (s)'].max()
        if max_comp_time == min_comp_time:
            group['Efficiency Score'] = 10.0
        else:
            group['Efficiency Score'] = 10 * \
                (1 - (group['Computation Time (s)'] - min_comp_time) /
                 (max_comp_time - min_comp_time))

        scores.append(group)

    # Combine the scored groups back into a single DataFrame
    scored_df = pd.concat(scores)

    # Calculate the final weighted score for each run
    # First, create the combined Total Accuracy Score
    scored_df['Total Accuracy Score'] = (scored_df['Accuracy Score'] * positional_acc_sub_weight +
                                         scored_df['Angular Accuracy Score'] * angular_acc_sub_weight)

    # Then, calculate the final Overall Score
    scored_df['Overall Score'] = (scored_df['Total Accuracy Score'] * accuracy_weight +
                                  scored_df['Stability Score'] * stability_weight +
                                  scored_df['Efficiency Score'] * efficiency_weight)

    # --- Final Summary ---
    # Average the scores for each solver across all initial angles
    final_scores = scored_df.groupby('Solver')[[
        'Total Accuracy Score', 'Stability Score', 'Efficiency Score', 'Overall Score'
    ]].mean().round(2)

    # Sort by the final overall score to rank the solvers
    final_scores = final_scores.sort_values(
        'Overall Score', ascending=False)

    return final_scores


def main():
    """
    Main function to load data, calculate scores, and print the report.
    """
    # --- Configuration ---
    base_dir = "/Users/jacob/Double_pendulum/Data/Error_Analysis_Data"
    input_csv = os.path.join(base_dir, "RMSE_data.csv")

    # Define the mapping from row index groups to angles (4 solvers per group)
    angle_map = {0: 30, 1: 60, 2: 90, 3: 120, 4: 150}

    try:
        df = pd.read_csv(input_csv)
        if df.empty:
            print("Error: The RMSE data file is empty.")
            return

        # --- Pre-process Data ---
        # Assign an 'Initial Angle' to each row for grouping (3 solvers per angle)
        df['Initial Angle (deg)'] = (df.index // 3).map(angle_map)

        # Extract a clean 'Solver' name from the filename
        df['Solver'] = df['File Name'].str.split('_').str[0]
        df.loc[df['File Name'].str.contains('bs_adaptive_free'),
               'Solver'] = 'bs_adaptive_free'

        # --- Calculate and Display Scores ---
        # You can change the weights here if you value accuracy over stability, or vice-versa.
        # The positional and angular accuracy scores are combined into a single 'Total Accuracy Score'.
        final_scores = calculate_scores(
            df, accuracy_weight=0.4, stability_weight=0.2, efficiency_weight=0.4,
            positional_acc_sub_weight=0.5, angular_acc_sub_weight=0.5)

        print("\n--- Overall Solver Performance Ranking (Score out of 10) ---")
        print(final_scores)

        # --- Save the final scores to a CSV file ---
        output_path = os.path.join(base_dir, "Final_Scores.csv")
        try:
            final_scores.to_csv(output_path, float_format='%.2f')
            print(f"\nFinal scores saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving final scores to CSV: {e}")
        print("\nMethodology:")
        print(
            " - 'Total Accuracy Score' is a 50/50 blend of positional and angular accuracy.")
        print(" - 'Stability Score' is a composite of:")
        print("     - Time before divergence (40%, higher is better)")
        print("     - Pre-divergence RMSE (40%, lower is better)")
        print("     - Post-divergence RMSE (20%, lower is better)")
        print(
            " - 'Efficiency Score' is based on 'Computation Time' (lower is better).")
        print(
            " - 'Overall Score' is a weighted average: 40% Accuracy, 20% Stability, 40% Efficiency.")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        print("Please run 'run_batch_error_analysis.py' first to generate the data.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
