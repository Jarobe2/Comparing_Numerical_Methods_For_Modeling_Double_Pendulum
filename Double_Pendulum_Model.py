# NOTE: The .py and .ipynb are equivalent, but it is HIGHLY recommended (For legibility) to use the .ipynb file.

## Library Imports ##
import warnings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag


def GL8_method(f, y0, t_array, args=(), max_iter=30, tol=1e-9):
    """
    Eighth-order Gauss-Legendre implicit Runge-Kutta integrator (4 stages) for ODEs.

    Parameters:
        f: function defining the system of ODEs (returns derivatives).
        y0: initial state vector.
        t_array: array of time points.
        args: additional arguments for f.
        max_iter (int): maximum number of fixed-point iterations per step. Increased to 30.
        tol (float): tolerance for convergence of the fixed-point iteration.

    Returns:
        y_out: array of states at each time point.
    """
    # 4-stage, 8th-order Gauss-Legendre coefficients (numerical values)
    # Source: https://github.com/JuliaGNI/RungeKutta.jl/blob/v0.5.16/src/tableaus/gauss.jl#L64-L74
    c = np.array([
        0.06943184420297371242, 0.33000947820757187099,
        0.66999052179242812901, 0.93056815579702628758
    ])
    b = np.array([
        0.17392742256872692871, 0.32607257743127307129,
        0.32607257743127307129, 0.17392742256872692871
    ])

    A = np.array([
        [0.0869637, -0.0266042,  0.0126275, -0.00355515],
        [0.188118,   0.163036,  -0.0278804,  0.0067355],
        [0.167192,   0.353953,   0.163036,  -0.0141907],
        [0.177483,   0.313445,   0.352677,   0.0869637]
    ])

    n = len(t_array)
    y_out = np.zeros((n, len(y0)))
    y_out[0] = y0

    for i in range(n - 1):
        h = t_array[i+1] - t_array[i]
        y_n = y_out[i]
        t_n = t_array[i]

        # Initial guess for stage derivatives K_i.
        # We use the derivative at the start of the step for all stages as a simple guess.
        K = np.array([f(y_n, t_n + c_j * h, *args) for c_j in c])

        # Fixed-point iteration to solve for implicit stages
        for _ in range(max_iter):
            K_old = K
            Y_stages = y_n + h * np.dot(A, K_old)
            K = np.array([f(Y_stages[j], t_n + c[j] * h, *args)
                         for j in range(4)])
            if np.linalg.norm(K - K_old) < tol:
                break
        else:
            # This block executes if the for loop completes without a 'break'.
            warnings.warn(
                f"GL8 fixed-point iteration did not converge at t={t_n:.4f} "
                f"within {max_iter} iterations. The result at this step may be inaccurate. "
                "Consider reducing the time step or increasing max_iter.", UserWarning)

        # Final update
        y_out[i+1] = y_n + h * np.dot(b, K)

    return y_out


def bs_adaptive_free(f, y0, t_span, t_eval, args=(), rtol=1e-9, atol=1e-12, safety=0.9, ifactor=5.0, dfactor=0.5, max_step=np.inf, initial_step=None):
    """
    Bulirsch-Stoer method with fully adaptive "free" stepping.

    Args:
        f (callable): Function defining the ODE system, f(t, y, *args).
        y0 (np.ndarray): Initial state vector.
        t_span (tuple): A tuple (t_start, t_end) for the integration interval.
        t_eval (np.ndarray): Array of time points where the solution is required (for dense output).
        args (tuple): Extra arguments to pass to f.
        rtol, atol (float): Relative and absolute error tolerances.
        safety (float): Safety factor for step size adjustment (e.g., 0.9).
        ifactor (float): Maximum factor to increase step size on success.
        dfactor (float): Factor to decrease step size on failure.
        max_step (float): Maximum allowed step size.
        initial_step (float, optional): Initial step size. If None, it's estimated.

    Returns:
        np.ndarray: Array of computed states at each time point in t_eval.
    """
    def _modified_midpoint(f_int, t0, y0, H, n, args_int):
        """
        Performs n steps of the modified midpoint method.
        Source: "Numerical Recipes in C" by Press et al., Section 16.3.
        This includes Gragg's smoothing step at the end.
        """
        h_sub = H / n
        y_mid = np.copy(y0)
        y_next = y_mid + h_sub * f_int(t0, y_mid, *args_int)
        for i in range(1, n):
            y_new = y_mid + 2 * h_sub * \
                f_int(t0 + i * h_sub, y_next, *args_int)
            y_mid = y_next
            y_next = y_new
        return 0.5 * (y_mid + y_next + h_sub * f_int(t0 + H, y_next, *args_int))

    def _bs_extrapolation_step(f_int, t_n, y_n, h, args_int):
        n_seq = [2, 4, 6, 8, 10, 12, 14, 16]
        k_max = len(n_seq)
        T = np.zeros((k_max, k_max, len(y_n)))
        T[0, 0] = _modified_midpoint(f_int, t_n, y_n, h, n_seq[0], args_int)
        for k in range(1, k_max):
            T[k, 0] = _modified_midpoint(
                f_int, t_n, y_n, h, n_seq[k], args_int)
            for j in range(1, k + 1):
                nk_ratio_sq = (n_seq[k] / n_seq[k-j])**2
                T[k, j] = T[k, j-1] + \
                    (T[k, j-1] - T[k-1, j-1]) / (nk_ratio_sq - 1.0)
            err_est = np.linalg.norm(T[k, k] - T[k-1, k-1])
            if k > 1 and err_est < np.linalg.norm(T[k-1, k-1] - T[k-2, k-2]):
                return T[k, k], T[k, k] - T[k-1, k-1]
        return T[k_max-1, k_max-1], T[k_max-1, k_max-1] - T[k_max-2, k_max-2]

    t_start, t_end = t_span
    y_out = np.zeros((len(t_eval), len(y0)))
    y_out[0] = y0
    t_current, y_current = t_start, np.copy(y0)
    y_prev = np.copy(y0)
    t_prev = t_start
    t_out_idx = 1

    def _cubic_hermite_interp(y0, y1, f0, f1, h, theta):
        """Cubic Hermite interpolation for dense output."""
        h00 = 2 * theta**3 - 3 * theta**2 + 1
        h10 = theta**3 - 2 * theta**2 + theta
        h01 = -2 * theta**3 + 3 * theta**2
        h11 = theta**3 - theta**2
        return h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1

    if initial_step is None:
        f0 = f(t_current, y_current, *args)
        scale = atol + np.abs(y_current) * rtol
        d0 = np.linalg.norm(y_current / scale)
        d1 = np.linalg.norm(f0 / scale)
        h = min(max_step, 0.01 * (d0 / d1) if d1 > 1e-5 else 1e-6)
    else:
        h = min(max_step, initial_step)

    while t_current < t_end:
        h = min(h, t_end - t_current)
        while True:
            y_next, error_est = _bs_extrapolation_step(
                f, t_current, y_current, h, args)
            scale = atol + np.maximum(np.abs(y_current), np.abs(y_next)) * rtol
            err_ratio = np.linalg.norm(error_est / scale) / np.sqrt(len(y0))
            if err_ratio <= 1.0:
                # Derivative at the start of the step
                f_prev = f(t_current, y_current, *args)
                t_prev = t_current
                y_prev = y_current
                t_current += h
                y_current = y_next
                # Derivative at the end of the step
                f_curr = f(t_current, y_current, *args)

                # Dense output: interpolate solution for requested time points in the step
                while t_out_idx < len(t_eval) and t_eval[t_out_idx] <= t_current:
                    theta = (t_eval[t_out_idx] - t_prev) / h
                    # Use cubic Hermite interpolation for smoother, more accurate dense output.
                    y_out[t_out_idx] = _cubic_hermite_interp(
                        y_prev, y_current, f_prev, f_curr, h, theta)
                    t_out_idx += 1

                h = min(max_step, safety * h * (1.0 / err_ratio) **
                        0.25 if err_ratio > 1e-8 else h * ifactor)
                break
            else:
                h = max(h * dfactor, safety * h * (1.0 / err_ratio)**0.25)
                if t_current + h <= t_current:
                    raise RuntimeError(
                        "Bulirsch-Stoer step size became too small.")
    return y_out


def dormand_prince_method(f, y0, t_array, args=(), method='DOP853', rtol=1e-6, atol=1e-12):
    """Numerical integration using Dormand-Prince (RK45 or DOP853) via scipy.solve_ivp."""
    sol = solve_ivp(
        fun=f,
        t_span=(t_array[0], t_array[-1]),
        y0=y0,
        t_eval=t_array,
        method=method,
        rtol=rtol,
        atol=atol
    )
    return sol.y.T


def double_pendulum(theta1_init, theta2_init, method_name, sim_time, rtol=1e-6, atol=1e-12):
    """
    Simulates the double pendulum system and returns the positions of the two masses.
    Expects the initial angular position for both in radians (theta_1_0, theta_2_0)
    Expects the numerical method to use (Options are RK4 and Euler)"""
    # Expects the initial angular position for both in radians (theta_1_0, theta_2_0)
    # Expects the numerical method to use (Options are RK4 and Euler)
    # Expects the total time to be simulated (t_final)

    # Simple Double Pendulum System - Configuration
    m1 = 0.044                     # mass 1 [kg]
    m2 = 0.006                    # mass 2 [kg]
    L1 = 0.0925                      # wire length of mass 1 [m]
    L2 = 0.07                    # wire length of mass 2 [m]
    g = 9.807                   # gravitational acceleration [m/s²]

    # Simple Double Pendulum System - Initial Values
    omega1_init = 0               # Angular velocity of mass 1 [rad/s]
    omega2_init = 0               # Angular velocity of mass 2 [rad/s]
    y0 = [theta1_init, omega1_init, theta2_init, omega2_init]

    t0 = 0                  # Initial time [seconds]
    # Desired time increment for fixed-step solvers. Reduced for better convergence.
    dt = 0.0001
    n_steps = int(sim_time / dt) + 1  # Number of steps
    t_array = np.linspace(t0, sim_time, n_steps, endpoint=True)

    # --- Start Timer ---
    start_time = time.perf_counter()

    def sds(t, x, m1, m2, L1, L2, g):
        """System of differential equations for the double pendulum."""
        # x[0] = theta_1 (Position, mass 1)
        # x[1] = omega_1 (Angular velocity, mass 1)
        # x[2] = theta_2 (Position, mass 2)
        # x[3] = omega_2 (Angular velocity, mass 2)
        # t is time, but it's not used in these equations (time-invariant system)

        theta1, omega1, theta2, omega2 = x

        dydt = np.zeros_like(x)
        delta_theta = theta1 - theta2
        c = np.cos(delta_theta)
        s = np.sin(delta_theta)

        dydt[0] = omega1
        dydt[2] = omega2

        # This implementation now algebraically matches the formulation you provided.
        # Note: sin(theta2-theta1) = -s and cos(theta2-theta1) = c

        # Denominator term, using the identity (m1+m2) - m2*cos^2(delta) = m1 + m2*sin^2(delta)
        den1 = L1 * (m1 + m2 * s**2)
        den2 = L2 * (m1 + m2 * s**2)

        # Numerator for the angular acceleration of bob 1 (omega1_dot)
        num1 = m2 * L1 * omega1**2 * (-s)*c + m2 * g * np.sin(
            theta2) * c + m2 * L2 * omega2**2 * (s) - (m1+m2) * g * np.sin(theta1)
        dydt[1] = num1 / den1

        # Numerator for the angular acceleration of bob 2 (omega2_dot)
        num2 = -m2 * L2 * omega2**2 * \
            (s)*c + (m1+m2) * (g * np.sin(theta1) * c -
                               L1 * omega1**2 * (s) - g * np.sin(theta2))
        dydt[3] = num2 / den2

        return dydt

    # --- Solver Dispatcher ---
    sds_args = (m1, m2, L1, L2, g)
    method_lower = method_name.lower()

    if method_lower == "gl8":
        # The GL8_method expects a function with the signature f(y, t, *args).
        # The sds function has the signature sds(t, y, *args).
        # This lambda correctly wraps sds to match what GL8_method expects.
        gl8_dynamics_wrapper = lambda y, t, *a: sds(t, y, *a)
        S = GL8_method(gl8_dynamics_wrapper, y0, t_array, args=sds_args)
    elif method_lower == "bs_adaptive_free":
        # This method returns both t and S, so we handle it differently.
        S = bs_adaptive_free(lambda t, y, *a: sds(t, y, *a),
                             y0, (t0, sim_time), t_eval=t_array, args=sds_args, rtol=rtol, atol=atol)
    elif method_lower in ["dop853"]:
        # Map user-friendly names to scipy's method names
        scipy_method_map = {
            "dop853": "DOP853",
        }  # type: ignore
        scipy_method = scipy_method_map.get(method_lower, method_lower)
        sol = solve_ivp(sds, (t0, sim_time), y0, method=scipy_method,
                        t_eval=t_array, args=sds_args, rtol=rtol, atol=atol)
        S = sol.y.T
    else:
        raise ValueError(
            "Method must be one of: 'GL8', 'bs_adaptive_free', 'DOP853'")

    # --- End Timer ---
    computation_time = time.perf_counter() - start_time

    theta1 = S[:, 0]     # theta_1 (Position, mass 1)
    theta2 = S[:, 2]     # theta_2 (Position, mass 2)

    # Converting polar coordinates to Cartesian
    # x_n, y_n → horizontal and vertical positions (mm)

    # Pendulum's mass #1
    x1 = L1*np.sin(theta1)
    y1 = -L1*np.cos(theta1)

    # Pendulum's mass #2
    x2 = x1 + L2*np.sin(theta2)
    y2 = y1 - L2*np.cos(theta2)

    return t_array, x1, y1, x2, y2, computation_time


def save_simulation_data(t_array, x0, y0, x1, y1, x2, y2, method_name, sim_time, computation_time):
    """Saves the simulation data to a CSV file."""
    # Define the directory and ensure it exists
    output_dir = os.path.join(os.path.dirname(
        __file__), "Simulation_data")
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize method_name for filename and create the full path
    # Replace spaces and dashes with underscores for a cleaner filename
    sanitized_method_name = method_name.replace(" ", "_").replace("-", "_")
    filename = f"{sanitized_method_name}_{sim_time:.2f}s.csv"
    filepath = os.path.join(output_dir, filename)

    # Combine the data for writing. The columns will be: time, x0, y0, x1, y1, x2, y2
    data_to_save = np.column_stack((t_array, x0, y0, x1, y1, x2, y2))
    header = "time,x0,y0,x1,y1,x2,y2"

    # Save to CSV using numpy, which is efficient for this kind of data
    np.savetxt(filepath, data_to_save, delimiter=",",
               header=header, comments='')
    print(f"\nSimulation data saved to: {filepath}")

    # --- Log Computation Time ---
    # Save the computation time to a separate log file for analysis.
    time_log_path = os.path.join(output_dir, "computation_times.csv")
    time_log_data = {
        "File Name": filename,
        "Computation Time (s)": computation_time
    }
    time_df = pd.DataFrame([time_log_data])
    file_exists = os.path.exists(time_log_path)
    time_df.to_csv(time_log_path, mode='a',
                   header=not file_exists, index=False, float_format='%.4f')
    print(
        f"Computation time of {computation_time:.4f}s logged to: {time_log_path}")


def plot_double_pendulum_path(theta1_init, theta2_init, method_name, sim_time, x_origin=0.0, y_origin=0.0, rtol=1e-6, atol=1e-12, save_data=False, show_plot=True):
    """Plots the path of the double pendulum and saves the data."""
    # Simulate the double pendulum. The returned coordinates are relative to a (0,0) pivot.
    t_array, x1_rel, y1_rel, x2_rel, y2_rel, computation_time = double_pendulum(
        theta1_init, theta2_init, method_name, sim_time, rtol=rtol, atol=atol)

    # Apply the origin offset to get absolute coordinates
    x1_abs = x1_rel + x_origin
    y1_abs = y1_rel + y_origin
    x2_abs = x2_rel + x_origin
    y2_abs = y2_rel + y_origin

    # The pivot point's coordinates are now the specified origin
    x0_abs = np.full_like(t_array, x_origin)
    y0_abs = np.full_like(t_array, y_origin)

    # Save the absolute coordinates to the CSV file if requested
    if save_data:
        save_simulation_data(t_array, x0_abs, y0_abs, x1_abs,
                             y1_abs, x2_abs, y2_abs, method_name, sim_time, computation_time)

    if show_plot:
        plt.figure(figsize=(8, 8), dpi=150)
        plt.plot(x1_abs, y1_abs, label='Bob 1 Path', color='red')
        plt.plot(x2_abs, y2_abs, label='Bob 2 Path', color='blue')
        plt.scatter([x_origin], [y_origin], color='black', label='Pivot Point')
        plt.title(f'Double Pendulum Path ({method_name})')
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')
        plt.plot([x1_abs[0]], [y1_abs[0]], marker='o', color='green',
                 label='Start Bob 1', zorder=5)
        plt.plot([x2_abs[0]], [y2_abs[0]], marker='o', color='green',
                 label='Start Bob 2', zorder=5)
        plt.plot([x1_abs[-1]], [y1_abs[-1]], marker='o', color='yellow',
                 label='End Bob 1', zorder=5)
        plt.plot([x2_abs[-1]], [y2_abs[-1]], marker='o', color='yellow',
                 label='End Bob 2', zorder=5)
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    print("Double Pendulum Path Plotter")
    try:
        theta1_input = float(
            input("Enter initial angle for bob #1 (in radians): "))
        theta2_input = float(
            input("Enter initial angle for bob #2 (in radians): "))
        method_input = input(
            "Enter numerical method (e.g. GL8, DOP853, bs_adaptive_free): ").strip()
        sim_time_input = float(input("Enter simulation time (in seconds): "))

        # Ask for tolerance settings
        # Default tolerances match scipy's defaults
        rtol_input = 1e-6
        atol_input = 1e-12
        if method_input.lower() in ["dop853", "bs", "bs_adaptive", "bs_adaptive_free"]:
            tol_choice = input(
                "Specify custom tolerance for the solver? (y/n) [default: n, rtol=1e-6, atol=1e-12]: ").lower().strip()
            if tol_choice == 'y':
                rtol_input = float(
                    input("Enter relative tolerance (e.g., 1e-6): "))
                atol_input = float(
                    input("Enter absolute tolerance (e.g., 1e-12): "))

        # Ask whether to save the data
        save_data_choice = input(
            "Save simulation data to a CSV file? (y/n) [default: n]: ").lower().strip() == 'y'

        plot_double_pendulum_path(theta1_input, theta2_input, method_input,
                                  sim_time_input,
                                  x_origin=0.0,
                                  y_origin=0.0,
                                  rtol=rtol_input, atol=atol_input, save_data=save_data_choice)
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"Error: {e}")
