import math
from scipy.special import ellipeinc
import numpy as np

G = 6.67430e-11

def orbital_radius(a, e, theta):
    """Radius at true anomaly theta (radians) for a Keplerian orbit."""
    return a * (1 - e**2) / (1 + e * math.cos(theta))

def orbital_parameters(mass_primary, mass_sat, r):
    """Gravitational force, circular orbital velocity, and escape velocity."""
    force = G * mass_primary * mass_sat / r**2
    v_orbital = math.sqrt(G * mass_primary / r)
    v_escape = math.sqrt(2 * G * mass_primary / r)
    return force, v_orbital, v_escape

def orbital_arc_length(a, b, theta1, theta2):
    """Arc length of the elliptical orbit between two eccentric anomalies."""
    e2 = 1 - (b / a)**2
    return a * (ellipeinc(theta2, e2) - ellipeinc(theta1, e2))


R_REFERENCE = 384400e3  # Moon's mean orbital radius (m)
mass_earth = 5.972e24


r_sat = float(input("Enter the distance of the satellite from Earth's center (m): "))
mass_sat = float(input("Enter the mass of the satellite (kg): "))
theta_deg = float(input("Enter the true anomaly in degrees: "))
theta = math.radians(theta_deg)


r_apoapsis = max(r_sat, R_REFERENCE)
r_periapsis = min(r_sat, R_REFERENCE)
a = (r_apoapsis + r_periapsis) / 2          # semi-major axis
e = (r_apoapsis - r_periapsis) / (r_apoapsis + r_periapsis)  # eccentricity
b = a * math.sqrt(1 - e**2)                 # semi-minor axis


r = orbital_radius(a, e, theta)


force, v_orb, v_esc = orbital_parameters(mass_earth, mass_sat, r)


arc = orbital_arc_length(a, b, 0, theta)

print(f"\nSemi-major axis:    {a:.2f} m")
print(f"Eccentricity:       {e:.6f}")
print(f"Radius at θ={theta_deg}°:  {r:.2f} m")
print(f"Gravitational force: {force:.4f} N")
print(f"Orbital velocity:    {v_orb:.2f} m/s")
print(f"Escape velocity:     {v_esc:.2f} m/s")
print(f"Arc length (0 to θ): {arc:.2f} m")

from scipy.optimize import minimize

# ---------------------------------------------------------------------
# 6. OPTIMIZATION FOR LAUNCH WINDOW & ENERGY
# ---------------------------------------------------------------------

def objective_function(params, optimization_target="energy"):
    """
    params[0]: v_p_scale -> scaling factor for initial velocity (around 1.0)
    params[1]: delta_theta -> adjustment to the initial Moon phase angle
    """
    v_p_scale, delta_theta = params
    
    # Apply parameterized initial conditions
    current_v_p = v_p * v_p_scale
    current_theta_moon0 = theta_moon0 + delta_theta
    
    # Temporary moon position function updated with new initial phase
    def temp_moon_position(t):
        ang = omega_m * t + current_theta_moon0
        return R_m * np.cos(ang), R_m * np.sin(ang)
    
    # Temporary Equations of Motion using the updated moon function
    def temp_eom(t, state):
        x, y, vx, vy = state
        r_e = np.hypot(x, y)
        xm, ym = temp_moon_position(t)
        dx, dy = x - xm, y - ym
        r_m = np.hypot(dx, dy)
        ax = -G * M_e * x / r_e**3 - G * M_m * dx / r_m**3
        ay = -G * M_e * y / r_e**3 - G * M_m * dy / r_m**3
        return [vx, vy, ax, ay]

    # Integrate the trajectory
    state0 = [r0, 0.0, 0.0, current_v_p]
    sol = solve_ivp(temp_eom, t_span, state0, t_eval=t_eval, rtol=1e-7, atol=1e-4)
    
    # Calculate tracking metrics
    x_sc, y_sc = sol.y[0], sol.y[1]
    xm_t, ym_t = temp_moon_position(sol.t)
    dist_to_moon = np.hypot(x_sc - xm_t, y_sc - ym_t)
    
    min_dist = np.min(dist_to_moon)
    idx_close = np.argmin(dist_to_moon)
    tof = sol.t[idx_close]  # Time of flight to closest approach
    
    # Penalty if the spacecraft misses the Moon entirely (e.g., misses by more than 5,000 km)
    miss_penalty = 0.0
    if min_dist > 5_000e3: 
        miss_penalty = (min_dist - 5_000e3) ** 2
        
    if optimization_target == "energy":
        # Minimize departure burn energy (v_p) + penalize missing the target
        delta_v_spent = abs(current_v_p - v_circ)
        return delta_v_spent + miss_penalty / 1e6
        
    elif optimization_target == "time":
        # Minimize time taken to reach the Moon + penalize missing the target
        return tof + miss_penalty / 100

# --- RUNNING THE OPTIMIZATION ---

# Initial guess: [1.0 (no change to v_p), 0.0 (no change to launch phase)]
initial_guess = [1.0, 0.0]

# Optimize for Energy (Delta-V)
res_energy = minimize(objective_function, initial_guess, args=("energy",), method="Nelder-Mead")

print("\n--- Optimization Results (Energy Minimal) ---")
print(f"Optimal Velocity Scale: {res_energy.x[0]:.4f} (New v_p: {v_p * res_energy.x[0] / 1e3:.2f} km/s)")
print(f"Optimal Launch Phase Shift: {np.degrees(res_energy.x[1]):.2f} degrees")
