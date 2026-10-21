
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# VEHICLE SUSPENSION SYSTEM SIMULATION
# Quarter-car model
# ==========================================

# Vehicle parameters
ms = 300.0       # Sprung mass (kg)
mu = 40.0        # Unsprung mass (kg)
ks = 15000.0     # Suspension stiffness (N/m)
cs = 1500.0      # Damping coefficient (N.s/m)
kt = 200000.0    # Tire stiffness (N/m)

# Simulation settings
t_start = 0.0
t_end = 5.0
dt = 0.001
time = np.arange(t_start, t_end + dt, dt)

# Road bump: 5 cm high, lasting 0.2 seconds
def road_profile(t):
    if 1.0 <= t <= 1.2:
        return 0.05
    return 0.0

# State vector:
# x[0] = body displacement (m)
# x[1] = body velocity (m/s)
# x[2] = wheel displacement (m)
# x[3] = wheel velocity (m/s)

def suspension_model(t, x):
    body_pos, body_vel, wheel_pos, wheel_vel = x

    r = road_profile(t)

    # Suspension spring and damper force
    suspension_force = (
        ks * (body_pos - wheel_pos)
        + cs * (body_vel - wheel_vel)
    )

    # Equations of motion
    body_acc = -suspension_force / ms

    wheel_acc = (
        suspension_force - kt * (wheel_pos - r)
    ) / mu

    return [
        body_vel,
        body_acc,
        wheel_vel,
        wheel_acc
    ]

# Initial conditions: vehicle at rest
initial_state = [0.0, 0.0, 0.0, 0.0]

# Solve differential equations
solution = solve_ivp(
    suspension_model,
    (t_start, t_end),
    initial_state,
    t_eval=time,
    method="RK45",
    max_step=dt,
    rtol=1e-7,
    atol=1e-9
)

# Extract results
t = solution.t
body = solution.y[0]
wheel = solution.y[2]

road = np.array([road_profile(ti) for ti in t])
suspension_travel = body - wheel

# ==========================================
# PLOT GRAPHS
# ==========================================

plt.figure(figsize=(10, 9))

plt.subplot(3, 1, 1)
plt.plot(t, road * 1000, "k--", linewidth=2)
plt.ylabel("Road height (mm)")
plt.title("Vehicle Suspension System Simulation")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, body * 1000, "b", label="Vehicle body")
plt.plot(t, wheel * 1000, "r", label="Wheel")
plt.ylabel("Displacement (mm)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, suspension_travel * 1000, "m")
plt.xlabel("Time (seconds)")
plt.ylabel("Suspension travel (mm)")
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("VEHICLE SUSPENSION SIMULATION RESULTS")
print("-------------------------------------")
print(f"Maximum body displacement: "
      f"{np.max(np.abs(body))*1000:.3f} mm")

print(f"Maximum wheel displacement: "
      f"{np.max(np.abs(wheel))*1000:.3f} mm")

print(f"Maximum suspension travel: "
      f"{np.max(np.abs(suspension_travel))*1000:.3f} mm")
