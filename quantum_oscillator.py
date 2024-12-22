''' Implementation of a discretized quantum simple harmonic oscillator. 
    Following implementation from - https://hef.ru.nl/~tbudd/mct/lectures/lattice_field_theory.html#equation-twopointcorrelator
'''
import numpy as np
import matplotlib.pylab as plt

# Initialize random number generator
rng = np.random.default_rng()

# Define lattice action and other functions

def lattice_action(x, omega_sq):
    '''Compute lattice action S_L[x] on state x.'''
    return 0.5 * ( np.dot(x - np.roll(x,1),x-np.roll(x,1)) + omega_sq*np.dot(x,x) )

def lattice_action_diff(x, t, newx, omega_sq):
    '''Compute change in action when x[t] is replaced by newx.'''
    w = len(x)
    return (x[t] - newx) * (x[(t-1)%w] + x[(t+1)%w] - (1 + omega_sq/2) * (newx + x[t]))

def oscillator_MH_step(x, omega_sq, delta):
    '''Perform Metropolis-Hastings update on uniform site with range delta.'''
    t = rng.integers(0, len(x))
    newx = x[t] + rng.uniform(-delta, delta)
    deltaS = lattice_action_diff(x, t, newx, omega_sq)
    if deltaS < 0 or rng.uniform() < np.exp(-deltaS):
        x[t] = newx
        return True
    return False

def run_oscillator_MH(x, omega_sq, delta, n, action_history=None):
    '''Perform n Metropolis-Hastings moves on state x.'''
    total_accept = 0
    for _ in range(n):
        if oscillator_MH_step(x, omega_sq, delta):
            total_accept += 1
        if action_history is not None:
            action_history.append(lattice_action(x, omega_sq))
    return total_accept

def two_point_function(states):
    '''Estimate two-point correlation <x(s+t)x(s)> from states (averaging over s).'''
    return np.array([np.mean(states * np.roll(states, t, axis=1)) for t in range(len(states[0]))])

# Parameters
time_length = 32
omegas = [0.4, 1.0, 1.5]
delta = 1.8 # chosen to have ~ 50% acceptance
equil_sweeps = 2000
measure_sweeps = 10
measurements = 500

# Initialize figure for plotting with 3 columns (one for each omega)
fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # Changed to 5 rows and 1 column for action history and two-point correlation

# Store action history and two-point correlation data for each omega
all_action_histories = []
all_two_point_funcs = []

# Run the simulation for each omega
for i, omega in enumerate(omegas):
    omega_sq = omega ** 2
    x_state = np.zeros(time_length)  # Initial state (all zeros)
    action_history = []

    # Perform equilibration sweeps (this is the "halfway" point)
    run_oscillator_MH(x_state, omega_sq, delta, equil_sweeps * time_length, action_history)
    halfway_state = x_state.copy()  # Capture the state after half the sweeps

    # Perform additional sweeps and measure final state
    run_oscillator_MH(x_state, omega_sq, delta, equil_sweeps * time_length, action_history)  # Continue from where it left off
    final_state = x_state.copy()  # Capture the final state

    # Plot the results for x (state at different stages) for each omega separately
    axes[i].plot(np.arange(time_length), np.zeros(time_length), label=f'Initial $\omega={omega}$', linestyle="--")
    axes[i].plot(np.arange(time_length), halfway_state, label=f'Halfway $\omega={omega}$')
    axes[i].plot(np.arange(time_length), final_state, label=f'Final $\omega={omega}$')

    # Collect action history for combined plot later
    all_action_histories.append(action_history)

    # Calculate and store two-point correlation function for combined plot later
    states = np.empty((measurements, time_length))
    for s in range(measurements):
        run_oscillator_MH(x_state, omega_sq, delta, measure_sweeps * time_length, action_history)
        states[s, :] = x_state

    two_point_fun = two_point_function(states)
    all_two_point_funcs.append(two_point_fun)

fig, axes2 = plt.subplots(2, 1, figsize=(12, 12))
# Plot Action History (combined for all omegas)
for i, action_history in enumerate(all_action_histories):
    axes2[0].plot(np.arange(len(action_history)), action_history, label=f"$\omega = {omegas[i]}$")
axes2[0].set_title("Action History")
axes2[0].set_xlabel("Sweep Number")
axes2[0].set_ylabel("Action $S_L$")
axes2[0].legend()

# Plot Two-Point Correlation Function (combined for all omegas)
for i, two_point_fun in enumerate(all_two_point_funcs):
    axes2[1].plot(np.arange(time_length), two_point_fun, label=f"$\\omega={omegas[i]}$")
axes2[1].set_title("Two-Point Correlation Function")
axes2[1].set_xlabel("$\\tau$")
axes2[1].set_ylabel(r"$\langle \hat{x}(i + \tau)\hat{x}(i)\rangle$")
axes2[1].legend()

# Customize x vs t plot labels (now specific for each omega)
axes[0].set_title("x vs t for different $\omega$")
axes[0].set_xlabel("Site (Time $\\tau$)")
axes[0].set_ylabel("State Value ($x_i$)")
axes[0].legend(loc='upper right')

axes[1].set_title("x vs t for different $\omega$")
axes[1].set_xlabel("Site (Time $\\tau$)")
axes[1].set_ylabel("State Value ($x_i$)")
axes[1].legend(loc='upper right')

axes[2].set_title("x vs t for different $\omega$")
axes[2].set_xlabel("Site (Time $\\tau$)")
axes[2].set_ylabel("State Value ($x_i$)")
axes[2].legend(loc='upper right')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
