#Written by Imre Vida
#Email: imre.vida@mail.polimi.it
#2025.10.29

import numpy as np

def generate_measured_signals(control_signals):
    """
    Generate 3 measured signal vectors based on 5 input control signals.

    Parameters:
        control_signals (ndarray): 2D array of shape (5, len(time)) representing the control signals.

    Returns:
        list: 3 measured signal vectors.
    """
    # Define Constants and Parameters
    circle_radius = 7.0  # Updated to 7 cm
    fiber_density = 0.7

    # Define positions and sizes of 5 points
    points = [
        {'position': (3.5, 3.5), 'size': 5},
        {'position': (-3.5, 3.5), 'size': 5},
        {'position': (1.75, 1.75), 'size': 6},
        {'position': (-1.75, 1.75), 'size': 6},
        {'position': (0.0, 3.5), 'size': 5}
    ]

    # Define measurement points on the circle at angles -15, 0, and 15 degrees
    angles_deg = [-15, 0, 15]
    angles_deg = [angle + 90 for angle in angles_deg]  # Adjust angles by 90 degrees
    angles_rad = np.deg2rad(angles_deg)

    measurement_points = [
        (circle_radius * np.cos(angle), circle_radius * np.sin(angle))
        for angle in angles_rad
    ]

    # Set up a time vector: 1 seconds with millisecond intervals
    time = np.arange(0, 1.001, 0.001)  # From 0 to 1 seconds, step 0.001s (1 ms)
    num_signals = control_signals.shape[0]
    # Generate random vectors for each slice (10 slices along z)
    #stretch control signal to match time length even if control signal is shorter
    control_signals_stretched = np.zeros((num_signals, len(time)))
    
    for i in range(num_signals):
        control_signals_stretched[i] = np.interp(time, np.linspace(0, 1, control_signals.shape[1]), control_signals[i])
    control_signals = control_signals_stretched
    num_signals = control_signals.shape[0]
    num_slices = 10
    slice_random_vectors = np.random.uniform(0, 1, (num_signals, num_slices, len(time)))

    # Update thresholds based on control_signals
    dynamic_thresholds = np.where(control_signals == 1, 0.7, 0.15)
    dynamic_thresholds_reshaped = dynamic_thresholds[:, None, :]

    # Apply thresholding principle for each slice, over time
    slice_binary_signals = (slice_random_vectors < dynamic_thresholds_reshaped).astype(int)

    # Apply moving average to each slice and each signal over time
    window_size = 3
    slice_moving_averages = np.zeros_like(slice_binary_signals)
    for i in range(num_signals):
        for j in range(num_slices):
            slice_moving_averages[i, j] = np.convolve(
                slice_binary_signals[i, j],
                np.ones(window_size)/window_size,
                mode='same'
            )

    # Calculate distances from each muscle bundle (points) to each measurement point (x, y, z)
    def euclidean_distance(px, py, pz, mx, my, mz):
        return np.sqrt((px - mx)**2 + (py - my)**2 + (pz - mz)**2)

    # Generate measured signals
    measurement_z = 0
    global_multiplier = 0.6
    generated_signals = []
    for m_idx, (mx, my) in enumerate(measurement_points):
        weighted_sum = np.zeros_like(time)
        for b_idx, bundle in enumerate(points):
            px, py = bundle['position']
            strength = np.pi * (bundle['size']**2) * fiber_density

            for slice_idx in range(num_slices):
                distance = euclidean_distance(px, py, slice_idx, mx, my, measurement_z)
                weight = global_multiplier / (distance + 1e-6)
                weight *= np.exp(-distance / 1.2)
                weighted_sum += slice_moving_averages[b_idx, slice_idx] * weight * strength
        generated_signals.append(weighted_sum)

    return generated_signals