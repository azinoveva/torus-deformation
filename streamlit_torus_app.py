import streamlit as st
import numpy as np
import plotly.graph_objects as go
import numpy as np
from perlin_noise import PerlinNoise
from pythonworley import worley

# Page config
st.set_page_config(
    page_title="Torus Deformer",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌀 Torus Deformer")

# Sidebar controls
st.sidebar.header("🎛️ Deformation Controls")

# Basic torus parameters
st.sidebar.subheader("📐 Basic Torus")
n_major = st.sidebar.slider("Major divisions", 20, 200, 80, 5)
n_minor = st.sidebar.slider("Minor divisions", 10, 100, 30, 5)
major_radius = st.sidebar.slider("Major radius (R)", 1.0, 8.0, 3.0, 0.1, help="Distance from center of hole to center of tube")
minor_radius = st.sidebar.slider("Minor radius (r)", 0.2, 3.0, 1.0, 0.1, help="Radius of the tube cross-section")
height_scale = st.sidebar.slider("Height scale", 0.1, 3.0, 1.0, 0.1, help="Scale factor for Z-axis height")

# Noise parameters
st.sidebar.subheader("🎯 Noise Deformations")
noise_enabled = st.sidebar.checkbox("Enable Noise", False)
if noise_enabled:
    noise_type = st.sidebar.selectbox(
        "Noise Type",
        ["Random", "Perlin", "Worley", "Spots"]
    )
    
    # Common noise parameter
    noise_scale = st.sidebar.slider("Noise Strength", 0.0, 1.0, 0.3, 0.05)
    
    # Perlin specific parameters
    if noise_type is "Perlin":
        noise_octaves = st.sidebar.slider("Octaves", 1, 8, 3, 1)

# Twist deformations
st.sidebar.subheader("🌀 Twist Deformations")
mobius_twist_enabled = st.sidebar.checkbox("Enable Möbius Twist", False)
if mobius_twist_enabled:
    mobius_strength = st.sidebar.slider("Möbius Twist Strength", 0.0, 3.0, 1.0, 0.1)

helical_warp_enabled = st.sidebar.checkbox("Enable Helical Warp", False)
if helical_warp_enabled:
    helical_strength = st.sidebar.slider("Helical Warp Strength", 0.0, 5.0, 1.0, 0.1)

s_deformation_enabled = st.sidebar.checkbox("Enable Saddle Deformation", False)
if s_deformation_enabled:
    s_strength = st.sidebar.slider("Saddle Deformation Strength", 0.0, 2.0, 0.8, 0.1)

# Spatial deformations
st.sidebar.subheader("🌍 Scaling Deformations")

gradient_scaling_enabled = st.sidebar.checkbox("Enable Gradient Scaling", False)
if gradient_scaling_enabled:
    scale_min = st.sidebar.slider("Scale Min", 0.1, 1.0, 0.5, 0.05)
    scale_max = st.sidebar.slider("Scale Max", 1.0, 3.0, 1.5, 0.05)

sine_wave_enabled = st.sidebar.checkbox("Enable Sine Wave Deformation", False)
if sine_wave_enabled:
    sine_amplitude = st.sidebar.slider("Sine Amplitude", 0.0, 1.0, 0.5, 0.05)
    sine_frequency = st.sidebar.slider("Sine Frequency", 1, 8, 3, 1)
    sine_phase = st.sidebar.slider("Sine Phase", 0.0, 2*np.pi, 0.0, 0.1)

# Cross-section modulation
st.sidebar.subheader("🔴 Cross-section Modulation")
cross_section_enabled = st.sidebar.checkbox("Enable Cross-section Modulation", False)
if cross_section_enabled:
    mod_amplitude = st.sidebar.slider("Modulation Amplitude", 0.0, 1.0, 0.3, 0.05)
    mod_frequency = st.sidebar.slider("Modulation Frequency", 1, 20, 5, 1)

# Main torus generation function
def generate_deformed_torus(n_major, n_minor, major_radius, minor_radius, height_scale, **kwargs):
    """Generate torus with all applied deformations"""
    R, r = major_radius, minor_radius  # Use the user-defined radii
    # Use endpoint=True for seamless torus
    u = np.linspace(0, 2*np.pi, n_major, endpoint=True)
    v = np.linspace(0, 2*np.pi, n_minor, endpoint=True)
    U, V = np.meshgrid(u, v)
    
    # Start with base torus
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V) * height_scale  # Apply height scaling
    
    # Apply cross-section modulation first
    if kwargs.get('cross_section_enabled'):
        mod = 1 + kwargs['mod_amplitude'] * np.sin(kwargs['mod_frequency'] * V)
        X = (R + r * mod * np.cos(V)) * np.cos(U)
        Y = (R + r * mod * np.sin(V)) * np.sin(U)
        Z = r * mod * np.sin(V) * height_scale  # Apply height scaling
    
    # Apply gradient scaling
    if kwargs.get('gradient_scaling_enabled'):
        gradient_factor = (kwargs['scale_max'] - kwargs['scale_min']) / 2
        center_scale = (kwargs['scale_max'] + kwargs['scale_min']) / 2
        # Use cosine function that naturally closes at endpoints
        scales = center_scale + gradient_factor * np.cos(u).reshape(1, -1)
        X = (R + r * scales * np.cos(V)) * np.cos(U)
        Y = (R + r * scales * np.sin(V)) * np.sin(U)
        Z = r * scales * np.sin(V) * height_scale  # Apply height scaling
    
    # Apply sine wave deformation
    if kwargs.get('sine_wave_enabled'):
        radius_mod = 1 + kwargs['sine_amplitude'] * np.sin(kwargs['sine_frequency'] * U + kwargs['sine_phase'])
        X = (R + r * radius_mod * np.cos(V)) * np.cos(U)
        Y = (R + r * radius_mod * np.sin(V)) * np.sin(U)
        Z = r * radius_mod * np.sin(V) * height_scale  # Apply height scaling
    
    # Apply S-deformation
    if kwargs.get('s_deformation_enabled'):
        s_bend_y = kwargs['s_strength'] * (np.sin(2 * U) + 0.3 * np.sin(4 * U))
        s_bend_z = kwargs['s_strength'] * (np.cos(2 * U) + 0.3 * np.cos(4 * U))
        X = X
        Y = Y + s_bend_y
        Z = Z + s_bend_z * height_scale  # Apply height scaling to S-deformation
    
    # Apply Möbius twist
    if kwargs.get('mobius_twist_enabled'):
        twist_angle = kwargs['mobius_strength'] * np.sin(U)
        cos_t = np.cos(twist_angle)
        sin_t = np.sin(twist_angle)
        Y_orig, Z_orig = Y.copy(), Z.copy()
        Y = Y_orig * cos_t - Z_orig * sin_t
        Z = Y_orig * sin_t + Z_orig * cos_t
    
    # Apply helical warp
    if kwargs.get('helical_warp_enabled'):
        warp_angle = kwargs['helical_strength'] * U
        cos_t = np.cos(warp_angle)
        sin_t = np.sin(warp_angle)
        Y_orig, Z_orig = Y.copy(), Z.copy()
        Y = Y_orig * cos_t - Z_orig * sin_t
        Z = Y_orig * sin_t + Z_orig * cos_t
    
    # Apply noise deformation last
    if kwargs.get('noise_enabled'):
        # Generate appropriate noise based on type
        if kwargs['noise_type'] == "Random":
            noise = generate_random_noise((n_minor, n_major))
        elif kwargs['noise_type'] == "Perlin":
            noise = generate_perlin_noise((n_minor, n_major), kwargs['noise_octaves'])
        elif kwargs['noise_type'] == "Worley":
            noise = generate_worley_noise((n_minor, n_major))
        elif kwargs['noise_type'] == "Spots":
            noise = generate_star_noise((n_minor, n_major))
        
        # Calculate normals for noise displacement
        normals = np.stack((
            np.cos(V) * np.cos(U),
            np.cos(V) * np.sin(U),
            np.sin(V)
        ), axis=2)
        
        # Apply noise displacement with proper wrapping
        X += normals[:,:,0] * kwargs['noise_scale'] * noise
        Y += normals[:,:,1] * kwargs['noise_scale'] * noise
        Z += normals[:,:,2] * kwargs['noise_scale'] * noise * height_scale
        
        # Ensure seamless wrapping by enforcing periodicity
        # For major ring (U direction) - wrap around at 2π
        X[:, 0] = X[:, -1]  # First and last columns should match
        Y[:, 0] = Y[:, -1]
        Z[:, 0] = Z[:, -1]
        
        # For minor ring (V direction) - wrap around at 2π
        X[0, :] = X[-1, :]  # First and last rows should match
        Y[0, :] = Y[-1, :]
        Z[0, :] = Z[-1, :]
    
    return X, Y, Z

# Noise generation functions with proper scale implementation
def generate_perlin_noise(shape, octaves, seed=None):
    if not seed:
        seed = np.random.randint(1, 1000000)
    noise_generator = PerlinNoise(octaves=octaves, seed=seed)
    noise = np.array([[noise_generator([i / shape[0], j / shape[1]]) for j in range(shape[1])] for i in range(shape[0])])
    return noise

def generate_random_noise(shape, seed=None):
    if not seed:
        seed = np.random.randint(1, 1000000)
    noise = np.random.rand(shape[0], shape[1])
    return noise

def generate_worley_noise(shape, seed=None):
    if not seed:
        seed = np.random.randint(1, 1000000)
    shape = (int(shape[1]/10), int(shape[0]/10))
    noise, cells = worley(shape, dens=10, seed=seed)
    noise = noise[0].T
    return noise

def generate_star_noise(shape, seed=None):
    # Make rectangular grid
    x, y = np.arange(shape[0]), np.arange(shape[1])
    x, y = np.meshgrid(x, y, indexing="ij")

    # Generate noise: random displacements r at random angles phi
    np.random.seed(0)
    phi = np.random.uniform(0, 2 * np.pi, x.shape)
    r = np.random.uniform(0, 0.5, x.shape)

    # Shrink star size to keep it inside its cell.
    # Alse, we want more small stars - for the background effect.
    # To do that we rescale displacements: r -> 1/2 - 0.001 / r.
    r = np.clip(0.5 - 1e-3 / r, 0, None)
    size = 20 * (0.5 - r) - 0.04

    return size

# Main app - automatically update when parameters change
# Collect all parameters
params = {
    'cross_section_enabled': cross_section_enabled,
    'mod_amplitude': mod_amplitude if cross_section_enabled else 0,
    'mod_frequency': mod_frequency if cross_section_enabled else 1,
    'gradient_scaling_enabled': gradient_scaling_enabled,
    'scale_min': scale_min if gradient_scaling_enabled else 1,
    'scale_max': scale_max if gradient_scaling_enabled else 1,
    'sine_wave_enabled': sine_wave_enabled,
    'sine_amplitude': sine_amplitude if sine_wave_enabled else 0,
    'sine_frequency': sine_frequency if sine_wave_enabled else 1,
    'sine_phase': sine_phase if sine_wave_enabled else 0,
    's_deformation_enabled': s_deformation_enabled,
    's_strength': s_strength if s_deformation_enabled else 0,
    'mobius_twist_enabled': mobius_twist_enabled,
    'mobius_strength': mobius_twist_enabled if mobius_twist_enabled else 0,
    'helical_warp_enabled': helical_warp_enabled,
    'helical_strength': helical_strength if helical_warp_enabled else 0,
    'noise_enabled': noise_enabled,
    'noise_type': noise_type if noise_enabled else "Perlin",
    'noise_scale': noise_scale if noise_enabled else 0.3,
    'noise_octaves': noise_octaves if noise_enabled and noise_type == "Perlin" else 1,
}

# Generate torus automatically
with st.spinner("Generating deformed torus..."):
    X, Y, Z = generate_deformed_torus(n_major, n_minor, major_radius, minor_radius, height_scale, **params)
    
    # Create 3D plot with Plotly - subtle gradient, no visible color bar
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Blues',  # Subtle blue gradient
        showscale=False,  # Hide color bar
        opacity=1.0,  # Full opacity for solid appearance
        cmin=Z.min(),  # Use Z values for subtle depth
        cmax=Z.max()
    )])
    
    fig.update_layout(
        title="Deformed Torus",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y", 
            zaxis_title="Z",
            aspectmode='cube',
            # Lock axis ranges for consistent viewing
            xaxis=dict(range=[-10, 10], autorange=False),
            yaxis=dict(range=[-10, 10], autorange=False),
            zaxis=dict(range=[-10, 10], autorange=False)
        ),
        width=800,
        height=600
    )  
    
    # Display plot
    plot_result = st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎨 How to Use:
1. **Adjust parameters** - see changes instantly!
2. **Enable deformations** you want to combine
3. **Tweak parameters** for each deformation
4. **Mix and match** different effects

""")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, NumPy, and Plotly")
