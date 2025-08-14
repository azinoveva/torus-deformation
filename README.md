# 🌀 Interactive Torus Deformer

A Streamlit app that creates 3D torus deformations to create dynamic shapes.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd torus-deformation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_torus_app.py
   ```

## 🎯 How to Use

### Basic Controls
- **Major/Minor Divisions**: Control the resolution of the torus
- **Major/Minor Radius**: Adjust the size and thickness
- **Height**: Control the height of the torus

### Deformation Types
- **Noise**: Random, Perlin, Worley, Starlike
- **Twists**: Möbius twist, helical warp, saddle-like deformation
- **Scaling**: Gradient and sine wave deformations
- **Cross-section**: Modulation of the tube radius


## 🧮 Mathematical Foundation

### Base Torus Parametrization

The base torus is defined by the parametric equations:

```
X(u,v) = (R + r·cos(v))·cos(u)
Y(u,v) = (R + r·cos(v))·sin(u)  
Z(u,v) = r·sin(v)
```

Where:
- **R** = Major radius (distance from center of hole to center of tube)
- **r** = Minor radius (radius of the tube cross-section)
- **u** ∈ [0, 2π] = Angle along the major ring
- **v** ∈ [0, 2π] = Angle along the minor ring (cross-section)

### 1. Cross-Section Polar Modulation

Modulates the minor radius based on the cross-section angle:

```
r_mod(v) = r · (1 + A·sin(f·v))
X_mod = (R + r_mod(v)·cos(v))·cos(u)
Y_mod = (R + r_mod(v)·sin(v))·sin(u)
Z_mod = r_mod(v)·sin(v)
```

Where:
- **A** = Modulation amplitude [0, 1]
- **f** = Modulation frequency (integer)

### 2. Gradient Scaling

Creates smooth bulging along the major ring using cosine interpolation:

```
scale(u) = scale_center + (scale_max - scale_min)/2 · cos(u)
X_scale = (R + r·scale(u)·cos(v))·cos(u)
Y_scale = (R + r·scale(u)·sin(v))·sin(u)
Z_scale = r·scale(u)·sin(v)
```

Where:
- **scale_center** = (scale_max + scale_min) / 2
- **scale_max, scale_min** = Maximum and minimum scaling factors

### 3. Sine Wave Deformation

Applies sinusoidal modulation to the minor radius:

```
r_mod(u) = r · (1 + A·sin(f·u + φ))
X_sine = (R + r_mod(u)·cos(v))·cos(u)
Y_sine = (R + r_mod(u)·sin(v))·sin(u)
Z_sine = r_mod(u)·sin(v)
```

Where:
- **A** = Amplitude [0, 1]
- **f** = Frequency (integer)
- **φ** = Phase shift [0, 2π]

### 4. Saddle-Like Deformation (Spatial Bending)

Bends the torus into a saddle shape by adding displacement functions:

```
bend_y(u) = S · (sin(2u) + 0.3·sin(4u))
bend_z(u) = S · (cos(2u) + 0.3·cos(4u))

X_s = X_base
Y_s = Y_base + bend_y(u)
Z_s = Z_base + bend_z(u)
```

Where:
- **S** = deformation strength [0, 2]

### 5. Möbius Twist

Creates a Möbius strip-like twist by rotating cross-sections:

```
twist_angle(u) = T · sin(u)
X_twist = X_base
Y_twist = Y_base·cos(twist_angle) - Z_base·sin(twist_angle)
Z_twist = Y_base·sin(twist_angle) + Z_base·cos(twist_angle)
```

Where:
- **T** = Twist strength [0, 3]

### 6. Helical Warp

Applies progressive rotation along the major ring:

```
warp_angle(u) = H · u
X_warp = X_base
Y_warp = Y_base·cos(warp_angle) - Z_base·sin(warp_angle)
Z_warp = Y_base·sin(warp_angle) + Z_base·cos(warp_angle)
```

Where:
- **H** = Helical warp strength [0, 5]

### 7. Noise Deformations

#### Random Noise

#### Perlin Noise
```
noise(x,y) = Σ(octave=1 to N) amplitude_octave · noise_octave(x·freq_octave, y·freq_octave)
amplitude_octave = persistence^(octave-1)
freq_octave = lacunarity^(octave-1)
```

#### Worley (Cellular) Noise
```
noise(x,y) = min(distance((x,y), feature_point_i)) for all i
distance = √((x-x_i)² + (y-y_i)²)
```

#### Star-Like Spot Noise

Built using Streamlit, NumPy, Plotly, and WebRTC
