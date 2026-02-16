import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Helper Functions ---
def ricker_wavelet(freq, length, dt):
    t = np.arange(-length/2, (length/2)+dt, dt)
    y = (1.0 - 2.0*(np.pi**2)*(freq**2)*(t**2)) * np.exp(-(np.pi**2)*(freq**2)*(t**2))
    return t, y

def calculate_rc(vp1, rho1, vp2, rho2):
    z1 = vp1 * rho1
    z2 = vp2 * rho2
    return (z2 - z1) / (z2 + z1)

# --- 2. Streamlit UI Layout ---
st.set_page_config(page_title="Seismic Tuning Wedge", layout="wide")
#st.title("🪨 Interactive Seismic Tuning Wedge Model by Felix Obere")
st.subheader("🪨 Interactive Seismic Tuning Wedge Model by Felix Obere")
# Sidebar Controls
st.sidebar.header("Model Parameters")
freq = st.sidebar.slider("Frequency (Hz)", 5, 60, 25)
vp_shale = st.sidebar.number_input("Vp Shale (m/s)", value=3200)
rho_shale = st.sidebar.number_input("Rho Shale (g/cc)", value=2.4)
vp_sand = st.sidebar.number_input("Vp Sand (m/s)", value=2500)
rho_sand = st.sidebar.number_input("Rho Sand (g/cc)", value=2.2)

# --- 3. The Physics Engine ---
# Constants
dt = 0.001 # 1ms sample rate
max_thickness_ms = 60 # Max thickness in Time (ms)
n_traces = 50 # Number of traces in the wedge

# Calculate Reflectivity
rc_top = calculate_rc(vp_shale, rho_shale, vp_sand, rho_sand)
rc_base = calculate_rc(vp_sand, rho_sand, vp_shale, rho_shale)

# Create Wavelet
t_wave, wavelet = ricker_wavelet(freq, 0.128, dt)

# Build the Wedge (in Time Domain for simplicity)
traces = []
amplitudes = []
thicknesses_ms = np.linspace(0, max_thickness_ms, n_traces)

for thick_ms in thicknesses_ms:
    # Create a reflectivity spike train
    nsamp = int(0.2 / dt) # 200ms window
    reflectivity = np.zeros(nsamp)
    
    # Place Top Sand (at 50ms)
    top_idx = int(0.05 / dt)
    reflectivity[top_idx] = rc_top
    
    # Place Base Sand (Top + Thickness)
    base_idx = top_idx + int((thick_ms/1000) / dt)
    if base_idx < nsamp:
        reflectivity[base_idx] = rc_base
        
    # Convolve
    seismic_trace = np.convolve(reflectivity, wavelet, mode='same')
    traces.append(seismic_trace)
    
    # Capture max absolute amplitude for tuning curve
    amplitudes.append(np.max(np.abs(seismic_trace)))

traces_arr = np.array(traces).T 

# --- 4. Visualization ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Seismic Wedge Display")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # 1. Variable Density Plot (Background)
    im = ax1.imshow(traces_arr, aspect='auto', cmap='RdBu', 
                    extent=[0, n_traces, 0.2, 0], vmin=-0.5, vmax=0.5)
    
    # 2. Overlay Wiggles
    # Excursion controls how wide the wiggles wave (gain)
    excursion = 1.0 
    t = np.linspace(0, 0.2, traces_arr.shape[0])
    
    # Loop to plot every trace (or skip every nth trace with range(0, n_traces, 2))
    for i in range(n_traces):
        # Shift trace to be centered at i + 0.5 to match imshow pixel centers
        trace_x = traces_arr[:, i] * excursion + (i + 0.5)
        
        # Plot the black line
        ax1.plot(trace_x, t, 'k-', linewidth=0.3)
        
        # Optional: Fill positive lobes
        ax1.fill_betweenx(t, (i + 0.5), trace_x, where=(trace_x > (i + 0.5)), color='k')
    
    # 3. Add Horizons
    x_coords = np.arange(n_traces) + 0.5
    top_t = np.full(n_traces, 0.05)
    base_t = 0.05 + (thicknesses_ms / 1000.0)
    
    ax1.plot(x_coords, top_t, 'r-', linewidth=1.5, label="Top Sand") # Changed to Red for visibility
    ax1.plot(x_coords, base_t, 'r--', linewidth=1.5, label="Base Sand")
    
    ax1.set_xlabel("Trace Number")
    ax1.set_ylabel("Time (s)")
    ax1.set_xlim(0, n_traces) # Ensure graph fits exactly
    ax1.legend(loc='upper right')
    ax1.grid(False)
    ax1.set_title("Red = Negative (Soft/Decrease AI) | Blue = Positive (Hard/Increase AI)")
    st.pyplot(fig1)

with col2:
    st.subheader("Tuning Curve Analysis")
    fig2, ax2 = plt.subplots(figsize=(4, 6))
    ax2.plot(thicknesses_ms, amplitudes, 'k-', linewidth=2)
    
    # Find Tuning Thickness
    tuning_idx = np.argmax(amplitudes)
    tuning_thick = thicknesses_ms[tuning_idx]
    
    ax2.axvline(tuning_thick, color='r', linestyle='--', label=f'Tuning: {tuning_thick:.1f}ms')
    ax2.set_xlabel("Thickness (ms)")
    ax2.set_ylabel("Composite Amplitude")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# --- 5. The Report ---
st.divider()
st.subheader("📝 Analysis Report")

# Calculate depth thickness: Z = (V * t) / 2
tuning_depth_m = (vp_sand * tuning_thick / 1000) / 2
tuning_depth_ft = tuning_depth_m * 3.28084

st.markdown(f"""
* **Tuning Thickness (Time):** The maximum constructive interference occurs at **{tuning_thick:.1f} ms** (TWT).
* **Tuning Thickness (Depth):** Based on a sand velocity of **{vp_sand} m/s**, this represents a true thickness of:
    * **{tuning_depth_m:.1f} meters**
    * **{tuning_depth_ft:.1f} feet**
* **Resolution Limit:** Below **{tuning_thick:.1f} ms**, amplitude diminishes due to destructive interference.
* **Lithology:** Encased in Shale (Vp={vp_shale}), the Sand (Vp={vp_sand}) creates a **{'Hard' if rc_top > 0 else 'Soft'}** kick at the top interface.
""")