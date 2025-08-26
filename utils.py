import streamlit as st
import numpy as np
import numpy as np

def amplify_and_scale_sigmoid(vad_value, steepness=6.0, vad_min=0.15, vad_max=0.85, output_min=1, output_max=8):
    # Normalize [vad_min, vad_max] to [0, 1]
    normalized = (np.clip(vad_value, vad_min, vad_max) - vad_min) / (vad_max - vad_min)

    # Apply sigmoid transformation
    # Transform [0,1] to [-steepness/2, steepness/2] for symmetry around 0.5
    x = steepness * (normalized - 0.5)
    sigmoid = 1 / (1 + np.exp(-x))

    # Scale to target range [output_min, output_max]
    scaled = output_min + sigmoid * (output_max - output_min)
    
    return scaled

def map_vad(slider_value, key=None, output_range=(1, 8)):

    vad_mapping = st.sidebar.radio(
        f"VAD Mapping on **{key}**:",
        options=["None", "V", "A", "D"],
        index=0,  # Default to "None"
        horizontal=True,
        key=key  # Unique key for the radio button
    )

    vad_amplification = st.sidebar.slider("VAD amplification", 1, 20, 6, 1, key=f"{key}_amplification")

    response_data = st.session_state['response'].get('data', {})
    if not response_data:
        st.sidebar.warning("No response data available, using slider value")
        output_value = slider_value

    # Then use the selection directly
    if vad_mapping == "V":
        output_value = np.mean(response_data['valence'])
        output_value = amplify_and_scale_sigmoid(output_value, steepness=vad_amplification, output_min=output_range[0], output_max=output_range[1])
    elif vad_mapping == "A":
        output_value = np.mean(response_data['arousal'])
        output_value = amplify_and_scale_sigmoid(output_value, steepness=vad_amplification, output_min=output_range[0], output_max=output_range[1])
    elif vad_mapping == "D":
        output_value = np.mean(response_data['dominance'])
        output_value = amplify_and_scale_sigmoid(output_value, steepness=vad_amplification, output_min=output_range[0], output_max=output_range[1])
    else:
        output_value = slider_value
    return output_value
