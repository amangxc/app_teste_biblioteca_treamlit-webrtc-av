import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
from scipy.signal import detrend, butter, filtfilt
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import time, math

# --- CONSTANTES ---
WINDOW_SIZE = 100
FS_ESTIMADA = 30
BPM_RANGE = [40, 240]

st.set_page_config(page_title="Monitor Cardíaco ao Vivo", layout="wide")
st.title("💓 Monitor de Batimentos Cardíacos (PPG - Ao Vivo)")
st.markdown(
    "Coloque o dedo sobre a câmera (flash ligado). O gráfico será atualizado em tempo real."
)

# Buffers globais
buffer_R = []


# Função para processar PPG e calcular sinal filtrado
def process_signal_ppg_live(data, min_bpm=40, max_bpm=240):
    if len(data) < WINDOW_SIZE:
        return None, None, None
    signal = np.array(data[-WINDOW_SIZE:])
    detrended = detrend(signal)

    nyq = 0.5 * FS_ESTIMADA
    low_cut = min_bpm / 60 / nyq
    high_cut = max_bpm / 60 / nyq
    b, a = butter(4, [low_cut, high_cut], btype="band")
    filtered = filtfilt(b, a, detrended)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)

    # FFT para cálculo do BPM
    N = len(filtered)
    yf = np.abs(fft(filtered))
    xf = fftfreq(N, d=1 / FS_ESTIMADA)
    mask = xf > 0
    bpm_freqs = xf[mask] * 60
    amps = yf[mask]
    idx_valid = np.where((bpm_freqs >= min_bpm) & (bpm_freqs <= max_bpm))[0]
    if len(idx_valid) == 0:
        bpm = None
    else:
        bpm = bpm_freqs[idx_valid[np.argmax(amps[idx_valid])]]

    return filtered, np.arange(len(filtered)) / FS_ESTIMADA, bpm


# Placeholder para gráfico
plot_placeholder = st.empty()
bpm_placeholder = st.empty()


# Função de callback para cada frame da câmera
def video_frame_callback(frame: av.VideoFrame):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    roi = img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    mean_rgb = np.mean(roi, axis=(0, 1))
    buffer_R.append(mean_rgb[2])  # canal R

    # Processa PPG
    filtered, t, bpm = process_signal_ppg_live(buffer_R, BPM_RANGE[0], BPM_RANGE[1])

    if filtered is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=filtered, mode="lines", line=dict(color="crimson", width=2))
        )
        fig.update_layout(
            title="Sinal PPG ao Vivo",
            xaxis_title="Tempo (s)",
            yaxis_title="Amplitude",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=300,
        )
        plot_placeholder.plotly_chart(fig, use_container_width=True)

    if bpm is not None:
        bpm_placeholder.metric("Frequência Cardíaca Estimada", f"{bpm:.1f} BPM")

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Inicia a câmera ao vivo com WebRTC
webrtc_streamer(
    key="ppg-live",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
