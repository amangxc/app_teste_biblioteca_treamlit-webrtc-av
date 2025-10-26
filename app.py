import streamlit as st
import cv2
import numpy as np
import time

# from scipy.signal import butter, filtfilt, find_peaks  <<<<< REMOVIDO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# --- 1. CONFIGURAÇÃO DA PÁGINA E CONSTANTES ---
st.set_page_config(page_title="Monitor BPM", layout="centered")
TAMANHO_TELA = 50  # Buffer de 50 frames
CALCULAR_A_CADA_N_FRAMES = 30  # Só calcula o BPM a cada 30 frames

# --- 2. CSS PARA ESCONDER "RUNNING" E "STOP" ---
css_limpeza_total = """
<style>
    div[data-testid="stWebRTCStatus"] { display: none; }
    div[key="camera_flash"] button[title="Stop"] { display: none; }
</style>
"""
st.markdown(css_limpeza_total, unsafe_allow_html=True)

# --- 3. FUNÇÃO DE FILTRO (REMOVIDA) ---
# A função bandpass_filter(scipy) foi removida por ser muito pesada.


# --- NOVA FUNÇÃO "LEVE" PARA ACHAR PICOS ---
def find_peaks_light(signal, distance, prominence_factor=0.5):
    """
    Uma versão muito mais leve do scipy.find_peaks.
    Acha picos que são maiores que seus vizinhos e que passam de um
    limite (threshold) dinâmico.
    """
    if len(signal) == 0:
        return []

    # Limite (threshold) dinâmico: Média + (fator * desvio_padrão)
    # Isso ajuda a ignorar picos pequenos que são só ruído
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    threshold = signal_mean + prominence_factor * signal_std

    candidate_peaks = []
    # Acha picos que são maiores que os vizinhos e maiores que o threshold
    for i in range(1, len(signal) - 1):
        if (
            signal[i] > signal[i - 1]
            and signal[i] > signal[i + 1]
            and signal[i] > threshold
        ):
            candidate_peaks.append(i)

    if not candidate_peaks:
        return []

    # Filtra os picos para garantir a "distância" mínima entre eles
    final_peaks = [candidate_peaks[0]]  # Adiciona o primeiro pico candidato
    for i in range(1, len(candidate_peaks)):
        # Se o pico atual está longe o suficiente do último pico adicionado...
        if candidate_peaks[i] - final_peaks[-1] >= distance:
            final_peaks.append(candidate_peaks[i])  # ...adiciona ele

    return np.array(final_peaks)


# --- 4. INICIALIZAÇÃO DO ESTADO GLOBAL (st.session_state) ---
if "buffer_G" not in st.session_state:
    st.session_state.buffer_G = []  # Só precisamos do buffer Green agora
if "bpm_fixado" not in st.session_state:
    st.session_state.bpm_fixado = None
if "bpm_primeiros" not in st.session_state:
    st.session_state.bpm_primeiros = []
if "medir_novamente" not in st.session_state:
    st.session_state.medir_novamente = True
if "camera_ligada" not in st.session_state:
    st.session_state.camera_ligada = False
if "info_bpm" not in st.session_state:
    st.session_state.info_bpm = "Aguarde..."


# --- 5. A CLASSE "PROCESSADORA" (VERSÃO LEVE) ---
class MeuProcessadorDeVideo(VideoProcessorBase):

    def __init__(self):
        self.contador_frames = 0
        self.tempo_espera = time.time()
        self.taxa_fps = 15.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.contador_frames += 1

        # --- Lógica de FPS (LEVE) ---
        if self.contador_frames % CALCULAR_A_CADA_N_FRAMES == 0:
            cont_final = time.time()
            tempo_decorrido = cont_final - self.tempo_espera
            if tempo_decorrido > 0:
                self.taxa_fps = CALCULAR_A_CADA_N_FRAMES / tempo_decorrido
            self.tempo_espera = time.time()

        # --- Lógica do ROI (LEVE) ---
        (largura, altura) = img.shape[:2]
        tamanho_roi = int(largura * 0.5)
        x_inicio = (altura - tamanho_roi) // 2
        y_inicio = (largura - tamanho_roi) // 2
        x_fim = x_inicio + tamanho_roi
        y_fim = y_inicio + tamanho_roi
        x_inicio, y_inicio = max(0, x_inicio), max(0, y_inicio)
        x_fim, y_fim = min(altura, x_fim), min(largura, y_fim)

        media_G = 0.0
        if x_fim > x_inicio and y_fim > y_inicio:
            roi_central = img[y_inicio:y_fim, x_inicio:x_fim]
            cv2.rectangle(img, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2)
            # Otimização: Só calcula a média do Verde (G), que é o que usamos
            media_G = np.mean(roi_central[:, :, 1])  # 1 = Canal Verde
        else:
            media_G = np.mean(img[:, :, 1])

        # --- Lógica de Buffer (LEVE) ---
        st.session_state.buffer_G.append(media_G)

        if len(st.session_state.buffer_G) > TAMANHO_TELA:
            st.session_state.buffer_G.pop(0)

        # --- LÓGICA DE CÁLCULO DE BPM (LEVE) ---
        if (
            self.contador_frames % CALCULAR_A_CADA_N_FRAMES == 0
            and len(st.session_state.buffer_G) == TAMANHO_TELA
        ):

            # Checagem do dedo na câmera (simplificada, só pelo Verde)
            media_G_recente = media_G
            # Se a média de verde for muito baixa, o dedo não está lá.
            dedo_na_camera = media_G_recente > 50

            if dedo_na_camera and st.session_state.medir_novamente:
                # Normalização (leve)
                G_array = np.array(st.session_state.buffer_G)
                Gnorm = G_array / (np.mean(G_array) + 1e-9)
                sinal_ac = Gnorm - np.mean(Gnorm)  # Sinal "detrended"

                # --- MUDANÇA PRINCIPAL ---
                # REMOVIDO: sinal_filtrado = bandpass_filter(...)
                # Vamos usar o sinal_ac "ruidoso" mesmo

                # REMOVIDO: peaks, _ = find_peaks(...)
                # USANDO: Nossa função leve
                peaks = find_peaks_light(
                    sinal_ac, distance=self.taxa_fps * 0.4, prominence_factor=0.5
                )
                # --- FIM DA MUDANÇA ---

                if len(peaks) > 1:
                    intervalos = np.diff(peaks) / self.taxa_fps
                    bpm_atual = 60 / np.mean(intervalos)

                    if st.session_state.bpm_fixado is None:
                        st.session_state.bpm_primeiros.append(bpm_atual)
                        if len(st.session_state.bpm_primeiros) >= 3:
                            st.session_state.bpm_fixado = np.mean(
                                st.session_state.bpm_primeiros
                            )

                    valor_a_mostrar = (
                        st.session_state.bpm_fixado
                        if st.session_state.bpm_fixado is not None
                        else bpm_atual
                    )
                    st.session_state.info_bpm = f"Seu BPM: {valor_a_mostrar:.1f}"
                else:
                    st.session_state.info_bpm = (
                        f"BPM: {st.session_state.bpm_fixado:.1f}"
                        if st.session_state.bpm_fixado is not None
                        else "Aguarde..."
                    )
            else:
                st.session_state.info_bpm = "Coloque o dedo e pressione 'Medir'"

        # --- Lógica de Desenho (LEVE) ---
        cor_texto = (0, 255, 0)
        if (
            "Aguarde" in st.session_state.info_bpm
            or "Coloque o dedo" in st.session_state.info_bpm
        ):
            cor_texto = (0, 0, 255)

        cv2.putText(
            img,
            st.session_state.info_bpm,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            cor_texto,
            2,
        )

        cv2.putText(
            img,
            f"FPS: {self.taxa_fps:.1f}",
            (img.shape[1] - 180, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 6. AS "CONSTRAINTS" (OTIMIZADAS) ---
video_constraints = {
    "video": {
        "facingMode": "environment",
        "width": {"ideal": 640},
        "height": {"ideal": 480},
        "frameRate": {"ideal": 15},
        "advanced": [{"torch": True}, {"focusMode": "continuous"}],
    }
}

# --- 7. A Interface do App ---
st.title("Monitor de Batimentos")

button_text = "Fechar Câmera" if st.session_state.camera_ligada else "Abrir Câmera"
if st.button(button_text):
    st.session_state.camera_ligada = not st.session_state.camera_ligada
    if st.session_state.camera_ligada:
        st.session_state.bpm_fixado = None
        st.session_state.bpm_primeiros = []
        st.session_state.medir_novamente = True
        st.session_state.info_bpm = "Aguarde..."
    st.rerun()

if st.button("Medir Novamente"):
    st.session_state.bpm_fixado = None
    st.session_state.bpm_primeiros = []
    st.session_state.medir_novamente = True
    st.session_state.info_bpm = "Aguarde..."
    st.rerun()

# --- 8. RENDERIZAÇÃO CONDICIONAL ---
if st.session_state.camera_ligada:
    webrtc_streamer(
        key="camera_flash",
        video_processor_factory=MeuProcessadorDeVideo,
        media_stream_constraints=video_constraints,
        async_processing=True,
        desired_playing_state=True,
    )
else:
    st.info("Clique em 'Abrir Câmera' para iniciar o monitoramento.")
