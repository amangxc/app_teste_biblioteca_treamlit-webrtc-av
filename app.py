import streamlit as st
import cv2
import numpy as np
import time
from scipy.signal import butter, filtfilt, find_peaks
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# --- 1. CONFIGURAÇÃO DA PÁGINA E CONSTANTES ---
st.set_page_config(page_title="Monitor BPM", layout="centered")
TAMANHO_TELA = 50  # <<<<<< OTIMIZAÇÃO: Buffer menor (era 100)

# --- 2. CSS PARA ESCONDER "RUNNING" E "STOP" ---
css_limpeza_total = """
<style>
    div[data-testid="stWebRTCStatus"] { display: none; }
    div[key="camera_flash"] button[title="Stop"] { display: none; }
</style>
"""
st.markdown(css_limpeza_total, unsafe_allow_html=True)


# --- 3. FUNÇÃO DE FILTRO  ---
def bandpass_filter(signal, low=0.8, high=3, fs=30):
    fr_nyquist = 0.5 * fs
    low = low / fr_nyquist
    high = high / fr_nyquist
    b, a = butter(2, [low, high], btype="band")
    return filtfilt(b, a, signal)


# --- 4. INICIALIZAÇÃO DO ESTADO GLOBAL (st.session_state) ---
if "buffer_R" not in st.session_state:
    st.session_state.buffer_R = []
if "buffer_G" not in st.session_state:
    st.session_state.buffer_G = []
if "buffer_B" not in st.session_state:
    st.session_state.buffer_B = []
if "bpm_fixado" not in st.session_state:
    st.session_state.bpm_fixado = None
if "bpm_primeiros" not in st.session_state:
    st.session_state.bpm_primeiros = []
if "medir_novamente" not in st.session_state:
    st.session_state.medir_novamente = True
if "camera_ligada" not in st.session_state:
    st.session_state.camera_ligada = False


# --- 5. A CLASSE "PROCESSADORA" (OTIMIZADA) ---
class MeuProcessadorDeVideo(VideoProcessorBase):

    def __init__(self):
        # Variáveis para FPS
        self.contador_frames = 0
        self.tempo_espera = time.time()
        self.taxa_fps = 15  # Valor inicial (combinando com a constraint)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # --- Lógica de FPS (do Paulo) ---
        self.contador_frames += 1
        # Otimização: calcular a cada 30 frames (aprox 2s)
        if self.contador_frames % 30 == 0:
            cont_final = time.time()
            tempo_decorrido = cont_final - self.tempo_espera
            if tempo_decorrido > 0:
                self.taxa_fps = self.contador_frames / tempo_decorrido
            self.contador_frames = 0
            self.tempo_espera = time.time()

        # --- Lógica do ROI  ---
        (largura, altura) = img.shape[:2]
        tamanho_roi = int(largura * 0.5)
        x_inicio = (altura - tamanho_roi) // 2
        y_inicio = (largura - tamanho_roi) // 2
        x_fim = x_inicio + tamanho_roi
        y_fim = y_inicio + tamanho_roi

        x_inicio = max(0, x_inicio)
        y_inicio = max(0, y_inicio)
        x_fim = min(altura, x_fim)
        y_fim = min(largura, y_fim)

        if x_fim > x_inicio and y_fim > y_inicio:
            roi_central = img[y_inicio:y_fim, x_inicio:x_fim]
            cv2.rectangle(img, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2)
            media_BGR = np.mean(roi_central, axis=(0, 1))
        else:
            cv2.rectangle(img, (0, 0), (largura // 4, altura // 4), (0, 0, 255), 2)
            media_BGR = np.mean(img, axis=(0, 1))

        # --- Lógica de Buffer ---
        st.session_state.buffer_B.append(media_BGR[0])
        st.session_state.buffer_G.append(media_BGR[1])
        st.session_state.buffer_R.append(media_BGR[2])

        if len(st.session_state.buffer_B) > TAMANHO_TELA:
            st.session_state.buffer_B.pop(0)
            st.session_state.buffer_G.pop(0)
            st.session_state.buffer_R.pop(0)

        # --- Lógica de Cálculo de BPM  ---
        if len(st.session_state.buffer_B) == TAMANHO_TELA:
            media_B_recente = media_BGR[0]
            media_G_recente = media_BGR[1]
            media_R_recente = media_BGR[2]

            vermelho = 60
            dedo_na_camera = (
                media_R_recente > media_G_recente + vermelho
                and media_R_recente > media_B_recente + vermelho
            )

            if dedo_na_camera and st.session_state.medir_novamente:
                Gnorm = np.array(st.session_state.buffer_G) / (
                    np.mean(st.session_state.buffer_G) + 1e-9
                )
                sinal_ac = Gnorm - np.mean(Gnorm)
                sinal_filtrado = bandpass_filter(
                    sinal_ac, low=0.8, high=3, fs=self.taxa_fps
                )
                peaks, _ = find_peaks(sinal_filtrado, distance=self.taxa_fps * 0.4)

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
                    info_bpm = f"Seu BPM: {valor_a_mostrar:.1f}"
                    cv2.putText(
                        img,
                        info_bpm,
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3,
                    )
                else:
                    info_bpm = (
                        f"BPM: {st.session_state.bpm_fixado:.1f}"
                        if st.session_state.bpm_fixado is not None
                        else "Aguarde..."
                    )
                    cv2.putText(
                        img,
                        info_bpm,
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    img,
                    "Coloque o dedo e pressione 'Medir'",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # --- Lógica de Desenho de FPS ---
        cv2.putText(
            img,
            f"FPS: {self.taxa_fps:.1f}",
            (img.shape[1] - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # --- Retorno ---
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- 6. AS "CONSTRAINTS" (OTIMIZADAS) ---
video_constraints = {
    "video": {
        "facingMode": "environment",  # Força câmera traseira
        "width": {"ideal": 640},  # <<<<<< OTIMIZAÇÃO: Menor resolução
        "height": {"ideal": 480},  # <<<<<< OTIMIZAÇÃO: Menor resolução
        "frameRate": {"ideal": 15},  # <<<<<< OTIMIZAÇÃO: Menos FPS
        "advanced": [{"torch": True}, {"focusMode": "continuous"}],  # Força flash
    }
}


# --- 7. A Interface do App (+ Botão de Reset) ---
st.title("Monitor de Batimentos")

# Lógica do botão "Abrir/Fechar Câmera"
button_text = "Fechar Câmera" if st.session_state.camera_ligada else "Abrir Câmera"
if st.button(button_text):
    st.session_state.camera_ligada = not st.session_state.camera_ligada

    # Se estivermos LIGANDO a câmera, resetamos a medição
    if st.session_state.camera_ligada:
        st.session_state.bpm_fixado = None
        st.session_state.bpm_primeiros = []
        st.session_state.medir_novamente = True

    st.rerun()  # Correção de experimental_rerun

# Botão de reset (para substituir a tecla 'm')
if st.button("Medir Novamente"):
    st.session_state.bpm_fixado = None
    st.session_state.bpm_primeiros = []
    st.session_state.medir_novamente = True
    st.rerun()  # Correção de experimental_rerun

# --- 8. RENDERIZAÇÃO CONDICIONAL ---
if st.session_state.camera_ligada:
    webrtc_streamer(
        key="camera_flash",  # Chave única
        video_processor_factory=MeuProcessadorDeVideo,
        media_stream_constraints=video_constraints,  # Aplica flash e câmera traseira
        async_processing=True,
        desired_playing_state=True,  # Tenta iniciar automaticamente
    )
else:
    st.info("Clique em 'Abrir Câmera' para iniciar o monitoramento.")
