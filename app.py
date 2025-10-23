import streamlit as st
import cv2
import numpy as np
import time
import av
import tempfile
import os
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, butter, filtfilt
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# ----------------- Parâmetros -----------------
TAMANHO_TELA = 100  # frames do buffer (ajustável)
VERMELHO_MINIMO = 40  # sensibilidade de detecção do dedo (ajuste fino)
BPM_RANGE = [40, 240]
FS_ESTIMADA = 30
TAM_SUAVIZACAO_BPM = 8  # média móvel para suavizar leitura
CALC_INTERVAL_S = 1.0  # intervalo mínimo entre cálculos de BPM (segundos)

# ----------------- Funções de processamento (POS) -----------------


def aplicar_pos_e_filtrar(C, fps):
    # C: N x 3 (RGB) matriz
    frame_count = len(C)
    l = int(fps * 1.6)  # janela ~1.6s
    H = np.zeros(frame_count)

    pp = np.array([[1, -1, 0], [1, 0, -1]])

    for n in range(frame_count):
        m = max(0, n - l + 1)
        if n - m + 1 == l:
            window_C = C[m : n + 1, :]
            mu = np.mean(window_C, axis=0)
            # evita divisão por zero
            mu[mu == 0] = 1.0
            Cn = window_C / mu[np.newaxis, :]
            S = pp @ Cn.T
            S1, S2 = S[0, :], S[1, :]
            sigma1 = np.std(S1)
            sigma2 = np.std(S2)
            alpha = sigma1 / sigma2 if sigma2 != 0 else 0
            h = S1 + alpha * S2
            h -= np.mean(h)
            H[m : n + 1] += h

    detrended_signal = detrend(H)
    nyquist_freq = 0.5 * fps
    low_cutoff = BPM_RANGE[0] / 60.0 / nyquist_freq
    high_cutoff = BPM_RANGE[1] / 60.0 / nyquist_freq

    # validações
    if low_cutoff <= 0:
        low_cutoff = 1e-4
    if high_cutoff >= 1:
        high_cutoff = 0.9999

    b, a = butter(4, [low_cutoff, high_cutoff], btype="band")
    try:
        filtered_signal = filtfilt(b, a, detrended_signal)
    except Exception:
        # caso o filtro falhe (sinal muito curto), retorna o detrended
        filtered_signal = detrended_signal

    if np.std(filtered_signal) != 0:
        normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(
            filtered_signal
        )
    else:
        normalized_signal = filtered_signal

    return normalized_signal, detrended_signal, H


def calcular_fft_bpm_snr(normalized_signal, fps):
    N = len(normalized_signal)
    if N < 3:
        return None, None, None, None

    yf = fft(normalized_signal)
    xf = fftfreq(N, 1 / fps)
    freqs = xf[xf > 0]
    amps = 2.0 / N * np.abs(yf[0 : N // 2])[1 : N // 2 + 1]

    valid_idx = np.where((freqs * 60 >= BPM_RANGE[0]) & (freqs * 60 <= BPM_RANGE[1]))
    valid_freqs = freqs[valid_idx]
    valid_amps = amps[valid_idx]

    if len(valid_freqs) == 0 or len(valid_amps) == 0:
        return None, None, None, None

    peak_freq = valid_freqs[np.argmax(valid_amps)]
    bpm = peak_freq * 60

    fundamental = np.max(valid_amps)
    noise_energy = np.sum(valid_amps) - fundamental
    snr = 10 * np.log10(fundamental / noise_energy) if noise_energy > 0 else 0

    return bpm, snr, valid_freqs * 60, valid_amps


# ----------------- Classe de processamento (WebRTC) -----------------


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.C = []  # buffer RGB
        self.valor_bpm = []
        self.contador_frames = 0
        self.tempo_espera = time.time()
        self.taxa_fps = FS_ESTIMADA
        self.media_bpm = 0.0
        self.info_status = "Coloque o dedo sobre a câmera (flash ligado)."
        self.last_bpm_time = 0.0

    def process_frame_logic(self, imagem):
        self.contador_frames += 1

        # Atualiza FPS a cada 60 frames
        if self.contador_frames % 60 == 0:
            cont_final = time.time()
            tempo_decorrido = cont_final - self.tempo_espera
            if tempo_decorrido > 0:
                self.taxa_fps = (
                    60.0 / tempo_decorrido if tempo_decorrido > 0 else FS_ESTIMADA
                )
            self.contador_frames = 0
            self.tempo_espera = time.time()

        h, w = imagem.shape[:2]
        tamanho_roi = int(min(h, w) * 0.4)
        x_inicio = (w - tamanho_roi) // 2
        y_inicio = (h - tamanho_roi) // 2
        x_fim, y_fim = x_inicio + tamanho_roi, y_inicio + tamanho_roi

        roi = imagem[y_inicio:y_fim, x_inicio:x_fim]
        if roi.size == 0:
            self.info_status = "Posicione a câmera corretamente."
            return imagem

        media_BGR = np.mean(roi, axis=(0, 1))
        # adiciona RGB (inverte BGR)
        self.C.append(media_BGR[::-1])
        if len(self.C) > TAMANHO_TELA:
            self.C.pop(0)

        media_R, media_G, media_B = media_BGR[2], media_BGR[1], media_BGR[0]
        dedo_na_camera = (
            media_R > media_G + VERMELHO_MINIMO and media_R > media_B + VERMELHO_MINIMO
        )

        # Lógica de estados
        if not dedo_na_camera:
            self.info_status = "Coloque o dedo na câmera (flash ligado)."
            self.valor_bpm.clear()
        elif len(self.C) < TAMANHO_TELA:
            self.info_status = "Coletando dados para análise..."
        elif self.taxa_fps <= 0:
            self.info_status = "Aguardando taxa de quadros válida..."
        else:
            # só processa no intervalo definido (ex: 1s)
            if time.time() - self.last_bpm_time >= CALC_INTERVAL_S:
                C_array = np.array(self.C)
                normalized_signal, detrended, H = aplicar_pos_e_filtrar(
                    C_array, self.taxa_fps
                )
                bpm_atual, snr, freqs, amps = calcular_fft_bpm_snr(
                    normalized_signal, self.taxa_fps
                )

                if bpm_atual is not None:
                    self.valor_bpm.append(bpm_atual)
                    if len(self.valor_bpm) > TAM_SUAVIZACAO_BPM:
                        self.valor_bpm.pop(0)
                    self.media_bpm = float(np.mean(self.valor_bpm))
                    self.info_status = (
                        f"BPM: {self.media_bpm:.1f} BPM | SNR: {snr:.1f} dB"
                    )
                else:
                    self.info_status = (
                        "Sinal fraco — mantenha o dedo imóvel e iluminado."
                    )

                self.last_bpm_time = time.time()

        # desenho da ROI e textos
        cv2.rectangle(imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2)
        cor_texto = (0, 255, 0) if "BPM:" in self.info_status else (0, 0, 255)
        cv2.putText(
            imagem,
            self.info_status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            cor_texto,
            2,
        )
        cv2.putText(
            imagem,
            f"FPS: {self.taxa_fps:.1f}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        return imagem

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_proc = self.process_frame_logic(img)
        return av.VideoFrame.from_ndarray(img_proc, format="bgr24")


# ----------------- Função para processar vídeo enviado -----------------


def process_uploaded_video(uploaded_file, duracao_analise):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = FS_ESTIMADA
        st.warning(f"FPS não detectado. Usando {fps} FPS.")

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        st.error("Erro ao abrir o vídeo.")
        return

    frames_limite = min(int(duracao_analise * fps), total_frames)
    frames_processados = 0
    C = []

    st.subheader(f"Processando vídeo — FPS detectado: {fps:.2f}")
    progress_bar = st.progress(0)

    # ROI central menor (1/4 frame)
    center_x, center_y = largura // 2, altura // 2
    roi_w, roi_h = largura // 4, altura // 4

    while cap.isOpened() and frames_processados < frames_limite:
        ret, frame = cap.read()
        if not ret:
            break
        roi = frame[
            center_y - roi_h : center_y + roi_h, center_x - roi_w : center_x + roi_w
        ]
        if roi.size > 0:
            means = np.mean(roi, axis=(0, 1))
            C.append(means[::-1])
        frames_processados += 1
        progress_bar.progress(frames_processados / frames_limite)

    cap.release()
    progress_bar.empty()
    os.remove(video_path)

    if frames_processados < int(10 * fps):
        st.error("Vídeo muito curto para análise (menos de 10s).")
        return

    C_array = np.array(C)
    normalized_signal, detrended, H = aplicar_pos_e_filtrar(C_array, fps)
    bpm_final, snr_final, fft_freqs, fft_amps = calcular_fft_bpm_snr(
        normalized_signal, fps
    )

    if bpm_final is None:
        st.error(
            "Nenhum batimento válido detectado. Verifique iluminação e imobilidade."
        )
        return

    st.success("Processamento concluído!")
    st.metric("Frequência Cardíaca Estimada", f"{bpm_final:.2f} BPM")
    st.metric("SNR Estimado", f"{snr_final:.2f} dB")

    timestamps = np.arange(len(C_array)) / fps

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sinal PPG Bruto (canal R) vs Normalizado")
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(x=timestamps, y=C_array[:, 0], mode="lines", name="Bruto (R)")
        )
        fig1.add_trace(
            go.Scatter(
                x=timestamps, y=normalized_signal, mode="lines", name="Normalizado"
            )
        )
        fig1.update_layout(xaxis_title="Tempo (s)", yaxis_title="Amplitude")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Espectro de Frequência (FFT)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=fft_freqs, y=fft_amps, mode="lines", name="FFT"))
        fig2.add_vline(
            x=bpm_final, line_dash="dash", annotation_text=f"Pico: {bpm_final:.1f} BPM"
        )
        fig2.update_layout(xaxis_title="Frequência (BPM)", yaxis_title="Amplitude")
        fig2.update_xaxes(range=BPM_RANGE)
        st.plotly_chart(fig2, use_container_width=True)


# ----------------- Interface Streamlit -----------------

st.set_page_config(layout="wide", page_title="rPPG - POS (Otimizado)")
st.title("Extrator rPPG (POS) — Tempo Real e Upload de Vídeo")

with st.sidebar:
    st.header("Instruções Rápidas")
    st.markdown(
        "- No celular: ligue o flash manualmente e coloque o dedo sobre a lente por 10–15s."
    )
    st.markdown(
        "- Se usar notebook, tente um ambiente iluminado e mantenha o dedo imóvel."
    )
    st.markdown(
        "- Navegadores não permitem ligar o flash via código; ative manualmente."
    )
    st.markdown(
        "\nSe quiser, posso adicionar um componente de gravação direto no navegador—me diga que eu implemento."
    )

st.sidebar.header("Modo de Operação")
modo = st.sidebar.radio("Selecionar modo:", ("Tempo Real (Webcam)", "Upload de Vídeo"))

if modo == "Tempo Real (Webcam)":
    st.header("Análise em Tempo Real (Webcam)")
    st.info(
        "Clique em START (permite a câmera). Coloque o dedo sobre a lente. O cálculo inicia automaticamente quando o dedo for detectado."
    )

    webrtc_ctx = webrtc_streamer(
        key="bpm-live-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        desired_playing_state=True,
    )

    st.markdown("---")
    st.markdown("**Dicas de diagnóstico:**")
    st.write(
        "- Se o valor estiver muito instável, tente aumentar a luz e manter o dedo imóvel."
    )
    st.write(
        "- Se FPS detectado for < 15, garanta que a câmera não está ocupada por outro app."
    )

elif modo == "Upload de Vídeo":
    st.header("Análise por Vídeo Gravado")
    st.info(
        "Faça upload de um vídeo de 10–60s com o dedo cobrindo a câmera (flash ligado) para análise offline."
    )

    uploaded_file = st.file_uploader(
        "Escolha um arquivo de vídeo", type=["mp4", "mov", "avi"]
    )
    duracao_analise = st.slider(
        "Duração do vídeo a analisar (s)", min_value=10, max_value=60, value=15, step=5
    )

    if uploaded_file is not None:
        # mostra preview (opcional)
        tfile_show = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_show.write(uploaded_file.read())
        tfile_show.close()
        st.video(tfile_show.name)
        # reabre o arquivo (streamlit já consumiu o .read() acima) -> read() esgotou, então precisa re-obter do uploaded_file
        # Simplificação: pedimos para o usuário recarregar o arquivo ao clicar no botão.
        if st.button("Extrair Sinal PPG"):
            # Para garantir o arquivo completo, pedimos ao usuário a seleção de novo (isso evita problemas de leitura dupla)
            with st.spinner("Processando vídeo..."):
                # Re-processa: uploaded_file já foi lido para preview; usamos tfile_show (arquivo temporário salvo)
                # Reabra o arquivo salvo e envie para a função
                with open(tfile_show.name, "rb") as f:

                    class DummyUploaded:
                        def __init__(self, data):
                            self._data = data

                        def read(self):
                            return self._data

                    data_bytes = f.read()
                    dummy = DummyUploaded(data_bytes)
                    process_uploaded_video(dummy, duracao_analise)
        # remove arquivo temporário de preview
        try:
            os.remove(tfile_show.name)
        except Exception:
            pass

# ----------------- Fim do arquivo -----------------
