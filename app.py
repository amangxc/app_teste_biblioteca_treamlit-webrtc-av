import streamlit as st
import cv2
import numpy as np
import time
import av
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Importações para Processamento de Sinal (Baseado no Tutorial) ---
from scipy.fft import fft, fftfreq  # Para Transformada Rápida de Fourier (FFT)
from scipy.signal import detrend, butter, filtfilt  # Para Limpar e Filtrar Tendências
import plotly.graph_objects as go  # Para Gráficos Interativos

# --- Importações solicitadas para WebRTC ---
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# --- Parâmetros de 'Paulo' / Tutorial POS ---
TAMANHO_TELA = 100  # Tamanho do Buffer
VERMELHO_MINIMO = 60  # Nível de vermelho mínimo para detectar o dedo (Lógica do Paulo)
BPM_RANGE = [40, 240]  # Faixa de BPM humano (0.67 Hz a 4 Hz)
FS_ESTIMADA = 30  # FPS padrão
TAM_JANELA_POS_S = 1.6  # Tamanho da janela deslizante do POS em segundos
TAM_SUAVIZACAO_BPM = 15  # Tamanho da lista de BPM para calcular a média

# --- FUNÇÕES POS PRINCIPAIS (Extraídas do Tutorial) ---


def aplicar_pos_e_filtrar(C, fps):
    """Aplica o método POS e faz o Detrend/Filtro Band-Pass."""
    frame_count = len(C)

    # 1. Aplicar POS com Janela Deslizante
    l = int(fps * TAM_JANELA_POS_S)  # Tamanho janela em frames (~1.6s)
    H = np.zeros(frame_count)  # Array para sinal PPG final

    # Matriz de Projeção POS (R-G, R-B)
    pp = np.array([[1, -1, 0], [1, 0, -1]])

    for n in range(frame_count):
        m = max(0, n - l + 1)
        if n - m + 1 == l:  # Janela cheia
            window_C = C[m : n + 1, :]  # Pedaço da matriz C

            mu = np.mean(window_C, axis=0)  # Média por coluna (R,G,B)
            Cn = window_C / mu[np.newaxis, :]  # Normalização (média=1)

            S = pp @ Cn.T  # Projeta em 2 direções (S1, S2)
            S1, S2 = S[0, :], S[1, :]

            sigma1 = np.std(S1)  # Desvio padrão S1
            sigma2 = np.std(S2)  # Desvio padrão S2
            alpha = sigma1 / sigma2 if sigma2 != 0 else 0  # Alfa ajusta pesos

            h = S1 + alpha * S2  # Combina ondas (soma pulso, cancela ruído)
            h -= np.mean(h)  # Tira média (zero-média: centra onda em 0)

            H[m : n + 1] += h  # Adiciona a H (overlap-add: suaviza junções)

    # 2. Processamento do Sinal (Detrend, Filtro Band-Pass, Normalização)
    detrended_signal = detrend(H)  # Remove tendência linear

    nyquist_freq = 0.5 * fps  # Limite de frequência (fps/2)
    low_cutoff = BPM_RANGE[0] / 60.0 / nyquist_freq  # Converte BPM para normalizado
    high_cutoff = BPM_RANGE[1] / 60.0 / nyquist_freq

    # Cria filtro Band-Pass (ordem 4)
    b, a = butter(4, [low_cutoff, high_cutoff], btype="band")

    # Aplica filtro (filtfilt aplica duas vezes para precisão)
    filtered_signal = filtfilt(b, a, detrended_signal)

    # Normaliza (média=0, std=1)
    if np.std(filtered_signal) != 0:
        normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(
            filtered_signal
        )
    else:
        normalized_signal = filtered_signal

    return normalized_signal, detrended_signal, H  # Retorna o sinal limpo


def calcular_fft_bpm_snr(normalized_signal, fps):
    """Calcula BPM e SNR via FFT"""
    N = len(normalized_signal)
    yf = fft(normalized_signal)  # FFT
    xf = fftfreq(N, 1 / fps)  # Frequências

    freqs = xf[xf > 0]  # Só positivas
    amps = 2.0 / N * np.abs(yf[0 : N // 2])[1 : N // 2 + 1]  # Amplitudes

    # Faixa de BPM (40-240 BPM)
    valid_indices = np.where(
        (freqs * 60 >= BPM_RANGE[0]) & (freqs * 60 <= BPM_RANGE[1])
    )
    valid_freqs = freqs[valid_indices]
    valid_amps = amps[valid_indices]

    if len(valid_freqs) == 0:
        return None, None, None, None  # Se não encontrou pico

    peak_freq = valid_freqs[np.argmax(valid_amps)]
    bpm = peak_freq * 60  # Hz para BPM

    # Calcular SNR
    fundamental = np.max(valid_amps)

    # Simples cálculo de energia (sinal = pico; ruído = resto da banda)
    signal_energy = fundamental
    noise_energy = np.sum(valid_amps) - signal_energy

    snr = 10 * np.log10(signal_energy / noise_energy) if noise_energy > 0 else 0

    return (
        bpm,
        snr,
        freqs * 60,
        amps,
    )  # Retorna BPM, SNR, Frequências (BPM) e Amplitudes


# --- CLASSE DE PROCESSAMENTO EM TEMPO REAL (WebRTC) ---


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.C = []  # Lista para guardar médias RGB (Sinal Bruto)
        self.valor_bpm = []
        self.contador_frames = 0
        self.tempo_espera = time.time()
        self.taxa_fps = FS_ESTIMADA  # FPS estimado
        self.media_bpm = 0.0
        self.info_status = "Aguardando START..."

    def process_frame_logic(self, imagem):
        """Lógica de Aquisição de ROI/Média RGB do Paulo, adaptada para WebRTC."""
        self.contador_frames += 1

        # 1. Cálculo de FPS (a cada 60 frames)
        if self.contador_frames % 60 == 0:
            cont_final = time.time()
            tempo_decorrido = cont_final - self.tempo_espera
            if tempo_decorrido > 0:
                self.taxa_fps = self.contador_frames / tempo_decorrido
            self.contador_frames = 0
            self.tempo_espera = time.time()

        # 2. Definição da ROI (Lógica do Paulo, focada no dedo)
        (largura, altura) = imagem.shape[:2]
        tamanho_roi = int(largura * 0.5)
        x_inicio = (altura - tamanho_roi) // 2
        y_inicio = (largura - tamanho_roi) // 2
        x_fim = x_inicio + tamanho_roi
        y_fim = y_inicio + tamanho_roi

        x_inicio = max(0, x_inicio)
        y_inicio = max(0, y_inicio)
        x_fim = min(altura, x_fim)
        y_fim = min(largura, y_fim)

        # 3. Análise da Média BGR na ROI
        media_BGR = np.mean(imagem[y_inicio:y_fim, x_inicio:x_fim], axis=(0, 1))

        # Adicionar o frame atual
        self.C.append(media_BGR[::-1])  # Inverte BGR para RGB

        # 4. Manutenção do Buffer (Tamanho Fixo)
        if len(self.C) > TAMANHO_TELA:
            self.C.pop(0)

        # 5. Cálculo do BPM (Usa a lógica de POS do Tutorial se o buffer estiver cheio)
        if len(self.C) == TAMANHO_TELA:

            # --- Validação Simples de Dedo (Lógica do Paulo) ---
            media_R_recente = media_BGR[2]
            media_G_recente = media_BGR[1]
            media_B_recente = media_BGR[0]
            dedo_na_camera = (
                media_R_recente > media_G_recente + VERMELHO_MINIMO
                and media_R_recente > media_B_recente + VERMELHO_MINIMO
            )

            if dedo_na_camera and self.taxa_fps > 0:
                C_array = np.array(self.C)

                # Aplica POS e Filtros (Do Tutorial)
                normalized_signal, _, _ = aplicar_pos_e_filtrar(C_array, self.taxa_fps)

                # Calcula FFT/BPM (Do Tutorial)
                bpm_atual, _, _, _ = calcular_fft_bpm_snr(
                    normalized_signal, self.taxa_fps
                )

                if bpm_atual is not None:
                    self.valor_bpm.append(bpm_atual)
                    if len(self.valor_bpm) > TAM_SUAVIZACAO_BPM:
                        self.valor_bpm.pop(0)

                    self.media_bpm = np.mean(self.valor_bpm)
                    self.info_status = f"O seu BPM estah em: {self.media_bpm:.1f}"
                else:
                    self.info_status = (
                        "Sinal fraco. Mantenha o dedo imóvel e iluminado."
                    )
            else:
                self.valor_bpm = []
                self.info_status = "Coloque o dedo na câmera e ligue o flash."
        else:
            self.info_status = "Coletando dados..."

        # 6. Exibição na Imagem
        cv2.rectangle(imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2)
        cv2.putText(
            imagem,
            self.info_status,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if "BPM" in self.info_status else (0, 0, 255),
            2,
        )
        cv2.putText(
            imagem,
            f"FPS: {self.taxa_fps:.1f}",
            (largura - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        return imagem

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Converte o frame do AV para array NumPy (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Processa a imagem com a lógica
        img_processada = self.process_frame_logic(img)

        # Retorna o frame processado
        return av.VideoFrame.from_ndarray(img_processada, format="bgr24")


# --- FUNÇÃO DE PROCESSAMENTO DE VÍDEO UPLOADED (Adaptação para o POS do Tutorial) ---


def process_uploaded_video(uploaded_file, duracao_analise):
    # Salva o arquivo temporariamente
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # Pega FPS
    if fps == 0:
        fps = FS_ESTIMADA  # Usa padrão se falhar
        st.warning(f"FPS não detectado. Usando {fps} FPS.")

    largura_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened() or fps == 0:
        st.error("Erro ao abrir ou ler o vídeo.")
        return

    frames_limite = int(duracao_analise * fps)
    frames_limite = min(frames_limite, total_frames)
    frames_processados = 0

    C = []  # Lista para guardar médias RGB (Sinal Bruto)

    st.subheader(f"Processando Vídeo - Taxa de Quadros (FPS): {fps:.2f}")

    progress_bar = st.progress(0)

    # 1. Lógica do ROI (adaptada do Tutorial)
    h, w = altura_frame, largura_frame
    center_x, center_y = w // 2, h // 2  # Centro
    roi_w, roi_h = w // 4, h // 4  # Tamanho ROI (1/4)

    while cap.isOpened() and frames_processados < frames_limite:
        retorno, frame = cap.read()
        if not retorno:
            break

        # Corta ROI central
        roi = frame[
            center_y - roi_h : center_y + roi_h, center_x - roi_w : center_x + roi_w
        ]

        if roi.size > 0:  # Se ROI não vazio
            means = np.mean(roi, axis=(0, 1))  # Média de pixels
            C.append(means[::-1])  # Inverte BGR para RGB

        frames_processados += 1
        progress_bar.progress(frames_processados / frames_limite)

    cap.release()
    progress_bar.empty()
    os.remove(video_path)  # Limpa o arquivo temporário

    if frames_processados < int(10 * fps):  # Video curto demais (<10s)
        st.error("Vídeo muito curto (analisado menos de 10s).")
        return

    C_array = np.array(C)  # Converte lista em matriz NumPy

    # --- Aplicar POS e Processamento de Sinal (Do Tutorial) ---
    normalized_signal, _, _ = aplicar_pos_e_filtrar(C_array, fps)

    # --- Calcular FFT, BPM e SNR (Do Tutorial) ---
    bpm_final, snr_final, fft_freqs, fft_amps = calcular_fft_bpm_snr(
        normalized_signal, fps
    )

    if bpm_final is None:
        st.error(
            "Nenhum batimento cardíaco válido foi detectado. Verifique a iluminação e a imobilidade do dedo/rosto."
        )
        return

    # --- Geração dos Gráficos (Do Tutorial) ---
    st.success("Processamento concluído!")

    # Métricas
    st.metric("Frequência Cardíaca Estimada", f"{bpm_final:.2f} BPM")
    st.metric("SNR Estimado", f"{snr_final:.2f} dB")

    col1, col2 = st.columns(2)  # Duas colunas

    timestamps = np.arange(len(C_array)) / fps  # Tempo em segundos

    # Gráfico 1: Sinal PPG Bruto vs. Filtrado (Série Temporal)
    with col1:
        st.subheader("Sinal PPG Bruto vs. Filtrado")
        fig1 = go.Figure()
        # Sinal Bruto (Canal Vermelho)
        fig1.add_trace(
            go.Scatter(
                x=timestamps,
                y=C_array[:, 0],
                mode="lines",
                name="Sinal Bruto (Canal Vermelho)",
            )
        )
        # Sinal Filtrado e Normalizado
        fig1.add_trace(
            go.Scatter(
                x=timestamps,
                y=normalized_signal,
                mode="lines",
                name="Sinal Filtrado e Normalizado",
            )
        )
        fig1.update_layout(
            title="Sinal PPG ao Longo do Tempo",
            xaxis_title="Tempo (s)",
            yaxis_title="Amplitude",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Análise de Frequência (FFT)
    with col2:
        st.subheader("Análise de Frequência (FFT)")
        fig2 = go.Figure()
        # Espectro de Frequência
        fig2.add_trace(
            go.Scatter(
                x=fft_freqs, y=fft_amps, mode="lines", name="Espectro de Frequência"
            )
        )
        # Linha do Pico (BPM)
        fig2.add_vline(
            x=bpm_final,
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Pico: {bpm_final:.1f} BPM",
        )
        fig2.update_layout(
            title="Espectro de Frequência do Sinal PPG",
            xaxis_title="Frequência (BPM)",
            yaxis_title="Amplitude",
        )
        fig2.update_xaxes(range=BPM_RANGE)
        st.plotly_chart(fig2, use_container_width=True)


# --- INTERFACE STREAMLIT ---
st.set_page_config(layout="wide")
st.title("Extrator de Sinal PPG e Frequência Cardíaca de Vídeo (rPPG com POS)")

st.sidebar.header("Modo de Operação")
modo = st.sidebar.radio(
    "Selecione o modo de análise:", ("Tempo Real (Webcam)", "Upload de Vídeo")
)

if modo == "Tempo Real (Webcam)":
    st.header("Análise em Tempo Real (Webcam)")
    st.info(
        "⚠️ **Instrução:** Clique em 'START' e coloque seu dedo indicador sobre a câmera do celular/webcam. Se estiver usando o celular, **ATIVE o flash manualmente** para melhor iluminação (o app usa a luz refletida/transmitida, o flash aumenta o sinal PPG)."
    )

    webrtc_streamer(
        key="bpm-live-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

elif modo == "Upload de Vídeo":
    st.header("Análise de Vídeo Gravado")
    st.info(
        "Faça o upload de um vídeo de **10 a 15 segundos** onde seu dedo cobre a câmera e o flash está ligado para análise do sinal de pulso (PPG). O app usa o método **POS** para alta precisão, conforme o artigo 'Algorithmic Principles of Remote-PPG'."
    )

    uploaded_file = st.file_uploader(
        "Escolha um arquivo de vídeo", type=["mp4", "mov", "avi"]
    )

    # Permite a escolha da duração
    duracao_analise = st.slider(
        "Duração do vídeo para análise (segundos):",
        min_value=10,
        max_value=60,
        value=15,
        step=5,
    )

    if uploaded_file is not None:
        # Mostra o vídeo (opcional)
        tfile_show = tempfile.NamedTemporaryFile(delete=False)
        tfile_show.write(uploaded_file.read())
        tfile_show.close()
        st.video(tfile_show.name)
        os.remove(tfile_show.name)

        if st.button("Extrair Sinal PPG"):
            with st.spinner("Processando..."):  # Spinner de loading
                process_uploaded_video(uploaded_file, duracao_analise)
