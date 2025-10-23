import streamlit as st
import cv2
import numpy as np
import time
import av
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, butter, filtfilt
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# ==============================================================
# PARÂMETROS GERAIS
# ==============================================================
TAMANHO_TELA = 100
VERMELHO_MINIMO = 60
BPM_RANGE = [40, 240]
FS_ESTIMADA = 30
TAM_JANELA_POS_S = 1.6
TAM_SUAVIZACAO_BPM = 15

# ==============================================================
# FUNÇÕES PRINCIPAIS DO MÉTODO POS
# ==============================================================


def aplicar_pos_e_filtrar(C, fps):
    frame_count = len(C)
    l = int(fps * TAM_JANELA_POS_S)
    H = np.zeros(frame_count)
    pp = np.array([[1, -1, 0], [1, 0, -1]])

    for n in range(frame_count):
        m = max(0, n - l + 1)
        if n - m + 1 == l:
            window_C = C[m : n + 1, :]
            mu = np.mean(window_C, axis=0)
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
    b, a = butter(4, [low_cutoff, high_cutoff], btype="band")
    filtered_signal = filtfilt(b, a, detrended_signal)
    normalized_signal = (
        (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
        if np.std(filtered_signal) != 0
        else filtered_signal
    )

    return normalized_signal, detrended_signal, H


def calcular_fft_bpm_snr(normalized_signal, fps):
    N = len(normalized_signal)
    yf = fft(normalized_signal)
    xf = fftfreq(N, 1 / fps)
    freqs = xf[xf > 0]
    amps = 2.0 / N * np.abs(yf[0 : N // 2])[1 : N // 2 + 1]
    valid_indices = np.where(
        (freqs * 60 >= BPM_RANGE[0]) & (freqs * 60 <= BPM_RANGE[1])
    )
    valid_freqs = freqs[valid_indices]
    valid_amps = amps[valid_indices]

    if len(valid_freqs) == 0:
        return None, None, None, None

    peak_freq = valid_freqs[np.argmax(valid_amps)]
    bpm = peak_freq * 60
    signal_energy = np.max(valid_amps)
    noise_energy = np.sum(valid_amps) - signal_energy
    snr = 10 * np.log10(signal_energy / noise_energy) if noise_energy > 0 else 0

    return bpm, snr, freqs * 60, amps


# ==============================================================
# PROCESSAMENTO EM TEMPO REAL (WEBCAM)
# ==============================================================


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.C = []
        self.valor_bpm = []
        self.contador_frames = 0
        self.tempo_espera = time.time()
        self.taxa_fps = FS_ESTIMADA
        self.media_bpm = 0.0
        self.info_status = "Aguardando START..."

    def process_frame_logic(self, imagem):
        self.contador_frames += 1
        if self.contador_frames % 60 == 0:
            cont_final = time.time()
            tempo_decorrido = cont_final - self.tempo_espera
            if tempo_decorrido > 0:
                self.taxa_fps = self.contador_frames / tempo_decorrido
            self.contador_frames = 0
            self.tempo_espera = time.time()

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

        media_BGR = np.mean(imagem[y_inicio:y_fim, x_inicio:x_fim], axis=(0, 1))
        self.C.append(media_BGR[::-1])
        if len(self.C) > TAMANHO_TELA:
            self.C.pop(0)

        if len(self.C) == TAMANHO_TELA:
            media_R, media_G, media_B = media_BGR[2], media_BGR[1], media_BGR[0]
            dedo_na_camera = (
                media_R > media_G + VERMELHO_MINIMO
                and media_R > media_B + VERMELHO_MINIMO
            )
            if dedo_na_camera and self.taxa_fps > 0:
                C_array = np.array(self.C)
                normalized_signal, _, _ = aplicar_pos_e_filtrar(C_array, self.taxa_fps)
                bpm_atual, _, _, _ = calcular_fft_bpm_snr(
                    normalized_signal, self.taxa_fps
                )
                if bpm_atual is not None:
                    self.valor_bpm.append(bpm_atual)
                    if len(self.valor_bpm) > TAM_SUAVIZACAO_BPM:
                        self.valor_bpm.pop(0)
                    self.media_bpm = np.mean(self.valor_bpm)
                    self.info_status = f"BPM estimado: {self.media_bpm:.1f}"
                else:
                    self.info_status = (
                        "Sinal fraco. Mantenha o dedo imóvel e iluminado."
                    )
            else:
                self.valor_bpm = []
                self.info_status = "Coloque o dedo na câmera e ligue o flash."
        else:
            self.info_status = "Coletando dados..."

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
        )
