import streamlit as st
import cv2
import numpy as np
from scipy.fft import (
    fft,
    fftfreq,
)  # Para FFT (Transformada Rápida de Fourier) [cite: 75]
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoHTMLAttributes
import av  # Biblioteca para manipulação de frames do WebRTC

# --- Constantes ---
TAMANHO_BUFFER = 100  # Número de frames para análise de PPG (janela deslizante)
FPS_PADRAO = 30
BPM_FAIXA_MIN_HZ = 0.67  # 40 BPM / 60s (Limites do filtro passa-banda) [cite: 14]
BPM_FAIXA_MAX_HZ = 3.0  # 180 BPM / 60s (Adaptado da faixa 40-240 BPM [cite: 14])
VERMELHO_MINIMO = (
    20  # Nível de vermelho para detectar o dedo (rPPG do tipo "dedo na câmera")
)

# --- Classe para Processamento de Vídeo em Tempo Real (rPPG) ---


# Herda de VideoProcessorBase para ser compatível com streamlit-webrtc
class RPPGProcessor(VideoProcessorBase):
    def __init__(self):
        # Inicialização dos buffers para armazenar as médias de cor
        self.buffer_B, self.buffer_G, self.buffer_R = [], [], []
        # Buffer para armazenar os BPMs calculados para suavização
        self.valor_bpm = []
        # Taxa de frames (FPS) inicial, será atualizada no processo
        self.taxa_fps = FPS_PADRAO
        # Mensagens de estado para a interface
        self.status_mensagem = "Aguardando inicialização..."
        self.status_cor = (255, 255, 255)  # Branco

    # Método para processar cada frame
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Converter o frame AV para um array NumPy (BGR)
        imagem = frame.to_ndarray(format="bgr24")

        # 2. Configuração da Região de Interesse (ROI) - Adaptado do seu código
        (altura, largura) = imagem.shape[:2]
        tamanho_roi = int(min(altura, largura) * 0.5)

        # Centralizar a ROI
        x_inicio = (largura - tamanho_roi) // 2
        y_inicio = (altura - tamanho_roi) // 2
        x_fim = x_inicio + tamanho_roi
        y_fim = y_inicio + tamanho_roi

        roi_central = imagem[y_inicio:y_fim, x_inicio:x_fim]

        # 3. Extrair Média BGR da ROI
        if roi_central.size > 0:
            # Média de pixels (Extração do Sinal Bruto [cite: 10])
            media_BGR = np.mean(roi_central, axis=(0, 1))
        else:
            media_BGR = np.mean(imagem, axis=(0, 1))
            cv2.putText(
                imagem,
                "Aviso: ROI Invalida",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Desenhar o retângulo da ROI (verde)
        cv2.rectangle(imagem, (x_inicio, y_inicio), (x_fim, y_fim), (0, 255, 0), 2)

        # 4. Atualizar Buffers (Séries Temporais C_r, C_g, C_b [cite: 11])
        self.buffer_B.append(media_BGR[0])
        self.buffer_G.append(media_BGR[1])
        self.buffer_R.append(media_BGR[2])

        # Manter o tamanho fixo (janela deslizante)
        if len(self.buffer_B) > TAMANHO_BUFFER:
            self.buffer_B.pop(0)
            self.buffer_G.pop(0)
            self.buffer_R.pop(0)
            # Mantém o buffer de BPM no mesmo tamanho relativo
            if len(self.valor_bpm) > (TAMANHO_BUFFER / 10):
                self.valor_bpm.pop(0)

        # 5. Processamento PPG e Cálculo de BPM (Só quando o buffer está cheio)
        media_bpm = 0
        self.status_mensagem = "Coloque o dedo na câmera"
        self.status_cor = (0, 0, 255)  # Vermelho

        if len(self.buffer_B) == TAMANHO_BUFFER:

            # 5a. Detecção de Dedo/Cor Vermelha (rPPG tipo dedo na câmera)
            media_R_recente = media_BGR[2]
            media_G_recente = media_BGR[1]
            media_B_recente = media_BGR[0]

            dedo_na_camera = (media_R_recente > media_G_recente + VERMELHO_MINIMO) and (
                media_R_recente > media_B_recente + VERMELHO_MINIMO
            )

            if dedo_na_camera:
                try:
                    # 5b. Normalização e Projeção POS (Adaptado do seu código)
                    # Normalização: Divide por média (remove parte constante DC [cite: 154])
                    Rnorm = np.array(self.buffer_R) / (np.mean(self.buffer_R) + 1e-9)
                    Gnorm = np.array(self.buffer_G) / (np.mean(self.buffer_G) + 1e-9)
                    Bnorm = np.array(self.buffer_B) / (np.mean(self.buffer_B) + 1e-9)

                    # Vetores S1 e S2 (similar ao POS/Método de Poh [cite: 25, 162])
                    S1 = Gnorm - Bnorm
                    S2 = -2 * Rnorm + Gnorm + Bnorm

                    # Fator de ajuste alpha (calcula o desvio padrão/proporção [cite: 156])
                    alpha = np.std(S1) / (np.std(S2) + 1e-9)
                    # Combinação (cancelando ruído [cite: 157])
                    h = S1 + (alpha * S2)

                    # Remover componente DC (Detrending básico [cite: 13, 201])
                    entrada_trf = h - np.mean(h)

                    # 5c. Cálculo de BPM com FFT (Transformada Rápida de Fourier [cite: 17, 75])
                    fs_fft = self.taxa_fps

                    if fs_fft > 0:
                        trf = np.fft.fft(entrada_trf)
                        frequencias = np.fft.fftfreq(TAMANHO_BUFFER, d=1.0 / fs_fft)

                        # 5d. Seleção da Faixa de BPM (Filtro Passa-Banda [cite: 14, 203])
                        indices_validos = np.where(
                            (frequencias >= BPM_FAIXA_MIN_HZ)
                            & (frequencias <= BPM_FAIXA_MAX_HZ)
                        )

                        if len(indices_validos[0]) > 0:
                            # 5e. Encontrar o pico de frequência (Frequência Dominante [cite: 17])
                            indice_pico = np.argmax(np.abs(trf[indices_validos]))
                            freq_dominante = frequencias[
                                indices_validos[0][indice_pico]
                            ]
                            bpm_atual = freq_dominante * 60

                            self.valor_bpm.append(bpm_atual)

                            # 5f. Média Suavizada
                            media_bpm = np.mean(self.valor_bpm)

                            self.status_mensagem = f"BPM: {media_bpm:.1f}"
                            self.status_cor = (0, 255, 0)  # Verde
                        else:
                            self.status_mensagem = "Pulso fraco/Ruído alto"
                            self.status_cor = (0, 165, 255)  # Laranja

                except Exception as e:
                    self.status_mensagem = "Erro no Cálculo PPG"
                    self.status_cor = (0, 0, 255)  # Vermelho

            else:
                # Se o dedo não for detectado
                self.valor_bpm = []
                self.status_mensagem = "Aguardando dedo na câmera"
                self.status_cor = (0, 0, 255)  # Vermelho

        # 6. Exibir Status e BPM no Frame (Substituindo cv2.imshow)
        cv2.putText(
            imagem,
            self.status_mensagem,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            self.status_cor,
            3,
        )
        cv2.putText(
            imagem,
            f"Buffer: {len(self.buffer_R)}/{TAMANHO_BUFFER}",
            (10, altura - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # 7. Retornar o frame processado
        return av.VideoFrame.from_ndarray(imagem, format="bgr24")


# --- Interface Streamlit ---

st.set_page_config(layout="wide")
st.title("Monitor de Frequência Cardíaca (rPPG) em Tempo Real")

st.info(
    """
    Este aplicativo utiliza a webcam do seu dispositivo e a técnica de Fotopletismografia Remota (rPPG)
    para estimar sua frequência cardíaca[cite: 48]. O método é otimizado para a técnica
    "dedo na câmera" (transmissão), baseada em princípios como o POS (Plane-Orthogonal-to-Skin)[cite: 28, 31].
    **Instruções:**
    1. Cubra a lente da câmera do seu celular/webcam (com o flash ligado, se for celular) com a ponta do dedo.
    2. O BPM será exibido no vídeo assim que o buffer de frames (janela de análise) estiver cheio.
    """
)

# Colunas para organizar o layout
col1, col2 = st.columns([3, 1])

with col1:
    # Acessa a câmera
    webrtc_streamer(
        key="rppg-stream",
        video_processor_factory=RPPGProcessor,
        # O WebRTC pede permissão para acessar o vídeo, e o audio é desativado
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=False,
            style={"width": "100%", "border": "3px solid #00f"},
        ),
    )

with col2:
    st.subheader("Controle")

    # Adicionando um botão de "Encerrar" com uma KEY única para resolver o erro
    if st.button("Encerrar Aplicação", key="unique_stop_button"):
        st.error("Aplicação encerrada. Recarregue a página para reiniciar.")
        # Como o Streamlit não tem um comando nativo de "stop",
        # a solução mais simples é exibir a mensagem de erro e desabilitar o script.
        st.stop()
