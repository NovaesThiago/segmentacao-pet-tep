# Segmentação de Instância de Pets com YOLOv8-seg

**Trabalho Final — Tópicos Especiais em Programação (TEC.1053)**
Instituto Federal do Piauí — Campus Picos · Prof. Daniel de Sousa Luz

**Grupo 1 — Tema: Segmentação**
Katelyn Moura, Diego Rodrigues, Idelmar Júnior, Renan Costa, Thiago Novaes

---

## 1. Contextualização

Segmentação é a tarefa de visão computacional que classifica **cada pixel** de uma imagem,
indo além da classificação (que só diz "o que" aparece) e da detecção (que só diz "onde", com
uma caixa delimitadora). Na **segmentação de instância**, além de identificar as classes,
o modelo distingue **cada objeto individualmente** com sua própria máscara — diferente da
segmentação semântica, que só rotula classes sem separar instâncias.

Neste projeto, treinamos um modelo para detectar e segmentar **pets** (cães e gatos) em
fotos, gerando a máscara de pixels exata de cada animal encontrado na imagem.

## 2. Metodologia

- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 7.349
  imagens de 37 raças, cada uma com uma máscara de segmentação (*trimap*: pet / fundo / borda)
  baixada automaticamente pelo `torchvision`. Como o YOLOv8-seg espera rótulos em **polígonos**
  (não máscaras densas), o script [`src/prepare_dataset.py`](src/prepare_dataset.py) converte
  cada trimap em um contorno (via `cv2.findContours`) e exporta no formato YOLO-seg
  (`images/` + `labels/*.txt` com uma classe: `pet`).
- **Arquitetura:** **YOLOv8n-seg**, da biblioteca [`ultralytics`](https://docs.ultralytics.com/tasks/segment/) —
  uma rede *single-stage* que prediz simultaneamente caixas delimitadoras, classes e um
  protótipo de máscara por instância, combinados por coeficientes preditos para cada detecção.
  Partimos do checkpoint `yolov8n-seg.pt` pré-treinado no COCO (transfer learning), o que acelera
  bastante a convergência com poucas épocas.
- **Treinamento:** gerenciado internamente pelo `ultralytics` (`model.train(...)`), que já inclui
  data augmentation, *loss* combinada (caixa + classe + máscara) e *learning rate schedule*.
- **Métricas de avaliação:** mAP50 e mAP50-95 de máscara (padrão do `ultralytics`, equivalentes
  ao usado no benchmark COCO), além de precisão e recall.

## 3. Pré-requisitos

- Python 3.10+
- GPU recomendada (o projeto foi desenhado para rodar no **Google Colab gratuito**, mas também
  funciona em CPU/GPU local — apenas mais lento)
- Dependências listadas em [`requirements.txt`](requirements.txt)

## 4. Passo a passo de execução

### Opção A — Google Colab (recomendado)

1. Abra [`notebooks/treinamento_colab.ipynb`](notebooks/treinamento_colab.ipynb) no Colab.
2. Em `Ambiente de execução > Alterar tipo de ambiente de execução`, selecione **GPU**.
3. Atualize a variável `REPO_URL` na primeira célula com a URL deste repositório.
4. Execute as células em ordem: instalação → preparação do dataset → treino → avaliação →
   geração de amostras visuais.

### Opção B — Ambiente local

```bash
git clone <url-deste-repositorio>
cd segmentacao-pets-tep
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Baixa o Oxford-IIIT Pet e converte trimaps -> polígonos YOLO-seg
python -m src.prepare_dataset

# 2) Treina o modelo (parte de yolov8n-seg.pt pré-treinado no COCO)
python -m src.train --epochs 30 --batch 16

# 3) Avalia no conjunto de teste
python -m src.evaluate --weights outputs/runs/train/weights/best.pt --split test

# 4) Gera imagens de exemplo com a máscara prevista sobreposta
python -m src.predict --weights outputs/runs/train/weights/best.pt --num-samples 6
```

Os resultados visuais são salvos em `outputs/samples/`; os pesos e gráficos de treino ficam em
`outputs/runs/train/`.

## 5. Estrutura do repositório

```
segmentacao-pets-tep/
├── README.md
├── requirements.txt
├── src/
│   ├── prepare_dataset.py   # Oxford-IIIT Pet -> polígonos no formato YOLO-seg
│   ├── train.py               # Treina o YOLOv8-seg (via ultralytics)
│   ├── evaluate.py             # Avalia métricas de máscara no split escolhido
│   └── predict.py               # Gera amostras visuais com a máscara prevista
├── notebooks/
│   └── treinamento_colab.ipynb
└── outputs/
    ├── runs/          # Gerado pelo ultralytics: pesos, curvas de treino, logs
    └── samples/        # Imagens de exemplo com a predição sobreposta
```

## 6. Resultados

_A preencher após o treinamento com os números finais de mAP50/mAP50-95 (máscara) obtidos no
conjunto de teste, e 2-3 imagens de exemplo da pasta `outputs/samples/`._

| Métrica | Valor |
|---|---|
| mAP50 (máscara, teste) | — |
| mAP50-95 (máscara, teste) | — |
| Precisão / Recall | — |

## 7. Dificuldades encontradas

_A preencher pelo grupo com base na experiência real de desenvolvimento (dados, hardware,
hiperparâmetros), conforme exigido no roteiro da apresentação._
