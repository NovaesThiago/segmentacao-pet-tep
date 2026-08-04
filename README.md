# Segmentação de instância de pets com YOLOv8-seg

**Trabalho Final — Tópicos Especiais em Programação (TEC.1053)**
Instituto Federal do Piauí — Campus Picos · Prof. Daniel de Sousa Luz

**Grupo 1 — Segmentação:** Katelyn Moura, Diego Rodrigues, Idelmar Júnior, Renan Costa e Thiago Novaes.

## Sobre o projeto

Este projeto treina um modelo de visão computacional para localizar cães e gatos e produzir uma máscara para cada animal encontrado. Diferentemente da classificação, que informa apenas o conteúdo da imagem, e da detecção, que retorna caixas delimitadoras, a segmentação de instância identifica os pixels de cada objeto individualmente.

O pipeline foi construído a partir do experimento registrado no notebook do grupo e organizado em scripts reproduzíveis para preparação dos dados, treino, avaliação e análise qualitativa.

## Metodologia

- **Dataset:** [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), com 7.349 imagens de 37 raças e máscaras do tipo *trimap*.
- **Preparação:** os pixels de pet e borda do trimap formam uma máscara binária. O maior contorno externo é convertido para coordenadas normalizadas no formato YOLO-seg.
- **Divisão dos dados:** 3.128 imagens de treino e 552 de validação, obtidas do split `trainval` com semente 42, além das 3.669 imagens do split oficial de teste.
- **Modelo:** YOLOv8n-seg pré-treinado no COCO, usando *transfer learning*.
- **Treino:** 30 épocas, imagens de 640 × 640, lote 16 e *early stopping* com `patience=15`.
- **Avaliação:** precisão, recall, mAP50 e mAP50-95 para caixas e máscaras.

## Requisitos

- Python 3.10 ou superior;
- dependências de [`requirements.txt`](requirements.txt);
- GPU recomendada. O experimento original foi executado no Google Colab com uma Tesla T4.

## Execução no Google Colab

Abra [`notebooks/treinamento_colab.ipynb`](notebooks/treinamento_colab.ipynb), atualize a variável `REPO_URL`, selecione uma GPU e execute as células em ordem. O notebook cobre todo o fluxo, inclusive a validação visual das anotações, a análise dos casos de baixa confiança e o download dos resultados.

## Execução local

```bash
git clone <url-deste-repositorio>
cd segmentacao-pet-tep
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instale as dependências e execute o pipeline:

```bash
pip install -r requirements.txt

# 1. Baixar o dataset e converter os trimaps para polígonos YOLO-seg
python -m src.prepare_dataset

# 2. Conferir visualmente uma amostra das anotações convertidas
python -m src.validate_annotations --num-samples 6

# 3. Treinar o modelo
python -m src.train --epochs 30 --imgsz 640 --batch 16 --patience 15

# 4. Avaliar no split oficial de teste
python -m src.evaluate --weights outputs/runs/train/weights/best.pt --split test

# 5. Salvar predições de exemplo
python -m src.predict --weights outputs/runs/train/weights/best.pt --num-samples 6 --conf 0.5

# 6. Encontrar os seis casos de menor confiança, processando em lotes
python -m src.analyze_errors --weights outputs/runs/train/weights/best.pt --num-cases 6 --batch-size 16
```

Os pesos, gráficos e logs do Ultralytics ficam em `outputs/runs/train/`. As inspeções de rótulo, predições e análises de erro ficam, respectivamente, em `outputs/annotation_checks/`, `outputs/samples/` e `outputs/error_cases/`.

Use `--help` em qualquer módulo para consultar todos os parâmetros. Por exemplo:

```bash
python -m src.train --help
```

## Resultados registrados

O notebook fornecido contém uma execução de 30 épocas em uma Tesla T4. Os valores abaixo foram obtidos nas **552 imagens de validação**; portanto, documentam o experimento original, mas não substituem uma avaliação final no split oficial de teste.

| Saída | Precisão | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Bounding box | 0,9920 | 0,9909 | 0,9944 | 0,8795 |
| Máscara | 0,9938 | 0,9928 | 0,9945 | 0,8501 |

O treinamento levou aproximadamente **0,588 hora** (cerca de 35 minutos) no ambiente registrado pelo notebook.

## Estrutura do repositório

```text
segmentacao-pet-tep/
├── README.md
├── requirements.txt
├── src/
│   ├── prepare_dataset.py       # trimaps para polígonos YOLO-seg
│   ├── validate_annotations.py  # inspeção visual dos rótulos
│   ├── train.py                 # treinamento e early stopping
│   ├── evaluate.py              # métricas de caixas e máscaras
│   ├── predict.py               # amostras de inferência
│   └── analyze_errors.py        # casos de menor confiança em lotes
├── notebooks/
│   └── treinamento_colab.ipynb
├── artigo/
│   ├── artigo.tex
│   └── artigo.pdf
└── outputs/                     # artefatos gerados durante a execução
```

## Dificuldades e decisões

- As anotações originais são máscaras densas, enquanto o YOLO-seg exige polígonos. A conversão usa o contorno externo do pet e inclui a classe de borda para fechar a silhueta.
- A inferência sobre todo o conjunto de avaliação pode consumir muita memória de GPU. A análise de erros divide as imagens em lotes configuráveis por `--batch-size`.
- O armazenamento do Colab é temporário. A última célula do notebook compacta a pasta `outputs/` para que pesos e gráficos possam ser baixados.
- O experimento original avaliou o modelo na validação. O pipeline atualizado preserva as 3.669 imagens do split oficial de teste para uma medição final independente.
