# Artigo (formato SBC)

Este artigo segue o template padrão da Sociedade Brasileira de Computação (SBC), usado em
eventos como SBC, WEIT, ERI, etc.

## Como compilar

Este diretório contém apenas `artigo.tex` e `referencias.bib` — faltam os arquivos de estilo
oficiais da SBC (`sbc-template.cls` e `sbc.bst`), que **não podem ser redistribuídos aqui** e
devem ser obtidos direto na fonte:

### Opção A — Overleaf (mais fácil)

1. Crie um projeto novo no [Overleaf](https://www.overleaf.com).
2. Em "New Project", escolha "Template Gallery" e busque por **"SBC"** — o template oficial
   já vem com `sbc-template.cls` e `sbc.bst` inclusos.
3. Substitua o `main.tex` do template pelo conteúdo de [`artigo.tex`](artigo.tex) e suba
   também o [`referencias.bib`](referencias.bib) (renomeando o `.bib` do template ou
   ajustando o `\bibliography{referencias}` para o nome usado).

### Opção B — Local (TeX Live / MiKTeX)

1. Baixe o template oficial em
   <https://www.sbc.org.br/documentos-4/templates-para-artigos-e-capitulos-de-livros/> (arquivo
   `sbc-template.zip`, contém `sbc-template.cls` e `sbc.bst`).
2. Copie `sbc-template.cls` e `sbc.bst` para esta pasta (`artigo/`).
3. Compile com:
   ```bash
   pdflatex artigo.tex
   bibtex artigo
   pdflatex artigo.tex
   pdflatex artigo.tex
   ```

## O que falta preencher

O texto já está completo, mas contém trechos marcados com `[preencher]` que só podem ser
respondidos depois de rodar o treinamento (veja o [README principal](../README.md)):

- Número de épocas efetivamente usado (Seção "Arquitetura e treinamento").
- Tabela de resultados (mAP50, mAP50-95, precisão, recall).
- Discussão qualitativa dos acertos/erros e das dificuldades do desenvolvimento.
- Figura de exemplo: depois de rodar `python -m src.predict`, copie uma imagem de
  `outputs/samples/` para uma pasta `artigo/figuras/` e descomente o bloco `\begin{figure}`
  na Seção de Resultados.
- Conclusão final com a síntese dos achados.
