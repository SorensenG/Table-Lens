# Trabalho Celmar - Table Lens

Sistema em Python para carregar arquivos CSV e gerar uma visualizacao Table Lens.

## Rodar com Docker

Na pasta do projeto:

```bash
docker compose up --build
```

Depois abra:

```text
http://localhost:8501
```

Para parar, pressione `Ctrl+C` no terminal.

## Rodar sem Docker

Se houver Python instalado:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Como usar

1. Clique em `Carregar CSV`.
2. Escolha as colunas no painel lateral.
3. Defina a ordenacao e a linha central do foco.
4. Ajuste o tipo das variaveis quando a inferencia automatica nao for suficiente.

Na visualizacao:

- Barras azuis representam variaveis quantitativas normalizadas.
- Cores representam variaveis nominais ou ordinais.
- Linhas mais altas representam a regiao de foco.
- A tabela `Foco` mostra os valores exatos das linhas destacadas.

## Arquivos esperados pelo trabalho

O app aceita qualquer CSV. 

Os arquivos podem ser baixados do Kaggle e carregados diretamente pela interface.

