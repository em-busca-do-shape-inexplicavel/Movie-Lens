# MovieLens Recommendation System

Sistema de recomendação desenvolvido para o Tech Challenge — Fase 02 da FIAP.
O projeto usa o MovieLens Small como analogia de um e-commerce:

| MovieLens | E-commerce |
|---|---|
| `userId` | Cliente |
| `movieId` | Produto |
| `rating` | Preferência/interação |
| `genres` e `tags` | Atributos do produto |

O dataset possui mais de 100 mil interações usuário-item e atende ao mínimo de
10 mil interações estabelecido pelo desafio.

## Estado atual

- Validação e EDA orientada a recomendação.
- Split temporal por usuário sem vazamento.
- Baseline de popularidade.
- Fatoração de matriz com NumPy.
- Precision@K, Recall@K, Hit Rate@K, Coverage@K e RMSE.
- Factory para modelos e Strategy para preprocessamento.
- Testes, Ruff, pre-commit e dependências gerenciadas com uv.

O baseline Scikit-Learn, o modelo PyTorch e a infraestrutura DVC/MLflow/Docker
serão adicionados nas próximas etapas.

## Estrutura

```text
configs/       configurações carregadas do ambiente
data/raw/      arquivos originais do MovieLens
notebooks/     validação, EDA e experimentos
scripts/       pontos de entrada executáveis
src/data/      loaders, splits e preprocessadores
src/models/    recomendadores e Factory
src/evaluation métricas de regressão e ranking
src/training/  utilidades de treinamento e seeds
tests/         testes automatizados
```

## Preparação do ambiente

Instale o [uv](https://docs.astral.sh/uv/) e execute:

```bash
uv sync
cp .env.example .env
```

Ative o ambiente criado pelo uv:

```bash
source .venv/bin/activate
```

Valide dependências e dados:

```bash
python scripts/validate_env.py
```

## Qualidade e testes

```bash
ruff check src tests scripts configs
ruff format --check src tests scripts configs
pytest
```

Para instalar os hooks locais:

```bash
pre-commit install
pre-commit run --all-files
```

## Notebooks

Execute o Jupyter dentro do ambiente:

```bash
jupyter notebook
```

Ordem recomendada:

1. `01_data_validation.ipynb`
2. `02_eda_recommendation.ipynb`
3. `03_baseline.ipynb`
4. `04_matrix_factorization.ipynb`

## Convenção de branches e commits

Branches têm um objetivo único:

```text
feat/<funcionalidade>
fix/<problema>
docs/<assunto>
chore/<manutencao>
```

Commits seguem Conventional Commits:

```text
feat(models): add matrix factorization baseline
fix(data): prevent temporal leakage in split
test(metrics): cover recall at k
docs(readme): document local setup
chore(deps): update uv lock file
```

Antes de encerrar uma branch, execute testes e lint. A branch só deve ser
integrada após esses comandos passarem.
