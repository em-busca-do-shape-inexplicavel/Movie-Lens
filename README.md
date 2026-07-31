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
- Baselines Scikit-Learn de média global e vieses aditivos.
- Fatoração de matriz com NumPy.
- Recomendador neural híbrido com embeddings PyTorch.
- Pipeline executável com early stopping, checkpoints reutilizáveis e tracking com MLflow.
- Precision@K, Recall@K, NDCG@K, Hit Rate@K, Coverage@K, RMSE e MAE.
- Factory para modelos e Strategy para preprocessamento.
- Testes, Ruff, pre-commit e dependências gerenciadas com uv.

MLflow é usado para tracking e registry local. DVC e Docker continuam fora do escopo.

## Resultado do baseline Scikit-Learn

O notebook `05_sklearn_baselines.ipynb` compara média global e vieses aditivos
com três níveis de regularização. A configuração foi escolhida somente na
validação temporal; depois, o modelo foi retreinado com treino e validação e
avaliado uma única vez no teste.

| Etapa | Modelo | RMSE | MAE | NDCG@10 | Recall@10 |
|---|---|---:|---:|---:|---:|
| Validação | Vieses, `alpha=0.0001` | 0.9489 | 0.7595 | 0.0363 | 0.0720 |
| Teste | Vieses, `alpha=0.0001` | 1.0176 | 0.8158 | 0.0311 | 0.0606 |

Esse baseline aprende a tendência de nota de cada usuário e filme, mas não uma
afinidade específica entre ambos. Ele funciona como referência intermediária
entre popularidade/média global e modelos personalizados, como fatoração de
matriz e embeddings.

## Resultado do recomendador PyTorch

O notebook `06_pytorch_recommender.ipynb` combina MSE de ratings com uma perda
pairwise baseada em feedback positivo e amostragem negativa. Na validação,
foram comparados pesos de ranking de 0,2, 1,0 e 2,0. O peso 2,0 venceu entre as
configurações neurais pelo NDCG@10.

| Etapa | Modelo | RMSE | MAE | NDCG@10 | Recall@10 |
|---|---|---:|---:|---:|---:|
| Validação | PyTorch, `ranking_weight=2.0` | 0.9784 | 0.7533 | 0.0331 | 0.0692 |
| Teste | PyTorch, `ranking_weight=2.0` | 1.0505 | 0.8051 | 0.0274 | 0.0468 |

Na mesma validação, o Scikit-Learn alcançou NDCG@10 de 0,0363. Portanto, a rede
neural ainda não vence o baseline aditivo em ranking, embora aprenda afinidades
usuário-filme e produza listas personalizadas. O resultado é mantido sem
maquiagem: modelos mais complexos precisam justificar seu custo com métricas.

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
src/tracking/  tracking e registro no MLflow
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

## Pipeline de treinamento

Os hiperparâmetros ficam em `params.yaml`. Para executar o fluxo completo:

```bash
python scripts/train.py --params params.yaml --output-dir artifacts
```

O comando realiza estas etapas:

1. carrega os ratings e aplica o split temporal;
2. treina com validação e seleciona a melhor época por early stopping;
3. retreina do zero com treino e validação pelo número de épocas escolhido;
4. avalia uma única vez no teste;
5. salva modelo, métricas, configuração e histórico de seleção;
6. registra a run e promove a versão no Model Registry.

Artefatos produzidos:

```text
artifacts/model.pt               rede e mapas de IDs para inferência
artifacts/metrics.json           métricas finais e tamanhos dos splits
artifacts/config.json            configuração efetivamente treinada
artifacts/selection_history.json curvas e decisão do early stopping
```

O `params.yaml` utilizado também é armazenado entre os artefatos da run no
MLflow, sem criar uma cópia adicional dentro de `artifacts/`.

Com os parâmetros atuais, o early stopping executou 6 épocas, selecionou a
época 3 e obteve RMSE de teste 1.0145 e NDCG@10 de 0.0242. `artifacts/` não é
versionado diretamente pelo Git; ele fica ao lado dos artefatos do MLflow.

## MLflow local

Inicie o servidor com:

```bash
mlflow server \
	--backend-store-uri sqlite:///mlflow.db \
	--default-artifact-root ./mlartifacts \
	--host 0.0.0.0 \
	--port 5000
```

Depois execute o treino normal. O pipeline registra parâmetros, métricas,
artefatos e a versão promovida do modelo `movie-lens-pytorch-recommender`.

Para carregar o modelo publicado:

```python
import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model(
	"models:/movie-lens-pytorch-recommender@production"
)
predictions = model.predict(
	pd.DataFrame({"user_id": [1], "movie_id": [1]})
)
```

O stage `Production` também é mantido para atender ao desafio, embora o MLflow
recomende aliases nas versões atuais. Para registrar novamente uma run já
existente no mesmo servidor configurado no `.env`:

```bash
python scripts/register_model.py --run-id RUN_ID
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
5. `05_sklearn_baselines.ipynb`
6. `06_pytorch_recommender.ipynb`

O script `scripts/register_model.py` registra uma run existente no Registry se
você quiser repetir a promoção fora do fluxo principal de treino.

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
