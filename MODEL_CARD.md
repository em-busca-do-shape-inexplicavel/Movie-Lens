# Model Card — MovieLens PyTorch Recommender

## 1. Identificação

| Campo | Valor |
|---|---|
| Nome do modelo | `movie-lens-pytorch-recommender` |
| Versão avaliada | 3 |
| Alias no MLflow | `production` |
| Tipo | Recomendador neural colaborativo |
| Framework | PyTorch |
| Dataset | MovieLens Latest Small |
| Status | Protótipo acadêmico, não aprovado para produção real |
| Versão deste documento | 1.0 |
| Data | 31 de julho de 2026 |

Este documento descreve o modelo selecionado pelo pipeline reproduzível do
Tech Challenge — Fase 02. Os números apresentados foram obtidos dos artefatos
`artifacts/metrics.json`, `artifacts/experiment_summary.json`,
`artifacts/config.json` e `artifacts/registry.json`.

## 2. Objetivo

O modelo estima a preferência de um usuário por um filme e gera uma lista
Top-10 de filmes ainda não vistos durante o treinamento. No cenário proposto
pelo desafio, usuários representam clientes, filmes representam produtos e
ratings representam feedback explícito.

O sistema foi construído para demonstrar um ciclo de MLOps reproduzível:

- preparação e versionamento de dados com DVC;
- seleção de modelo sem consultar antecipadamente o teste;
- rastreamento de experimentos e artefatos com MLflow;
- registro e promoção do modelo selecionado;
- execução local ou em contêineres Docker.

## 3. Usos pretendidos

O modelo pode ser usado para:

- estudo de sistemas de recomendação com feedback explícito;
- comparação offline entre baselines e recomendadores neurais;
- demonstração de split temporal, early stopping, DVC e MLflow;
- geração experimental de recomendações para usuários presentes no treino.

O modelo não deve ser usado para:

- decisões comerciais ou pessoais de alto impacto;
- recomendação para usuários ou itens novos sem uma estratégia de cold start;
- inferir características demográficas, personalidade ou atributos sensíveis;
- substituir testes online, análise de negócio ou validação humana;
- operar diretamente em produção sem monitoramento, segurança e revalidação.

## 4. Dados

### 4.1 Fonte e escopo

Foi utilizada a distribuição MovieLens Latest Small, disponibilizada pelo
[GroupLens](https://grouplens.org/datasets/movielens/latest/). O conjunto local
possui:

| Característica | Valor |
|---|---:|
| Avaliações | 100.836 |
| Usuários | 610 |
| Filmes no catálogo | 9.742 |
| Filmes com alguma avaliação | 9.724 |
| Tags | 3.683 |
| Escala de rating | 0,5 a 5,0 |
| Média dos ratings | 3,5016 |
| Período das interações | março de 1996 a setembro de 2018 |

Embora o dataset também forneça gêneros, títulos, tags e links, o modelo final
usa apenas `user_id`, `movie_id`, `rating` e `timestamp`. Portanto, ele é um
modelo colaborativo, não um recomendador baseado em conteúdo.

### 4.2 Preparação

O preprocessamento:

1. seleciona as quatro colunas usadas pelo modelo;
2. remove valores ausentes e linhas duplicadas;
3. valida ratings entre 0,5 e 5,0;
4. normaliza os tipos das colunas;
5. ordena as interações por usuário e instante.

### 4.3 Split temporal

Foi aplicado `temporal_leave_two_out` por usuário:

- todas as interações anteriores às duas últimas formam o treino;
- a penúltima interação de cada usuário forma a validação;
- a última interação de cada usuário forma o teste.

| Split | Interações | Finalidade |
|---|---:|---|
| Treino | 99.616 | Ajuste dos candidatos |
| Validação | 610 | Early stopping e seleção |
| Teste | 610 | Avaliação única do vencedor |

Esse desenho preserva a ordem cronológica e evita que uma interação futura seja
usada para prever uma interação passada. Após a escolha, o vencedor é retreinado
com treino e validação pelo número de épocas selecionado, mantendo o teste
isolado até a avaliação final.

## 5. Arquitetura e treinamento

O modelo contém:

- embeddings de usuário e filme com 16 dimensões;
- produto elemento a elemento entre os dois embeddings;
- concatenação dos embeddings e da interação;
- camada densa com 32 unidades e ativação ReLU;
- saída sigmoide convertida para a escala de ratings entre 0,5 e 5,0.

A função objetivo combina erro quadrático médio e perda pairwise. Ratings a
partir de 4,0 são tratados como feedback positivo; para esses exemplos, o
treinamento compara o item positivo com um item não observado amostrado.

| Hiperparâmetro | Valor selecionado |
|---|---:|
| `embedding_dim` | 16 |
| `hidden_dims` | `[32]` |
| `learning_rate` | 0,001 |
| `weight_decay` | 0,00001 |
| `ranking_weight` | 1,0 |
| `relevance_threshold` | 4,0 |
| `batch_size` | 2.048 |
| Máximo de épocas | 30 |
| Época selecionada | 3 |
| Paciência do early stopping | 3 |
| Melhoria mínima | 0,001 |
| Seed | 42 |
| Dispositivo | CPU |

## 6. Seleção do modelo

Foram rastreados três candidatos no MLflow. A única diferença entre eles foi o
peso da perda de ranking. O critério de escolha foi maximizar NDCG@10 na
validação temporal.

| Candidato | `ranking_weight` | RMSE | MAE | NDCG@10 | Recall@10 | Época |
|---|---:|---:|---:|---:|---:|---:|
| `ranking-02` | 0,2 | 0,9071 | 0,7099 | 0,0330 | 0,0605 | 4 |
| `ranking-10` | 1,0 | 0,9219 | 0,7277 | **0,0361** | **0,0663** | 3 |
| `ranking-20` | 2,0 | 0,9379 | 0,7403 | 0,0345 | 0,0605 | 3 |

O candidato `ranking-10` foi selecionado. O early stopping interrompeu sua
avaliação após seis épocas e restaurou os pesos da época 3, que apresentou o
melhor RMSE de validação desse candidato.

## 7. Desempenho final

| Métrica no teste | Valor |
|---|---:|
| RMSE | 0,994574 |
| MAE | 0,776065 |
| Precision@10 | 0,004132 |
| Recall@10 | 0,041322 |
| NDCG@10 | 0,024468 |
| Hit Rate@10 | 0,041322 |
| Catalog Coverage@10 | 0,008762 |

RMSE e MAE foram calculados sobre as 610 interações de teste. As métricas de
ranking consideram os 363 usuários cuja última interação possui rating maior
ou igual a 4,0. Como existe no máximo um item relevante por usuário no teste,
Recall@10 e Hit Rate@10 têm o mesmo valor.

O baseline aditivo do Scikit-Learn obteve RMSE 1,0176, MAE 0,8158, NDCG@10
0,0311 e Recall@10 0,0606 no mesmo teste. Assim, o modelo neural apresentou
menor erro de rating, mas o baseline teve melhor qualidade de ranking. O modelo
neural foi publicado para cumprir e avaliar a abordagem PyTorch, não porque
tenha dominado todos os baselines.

## 8. Limitações

- **Cold start de usuário:** usuários desconhecidos recebem o mesmo score médio
  para todos os itens; o desempate por ID não representa preferência real.
- **Cold start de item:** filmes ausentes do treino não pertencem ao catálogo do
  modelo e não podem ser recomendados.
- **Baixa cobertura:** as listas de teste cobrem aproximadamente 0,88% do
  catálogo treinado, indicando concentração das recomendações.
- **Poucos usuários:** 610 usuários são insuficientes para representar uma
  população ampla ou um e-commerce real.
- **Dados históricos:** as interações terminam em 2018 e podem não refletir
  preferências e catálogo atuais.
- **Feedback explícito:** ratings não representam diretamente cliques, compras,
  disponibilidade, preço ou satisfação após consumo.
- **Sem atributos de conteúdo:** gêneros e tags não ajudam usuários ou filmes
  com pouco histórico.
- **Avaliação offline:** não foram medidos CTR, conversão, diversidade percebida,
  novidade, satisfação, latência sob carga ou impacto causal.
- **Amostragem negativa simples:** um item não observado não é necessariamente
  um item rejeitado pelo usuário.

## 9. Vieses e considerações de uso responsável

O MovieLens contém avaliações fornecidas voluntariamente. Usuários que avaliam
muitos filmes e itens que recebem mais exposição tendem a aparecer mais nos
dados, gerando vieses de participação, popularidade e sobrevivência.

O dataset utilizado não fornece atributos demográficos suficientes para uma
auditoria de equidade por grupos protegidos. A ausência desses dados impede
medir disparidades; ela não demonstra que o modelo seja justo. Em uma aplicação
real, seria necessário avaliar exposição, diversidade e qualidade por grupos,
com consentimento, minimização de dados e revisão jurídica adequada.

Recomendações repetidas podem ainda criar ciclos de feedback: itens expostos
recebem novas interações e ganham vantagem sobre itens pouco exibidos. Estratégias
de exploração, diversidade e controle de popularidade devem ser testadas antes
de qualquer uso real.

## 10. Reprodutibilidade e rastreabilidade

Com o ambiente configurado e o servidor MLflow disponível:

```bash
uv sync --frozen
cp .env.example .env
docker compose up -d mlflow
dvc repro --force
dvc metrics show
```

O pipeline registra os candidatos no experimento `movie-lens-recommender` e
publica somente o vencedor como `movie-lens-pytorch-recommender`. A versão final
passa por `Staging`, chega a `Production` e recebe o alias `production`.

Principais artefatos:

| Artefato | Função |
|---|---|
| `artifacts/model.pt` | Pesos, mapas de IDs e metadados de inferência |
| `artifacts/metrics.json` | Métricas finais |
| `artifacts/config.json` | Configuração efetivamente treinada |
| `artifacts/selection_history.json` | Curvas e decisão do early stopping |
| `artifacts/experiment_summary.json` | Comparação dos candidatos e run IDs |
| `artifacts/registry.json` | Run final e versão registrada |

## 11. Monitoramento recomendado

Antes de promover uma nova versão em um cenário real, recomenda-se acompanhar:

- RMSE, MAE, NDCG@10, Recall@10 e cobertura em uma janela temporal recente;
- proporção de usuários e itens desconhecidos;
- mudança na distribuição de ratings, usuários e itens;
- diversidade e concentração de exposição;
- latência, falhas de inferência e tamanho do catálogo;
- comparação online contra um baseline seguro.

Quedas relevantes ou aumento de cold start devem bloquear a promoção automática
e acionar investigação, novo treinamento ou retorno à versão anterior.

## 12. Aprovação

A marcação `Production` no MLflow representa o estágio demonstrado no Tech
Challenge. Ela não constitui aprovação para operação comercial. O modelo deve
permanecer como protótipo até que dados, segurança, privacidade, desempenho,
equidade e impacto sejam avaliados no contexto real de uso.
