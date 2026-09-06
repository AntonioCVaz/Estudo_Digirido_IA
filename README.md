# Estudo Dirigido – Inteligência Artificial 2026.1
### Classificador Bayesiano Ingênuo (Naive Bayes) aplicado ao Censo da Educação Superior 2020

**Disciplina:** Estudo Dirigido – Inteligência Artificial 2026.1 — Prof. Dr. Luis Filipe
**Curso:** Bacharelado em Ciência da Computação — UFAPE
**Dupla:** Antônio Carlos da Silva Batista Vaz e Paulo Eduardo Vieira Souza

---

## 1. Objetivo

Investigar experimentalmente o funcionamento de classificadores Bayesianos a partir da modelagem probabilística de características de uma base de dados real, cobrindo: distribuições de probabilidade por classe, verossimilhança, razão de verossimilhanças, Teorema de Bayes, fronteiras de decisão e um classificador Naive Bayes multivariado avaliado por matriz de confusão.

## 2. Base de dados

**Fonte:** [Microdados do Censo da Educação Superior 2020 (Kaggle)](https://www.kaggle.com/datasets/mistergomes/microdados-educacao-superior-2020)

O objetivo é prever a **categoria administrativa** (rede) de cursos de graduação no Brasil (Pública ou Privada).

| Item | Descrição |
|---|---|
| Variável alvo (Y) | `TP_REDE` → mapeada para `Y_REDE` (0 = Pública, 1 = Privada) |
| X1 | `QT_ING` — Quantidade de ingressantes (**contínua**) |
| X2 | `QT_MAT` — Quantidade de matrículas (**contínua**) |
| X3 | `TP_MODALIDADE_ENSINO` → `X3_MODALIDADE` — Modalidade de ensino (**categórica binária**: 0 = Presencial, 1 = EaD) |
| Nº de observações | 335.629 linhas após limpeza (`dropna`), de um arquivo original de 335.630 linhas com cabeçalho |
| Distribuição das classes | ≈ 95,2% Privada / 4,8% Pública (fortemente desbalanceada) |

## 3. O que o código faz

1. **Download e verificação de integridade:** baixa o dataset via API do Kaggle (`kagglehub`), em vez de upload manual no Colab, pois o upload manual truncava o arquivo silenciosamente. Uma checagem de linhas (`wc -l`) garante que o arquivo está íntegro antes de prosseguir.
2. **ETL:** seleciona as colunas de interesse, remove nulos e binariza `Y_REDE` e `X3_MODALIDADE`.
3. **Divisão treino/teste:** split estratificado 70/30 (`random_state=42`), preservando o desbalanceamento das classes em ambos os conjuntos; os CSVs (`treino.csv` e `teste.csv`) são exportados.
4. **Hipóteses de distribuição:** X1 e X2 são modeladas como **Exponencial** (não Normal — decisão validada por comparação AIC / teste de Kolmogorov-Smirnov nos dados de treino) e X3 como **Bernoulli**.
5. **Análise univariada de X1, X2 e X3:** cálculo de verossimilhança, razão de verossimilhanças Λ(x), probabilidade a posteriori via Teorema de Bayes, e fronteira de decisão (solução linear, já que ambas as classes usam Exponencial). Para X3, apresenta a regra de decisão por categoria e um ranking de poder discriminativo via Coeficiente de Sobreposição (OVL).
6. **Classificador Naive Bayes multivariado:** combina X1, X2 (Exponencial) e X3 (Bernoulli, com suavização de Laplace/epsilon para evitar `log(0)`), assumindo independência condicional entre as características dado Y.
7. **Avaliação:** matriz de confusão e métricas (acurácia, precisão, recall, F1) calculadas sobre o conjunto de teste.
8. **Discussão crítica:** limitações da hipótese de independência condicional (X1 e X2 são correlacionadas na prática) e do desbalanceamento de classes (viés do modelo em favor da classe majoritária "Privada").

## 4. Estrutura do repositório

```
.
├── estudo_dirigido_antônio_paulo.py   # Script principal (gerado a partir do notebook Colab)
├── README.md
├── requirements.txt
├── treino.csv                          # gerado em tempo de execução
└── teste.csv                           # gerado em tempo de execução
```

## 5. Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/AntonioCVaz/Estudo_Digirido_IA
   cd Estudo_Digirido_IA
   ```
2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure as credenciais da API do Kaggle** (necessário para o `kagglehub` baixar o dataset):
   - Gere um token em [kaggle.com/settings](https://www.kaggle.com/settings) → *Create New Token* (baixa um arquivo `kaggle.json`).
   - Coloque o arquivo em `~/.kaggle/kaggle.json` (Linux/Mac) ou `C:\Users\<usuário>\.kaggle\kaggle.json` (Windows), com permissão `600`:
     ```bash
     mkdir -p ~/.kaggle
     mv kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

## 6. Execução

```bash
python estudo_dirigido_antônio_paulo.py
```

O script irá:
- baixar e validar o dataset automaticamente;
- gerar `treino.csv` e `teste.csv`;
- imprimir no console os parâmetros estimados (priors, lambdas, probabilidades), os exemplos de verossimilhança/razão de verossimilhanças/posteriori e as fronteiras de decisão;
- exibir os gráficos de histograma x densidade ajustada (X1 e X2) e a matriz de confusão;
- imprimir o relatório final de métricas (acurácia, precisão, recall, F1).

> Observação: o script foi extraído de um notebook Google Colab; ao rodar localmente, os `plt.show()` abrirão janelas de gráfico (ou, em ambiente sem display, salve as figuras com `plt.savefig(...)` conforme necessário).

## 7. Principais resultados (conjunto de teste, 100.689 observações)

| Métrica | Pública (0) | Privada (1) |
|---|---|---|
| Precisão | 0.29 | 0.97 |
| Recall | 0.50 | 0.94 |
| F1-score | 0.37 | 0.96 |

Acurácia geral: **0.92**

Ranking de poder discriminativo isolado (1 − OVL): 1º X2 (Matrículas), 2º X3 (Modalidade), 3º X1 (Ingressantes).

## 8. Limitações da modelagem

- **Independência condicional violada:** número de ingressantes e de matrículas são fortemente correlacionados na prática, então o modelo conta parcialmente a mesma evidência duas vezes.
- **Desbalanceamento de classes (95% Privada):** o prior dominante faz o classificador exigir evidência numérica extrema para prever "Pública", gerando muitos falsos negativos para essa classe (precisão de apenas 0.29).

## 9. Entrega

- Vídeo de apresentação (até 15 min): (https://youtu.be/-48QXFt48k0)
- Base de dados: [Kaggle — Microdados Censo da Educação Superior 2020](https://www.kaggle.com/datasets/mistergomes/microdados-educacao-superior-2020)
