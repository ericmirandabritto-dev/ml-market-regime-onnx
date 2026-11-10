# Market Regime ONNX

Pipeline de pesquisa quantitativa que classifica candles OHLCV em regimes de baixa, lateralização e alta. O projeto cobre ingestão, engenharia de atributos, avaliação cronológica e exportação ONNX com contrato de inferência para MetaTrader 5.

**Tecnologias.** Python 3.12, pandas, scikit-learn, ONNX Runtime, PowerShell e MQL5.

## Fluxo de dados

1. **Ingestão.** Leia um CSV de candles fechados ou extraia o histórico do MetaTrader 5.
2. **Features e rótulos.** Calcule retornos defasados, amplitude e pavios, ATR normalizado, volatilidade, volume relativo e diferenças de médias exponenciais. O retorno futuro normalizado pelo ATR define `0` (baixa), `1` (lateralização) ou `2` (alta).
3. **Avaliação.** Use um holdout cronológico para a primeira medição ou janelas walk-forward para observar a estabilidade ao longo do tempo. O holdout exclui do treino rótulos cujo horizonte alcança o teste. No walk-forward, o purge mínimo acompanha o horizonte do rótulo.
4. **Exportação.** Grave o modelo, a ordem das features, os parâmetros de entrada e as métricas. Com `--validate-onnx`, o erro absoluto máximo entre as probabilidades Python e ONNX deve ficar em até `1e-4`.

## Primeira execução

Use Windows, PowerShell 7 e Python 3.12 disponível como `python`. Para indicar outro executável, use `./setup_venv.ps1 -PythonExe "caminho/do/python.exe"`. O instalador do clone público fixa as versões das dependências diretas do fluxo principal.

```powershell
git clone https://github.com/ericmirandabritto-dev/ml-market-regime-onnx.git
cd ml-market-regime-onnx
./setup_venv.ps1
$py = "./.venv/Scripts/python.exe"
& $py -B -m unittest discover -s tests -v
```

Os testes geram candles sintéticos e exercitam a CLI, a separação temporal, o cache incremental, a preservação do modelo anterior em caso de falha e a inferência ONNX. Eles rodam sem terminal MT5 e sem dados de corretora.

### Treinar com candles próprios

Prepare um CSV separado por vírgulas, com uma linha por candle fechado e timestamps na mesma referência horária em todo o arquivo. Use a coluna `volume` para `tick_volume` nas exportações do MT5.

```csv
time,open,high,low,close,volume
2025-01-02 09:00:00,21000.0,21014.0,20995.0,21008.0,320
```

O trecho ilustra o esquema. O treino precisa de histórico suficiente para calcular as features e observar as três classes no conjunto de treino.

```powershell
& $py src/train_market_regime_onnx.py --input data/candles.csv --artifacts artifacts/holdout --validate-onnx
```

A configuração padrão usa regressão logística, horizonte de 8 barras, limiar de 1,5 ATR, 20% das amostras para teste e seed 42. Para personalizar o YAML, gere uma cópia dos padrões e passe `--config market-regime.yaml` ao comando de treino.

```powershell
& $py -c "import sys, yaml; sys.path.insert(0, 'src'); from market_regime_core import load_config; print(yaml.safe_dump(load_config(), sort_keys=False))" | Set-Content -Encoding utf8 market-regime.yaml
```

### Avaliar em janelas walk-forward

Os tamanhos abaixo são contados em barras. Use um CSV que cubra várias janelas completas e ajuste os tamanhos de acordo com seu histórico.

```powershell
& $py src/train_market_regime_onnx.py --input data/candles.csv --artifacts artifacts/walkforward --mode walkforward --train-window 5000 --trade-window 1000 --step-window 1000 --min-folds 3 --min-class-count-per-train 30 --symbol US100 --timeframe M5 --validate-onnx
```

Compare a balanced accuracy por janela, sua dispersão, a cobertura e a distribuição das classes. O score padrão penaliza a dispersão, a baixa cobertura e a falta de folds válidos. Ao usar o [sweep](src/sweep_market_regime.py) para escolher hiperparâmetros, reserve um período posterior separado para a avaliação final.

### Extrair candles do MT5

Em Windows, com o MetaTrader 5 aberto e autenticado, instale a integração Python e informe o símbolo como aparece na sua corretora. O extrator pula o candle em formação por padrão. Também é possível compilar e executar o [exportador MQL5](mt5/ExportUS100Candles.mq5) no MetaEditor.

```powershell
& $py -m pip install MetaTrader5
& $py src/extract_mt5_candles.py --symbol US100 --timeframe M5 --bars 50000 --output data/candles.csv
```

## Código e artefatos

| Componente | Responsabilidade |
| --- | --- |
| [Treino](src/train_market_regime_onnx.py) e [núcleo do modelo](src/market_regime_core.py) | Features, classificação, holdout e exportação ONNX |
| [Walk-forward](src/walkforward.py) e [sweep](src/sweep_market_regime.py) | Janelas cronológicas, comparação de configurações e manifestos |
| [Sincronização SQLite](src/run_research_db_sync.py) | Catálogo local de relatórios e pacotes de pesquisa |
| [Gestão de posições](src/build_position_management_datasets.py) | Construção de datasets a partir de candles e relatórios XLSX do Strategy Tester |
| [Testes](tests/test_portfolio.py) | Regressões temporais, cache, publicação local do modelo e treino real com dados sintéticos |

O holdout gera `model.onnx`, `model_metadata.json`, `runtime_contract.csv`, `test_predictions.csv`, `confusion_matrix.csv` e `classification_report.json`. O walk-forward adiciona modelos por fold, `summary.csv`, `oos_predictions.csv` e manifestos de pesquisa e runtime. Para ingestão local, execute `& $py src/run_research_db_sync.py --db db/research.sqlite`.

## Escopo do portfólio

O fluxo verificável neste clone é o treino offline, a avaliação temporal e a exportação do modelo. Os dados de mercado, relatórios, configurações operacionais e modelos gerados ficam no ambiente do usuário.

Os scripts de gestão de posições exigem YAML próprio, relatório XLSX e `openpyxl`, com `pyarrow` para Parquet. Os modelos neurais exigem PyTorch. A automação do Strategy Tester depende do EA, dos templates e do projeto irmão `MT5_WalkForward_Window_Optimizer`. O contrato CSV descreve a interface para integração. A verificação de paridade roda no ONNX Runtime em CPU e a integração no terminal requer validação própria. As métricas são de classificação de regime. Resultados financeiros exigem backtest separado com custos e regras de execução.
