#property copyright "Criado por Codex"
#property version   "1.00"
#property strict
#property script_show_inputs

input string          InpSymbol         = "US100";                              // Simbolo a exportar
input ENUM_TIMEFRAMES InpTimeframe      = PERIOD_M5;                            // Timeframe dos candles
input int             InpBarsToExport   = 50000;                                // Quantidade maxima de barras
input bool            InpSkipCurrentBar = true;                                 // Ignorar candle ainda em formacao
input string          InpOutputFile     = "ML_MarketRegime_ONNX\\us100_m5_candles.csv"; // Caminho relativo a MQL5\\Files
input bool            InpUseCommonFiles = false;                                // Salvar em Terminal Common\\Files

//+------------------------------------------------------------------+
string ResolveSandboxRoot()
  {
   if(InpUseCommonFiles)
      return TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files";

   return TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files";
  }

//+------------------------------------------------------------------+
string ResolveOutputPath()
  {
   return ResolveSandboxRoot() + "\\" + InpOutputFile;
  }

//+------------------------------------------------------------------+
void OnStart()
  {
   string symbol = InpSymbol;
   if(symbol == "")
      symbol = _Symbol;

   if(!SymbolSelect(symbol, true))
     {
      PrintFormat("Falha ao selecionar simbolo %s. Erro=%d", symbol, GetLastError());
      return;
     }

   MqlRates rates[];
   ResetLastError();
   int copied = CopyRates(symbol, InpTimeframe, 0, InpBarsToExport, rates);
   if(copied <= 0)
     {
      PrintFormat("Falha ao copiar rates de %s em %s. Erro=%d", symbol, EnumToString(InpTimeframe), GetLastError());
      return;
     }

   int last_index = copied - 1;
   if(InpSkipCurrentBar)
      last_index--;

   if(last_index < 0)
     {
      Print("Nao ha barras fechadas suficientes para exportar.");
      return;
     }

   int flags = FILE_WRITE | FILE_CSV | FILE_ANSI;
   if(InpUseCommonFiles)
      flags |= FILE_COMMON;

   ResetLastError();
   int file_handle = FileOpen(InpOutputFile, flags, ',');
   if(file_handle == INVALID_HANDLE)
     {
      PrintFormat("Falha ao abrir arquivo %s. Erro=%d", InpOutputFile, GetLastError());
      PrintFormat("Sandbox esperado: %s", ResolveSandboxRoot());
      return;
     }

   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   if(digits < 0)
      digits = 6;

   FileWrite(file_handle, "time", "open", "high", "low", "close", "volume");

   int written = 0;
   for(int i = 0; i <= last_index; i++)
     {
      FileWrite(
         file_handle,
         TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES | TIME_SECONDS),
         DoubleToString(rates[i].open, digits),
         DoubleToString(rates[i].high, digits),
         DoubleToString(rates[i].low, digits),
         DoubleToString(rates[i].close, digits),
         (long)rates[i].tick_volume
      );
      written++;
     }

   FileClose(file_handle);

   PrintFormat("Exportacao concluida: %d candles de %s %s", written, symbol, EnumToString(InpTimeframe));
   PrintFormat("Arquivo gerado em: %s", ResolveOutputPath());
   Print("Coluna volume = tick_volume do MT5.");
  }
