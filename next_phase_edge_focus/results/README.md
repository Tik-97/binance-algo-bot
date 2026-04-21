Next phase results storage

Active baseline:

- `Tier 2.87 realistic`

Current active result folders:

- `next_phase_edge_focus/results/binance_tier287_realistic_backtest_v1`
- `next_phase_edge_focus/results/binance_tier287_eth_stress_test_v1`

What each folder means:

- `binance_tier287_realistic_backtest_v1`
  - main realistic Binance benchmark
  - the current source of truth for baseline performance

- `binance_tier287_eth_stress_test_v1`
  - ETH/Binance adverse execution package
  - same logic, worse fills
  - used to sanity-check how much live friction can hurt us

Current baseline summary:

- `42` trades
- `ending_balance = 645.87`
- `win_rate = 88.10%`
- `profit_factor = 7.872`

Current stress summary:

- `ETH adverse light`: `620.00`
- `ETH adverse medium`: `562.38`
- `ETH adverse hard`: `498.29`

Only save new results here if:

- they are a real successor candidate
- or they are operational validation for the active baseline

Do not refill this folder with broad historical experiments again.
