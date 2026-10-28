# Graph Report - shadow-bot  (2026-10-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 145 nodes · 248 edges · 9 communities (6 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3fc45c65`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Nautilus Backtest Live
- Shadow Strategy Engine
- FastAPI Dashboard Backend
- Docs Safety And UI
- Research Data Sweep
- Bot Control Channel
- Shadow Timemachine
- PySR
- VectorBT

## God Nodes (most connected - your core abstractions)
1. `ShadowStrategy` - 24 edges
2. `Dashboard Control Panel` - 9 edges
3. `ShadowConfig` - 7 edges
4. `main()` - 7 edges
5. `main()` - 6 edges
6. `_spawn()` - 6 edges
7. `main()` - 6 edges
8. `_atomic_write()` - 6 edges
9. `Shadow Bot` - 6 edges
10. `synthetic()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `fastapi package` --conceptually_related_to--> `Dashboard Control Panel`  [INFERRED]
  requirements.txt → dashboard/index.html
- `uvicorn package` --conceptually_related_to--> `Dashboard Control Panel`  [INFERRED]
  requirements.txt → dashboard/index.html
- `main()` --calls--> `ShadowConfig`  [EXTRACTED]
  backtest.py → shadow_bot/strategy.py
- `main()` --calls--> `ShadowStrategy`  [EXTRACTED]
  backtest.py → shadow_bot/strategy.py
- `main()` --calls--> `ShadowStrategy`  [EXTRACTED]
  run_live.py → shadow_bot/strategy.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Signal Generation Ensemble** — shadow_timemachine_integration_plan_and_prompt_en_kronos, shadow_timemachine_integration_plan_and_prompt_en_arch, shadow_timemachine_integration_plan_and_prompt_en_tick, shadow_timemachine_integration_plan_and_prompt_en_pysr [EXTRACTED 1.00]
- **Fetch Sweep Backtest Dashboard Loop** — readme_research_workflow, dashboard_index_research_fetch, dashboard_index_research_sweep, readme_vectorbt [INFERRED 0.75]
- **Probabilistic Forecasting Pipeline** — shadow_timemachine_integration_plan_and_prompt_en_mapie, shadow_timemachine_integration_plan_and_prompt_en_laya, shadow_bot_strategy [INFERRED 0.80]
- **Pause Resume Flatten Kill Switch** — dashboard_index_pause, dashboard_index_resume, dashboard_index_flatten, dashboard_index_kill_switch, readme_safety_controls [INFERRED 0.85]

## Communities (9 total, 3 thin omitted)

### Community 0 - "Nautilus Backtest Live"
Cohesion: 0.09
Nodes (4): main(), ShadowConfig, day-trader-skill, Nautilus Trader

### Community 2 - "FastAPI Dashboard Backend"
Cohesion: 0.15
Nodes (14): _alive(), bot_start(), bot_stop(), BotParams, command(), Command, FetchParams, index() (+6 more)

### Community 3 - "Docs Safety And UI"
Cohesion: 0.09
Nodes (19): POST /api/command, GET /api/state, POST /api/bot/start, POST /api/bot/stop, Dashboard Control Panel, Emergency Flatten, Pause Command, POST /api/research/fetch (+11 more)

### Community 4 - "Research Data Sweep"
Cohesion: 0.18
Nodes (6): main(), synthetic(), fetch(), load(), main(), run_one()

### Community 5 - "Bot Control Channel"
Cohesion: 0.28
Nodes (6): state(), _atomic_write(), _read(), read_state(), send_command(), write_state()

### Community 6 - "Shadow Timemachine"
Cohesion: 0.33
Nodes (6): arch, Kronos, Laya, MAPIE, Shadow Timemachine, tick

## Knowledge Gaps
- **19 isolated node(s):** `POST /api/bot/stop`, `Pause Command`, `POST /api/research/fetch`, `Resume Command`, `Live Strategy Parameters` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 54 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ShadowStrategy` connect `Shadow Strategy Engine` to `Nautilus Backtest Live`, `Research Data Sweep`?**
  _High betweenness centrality (0.218) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Dashboard Control Panel` (e.g. with `Shadow Bot` and `fastapi package`) actually correct?**
  _`Dashboard Control Panel` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `POST /api/bot/stop`, `Pause Command`, `POST /api/research/fetch` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Nautilus Backtest Live` be split into smaller, more focused modules?**
  _Cohesion score 0.09195402298850575 - nodes in this community are weakly interconnected._
- **Why does `main()` connect `Nautilus Backtest Live` to `Shadow Strategy Engine`, `Research Data Sweep`, `Bot Control Channel`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Should `Shadow Strategy Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.14153846153846153 - nodes in this community are weakly interconnected._
- **Why does `main()` connect `Research Data Sweep` to `Nautilus Backtest Live`, `Shadow Strategy Engine`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._