from pathlib import Path


REQUIRED_COLUMNS = [
    "symbol",
    "side",
    "opened",
    "closed",
    "avgCost",
    "avgClosePrice",
    "closedVolume",
    "closingPnl",
    "maxOpenInterest",
    "status",
]

OPTIONAL_COLUMNS = [
    "id",
    "updateTime",
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

DEFAULT_INPUT_CANDIDATES = [
    "trades.json",
    "trades.csv",
    "tradeMaster.json",
    "tradeMaster.csv",
]

OUTPUT_FILENAMES = {
    "cleaned_trades": "cleaned_trades.csv",
    "summary_report": "summary_report.txt",
    "symbol_stats": "symbol_stats.csv",
    "side_stats": "side_stats.csv",
    "time_stats": "time_stats.csv",
    "profile_stats": "profile_stats.csv",
    "simulation_summary": "simulation_summary.csv",
    "edge_checks": "edge_checks.csv",
    "market_filter_stats": "market_filter_stats.csv",
    "execution_mode_split": "execution_mode_split.csv",
    "execution_fill_delay_summary": "execution_fill_delay_summary.csv",
    "execution_mode_comparison": "execution_mode_comparison.csv",
    "eth_execution_mode_trades": "eth_execution_mode_trades.csv",
    "eth_cluster_execution_joined": "eth_cluster_execution_joined.csv",
    "eth_cluster_execution_summary": "eth_cluster_execution_summary.csv",
    "eth_cluster_execution_quality": "eth_cluster_execution_quality.csv",
    "eth_cluster_execution_quality_two_bucket": "eth_cluster_execution_quality_two_bucket.csv",
    "eth_execution_reanchored_entry_quality": "eth_execution_reanchored_entry_quality.csv",
    "eth_execution_reanchored_summary": "eth_execution_reanchored_summary.csv",
    "eth_execution_reanchored_cluster_quality": "eth_execution_reanchored_cluster_quality.csv",
    "eth_execution_reanchored_distortion": "eth_execution_reanchored_distortion.csv",
    "eth_cluster1_short_real_trades": "eth_cluster1_short_real_trades.csv",
    "eth_cluster1_short_model_signals": "eth_cluster1_short_model_signals.csv",
    "eth_cluster1_short_overlap_matches": "eth_cluster1_short_overlap_matches.csv",
    "eth_cluster1_short_overlap_summary": "eth_cluster1_short_overlap_summary.csv",
    "eth_cluster1_short_overlap_quality": "eth_cluster1_short_overlap_quality.csv",
    "eth_cluster1_short_refinement_variants": "eth_cluster1_short_refinement_variants.csv",
    "eth_cluster1_short_refinement_results": "eth_cluster1_short_refinement_results.csv",
    "eth_cluster1_short_refinement_best_matches": "eth_cluster1_short_refinement_best_matches.csv",
    "eth_cluster1_short_refinement_tradeoffs": "eth_cluster1_short_refinement_tradeoffs.csv",
    "eth_cluster1_short_timing_error_matches": "eth_cluster1_short_timing_error_matches.csv",
    "eth_cluster1_short_timing_error_summary": "eth_cluster1_short_timing_error_summary.csv",
    "eth_cluster1_short_timing_error_quality": "eth_cluster1_short_timing_error_quality.csv",
    "eth_cluster1_short_final_timing_variants": "eth_cluster1_short_final_timing_variants.csv",
    "eth_cluster1_short_final_timing_results": "eth_cluster1_short_final_timing_results.csv",
    "eth_cluster1_short_final_timing_tradeoffs": "eth_cluster1_short_final_timing_tradeoffs.csv",
    "eth_cluster1_short_final_timing_best_matches": "eth_cluster1_short_final_timing_best_matches.csv",
    "core_vs_noise_summary": "core_vs_noise_summary.csv",
    "flagged_extreme_losers": "flagged_extreme_losers.csv",
    "flagged_faith_trades": "flagged_faith_trades.csv",
    "entry_quality_per_trade": "entry_quality_per_trade.csv",
    "excursion_metrics": "excursion_metrics.csv",
    "exit_efficiency": "exit_efficiency.csv",
    "entry_quality_summary": "entry_quality_summary.csv",
    "symbol_entry_quality": "symbol_entry_quality.csv",
    "side_entry_quality": "side_entry_quality.csv",
    "market_context_at_entry": "market_context_at_entry.csv",
    "strong_vs_weak_entry_comparison": "strong_vs_weak_entry_comparison.csv",
    "eth_entry_context_comparison": "eth_entry_context_comparison.csv",
    "eth_long_context_comparison": "eth_long_context_comparison.csv",
    "eth_short_context_comparison": "eth_short_context_comparison.csv",
    "eth_context_summary": "eth_context_summary.txt",
    "eth_bollinger_compatibility_summary": "eth_bollinger_compatibility_summary.csv",
    "eth_bollinger_short_hypotheses": "eth_bollinger_short_hypotheses.csv",
    "eth_bollinger_long_hypotheses": "eth_bollinger_long_hypotheses.csv",
    "eth_setup_clusters": "eth_setup_clusters.csv",
    "eth_cluster_summary": "eth_cluster_summary.csv",
    "eth_best_cluster_profile": "eth_best_cluster_profile.csv",
    "eth_cluster_descriptions": "eth_cluster_descriptions.txt",
    "eth_cluster_feature_ranges": "eth_cluster_feature_ranges.csv",
    "eth_cluster_rule_extraction": "eth_cluster_rule_extraction.txt",
    "eth_cluster_rule_quality": "eth_cluster_rule_quality.csv",
    "eth_replication_signals_cluster1": "eth_replication_signals_cluster1.csv",
    "eth_replication_signals_cluster2": "eth_replication_signals_cluster2.csv",
    "eth_replication_summary": "eth_replication_summary.csv",
    "eth_replication_stage_comparison": "eth_replication_stage_comparison.csv",
    "eth_timing_feature_ranking": "eth_timing_feature_ranking.csv",
    "eth_cluster1_tightened_candidates": "eth_cluster1_tightened_candidates.csv",
    "eth_cluster2_tightened_candidates": "eth_cluster2_tightened_candidates.csv",
    "eth_tightened_filter_summary": "eth_tightened_filter_summary.csv",
    "eth_tightened_filter_stage_comparison": "eth_tightened_filter_stage_comparison.csv",
    "eth_cluster1_micro_trigger_ranking": "eth_cluster1_micro_trigger_ranking.csv",
    "eth_cluster1_micro_trigger_candidates": "eth_cluster1_micro_trigger_candidates.csv",
    "eth_cluster1_micro_trigger_quality": "eth_cluster1_micro_trigger_quality.csv",
    "eth_cluster1_structural_confirmation_ranking": "eth_cluster1_structural_confirmation_ranking.csv",
    "eth_cluster1_final_trigger_candidates": "eth_cluster1_final_trigger_candidates.csv",
    "eth_cluster1_final_trigger_quality": "eth_cluster1_final_trigger_quality.csv",
    "eth_cluster1_effort_result_hypothesis": "eth_cluster1_effort_result_hypothesis.csv",
    "eth_cluster1_effort_result_ranking": "eth_cluster1_effort_result_ranking.csv",
    "eth_cluster1_effort_result_quality": "eth_cluster1_effort_result_quality.csv",
    "eth_proxy_short_results": "eth_proxy_short_results.csv",
    "eth_proxy_long_results": "eth_proxy_long_results.csv",
    "eth_proxy_stage_comparison": "eth_proxy_stage_comparison.csv",
    "eth_proxy_exit_comparison": "eth_proxy_exit_comparison.csv",
    "eth_short_proxy_validation_summary": "eth_short_proxy_validation_summary.csv",
    "eth_short_proxy_friction_comparison": "eth_short_proxy_friction_comparison.csv",
    "eth_short_proxy_parameter_robustness": "eth_short_proxy_parameter_robustness.csv",
    "eth_short_proxy_stage_consistency": "eth_short_proxy_stage_consistency.csv",
    "eth_short_proxy_monthly_breakdown": "eth_short_proxy_monthly_breakdown.csv",
}

PLOTS_DIRNAME = "plots"
KLINES_CACHE_DIRNAME = ".cache/binance_klines"
SIMULATION_RUNS = 1000
SIMULATION_SEED = 42
EMA_INTERVAL = "1h"
EMA_PERIOD = 50
BINANCE_MAX_RETRIES = 5
BINANCE_BACKOFF_SECONDS = 0.5
ENTRY_QUALITY_INTERVALS = ["1m", "5m", "15m", "1h"]
ENTRY_REACTION_HORIZONS_MINUTES = [1, 5, 15, 30, 60, 120, 240, 480, 720, 1440]

HOLDING_BUCKETS = [
    ("<5 min", 0.0, 5.0 / 60.0),
    ("5-15 min", 5.0 / 60.0, 15.0 / 60.0),
    ("15-60 min", 15.0 / 60.0, 1.0),
    ("1-4 h", 1.0, 4.0),
    ("4-12 h", 4.0, 12.0),
    ("12-24 h", 12.0, 24.0),
    ("24h+", 24.0, float("inf")),
]

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

UTC_LABEL = "UTC"


def resolve_output_path(base_dir: Path, key: str) -> Path:
    return base_dir / OUTPUT_FILENAMES[key]
