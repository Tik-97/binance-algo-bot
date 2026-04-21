import argparse
import sys
from pathlib import Path

from analysis import (
    analyze_concentration,
    analyze_holding_efficiency,
    analyze_position_behavior,
    basic_stats,
    build_side_stats,
    build_symbol_stats,
    build_time_stats,
    classify_trader_style,
    console_summary,
    create_extended_plots,
    create_plots,
    generate_summary_report,
    normalize_trades,
)
from config import PLOTS_DIRNAME, SIMULATION_RUNS, SIMULATION_SEED, resolve_output_path
from core_analysis import create_core_vs_noise_plots, run_core_vs_noise_analysis
from execution_mode_split import create_execution_mode_split_plots, run_execution_mode_split_analysis
from ethShortTrader import (
    create_eth_execution_aware_archetype_plots,
    run_eth_execution_aware_archetype_analysis,
)
from eth_execution_reanchored_entry_quality import (
    create_eth_execution_reanchored_plots,
    run_eth_execution_reanchored_entry_quality_analysis,
)
from eth_cluster1_short_overlap_validation import (
    create_reanchored_cluster1_short_validation_plots,
    run_reanchored_cluster1_short_validation_analysis,
)
from eth_cluster1_short_precision_refinement import (
    create_cluster1_short_precision_refinement_plots,
    run_cluster1_short_precision_refinement_analysis,
)
from eth_cluster1_short_timing_error_anatomy import (
    create_cluster1_short_timing_error_plots,
    run_cluster1_short_timing_error_anatomy_analysis,
)
from eth_cluster1_short_final_timing_test import (
    create_cluster1_short_final_timing_test_plots,
    run_cluster1_short_final_timing_test_analysis,
)
from eth_cluster1_structural_confirmation import (
    create_eth_cluster1_structural_confirmation_plots,
    run_eth_cluster1_structural_confirmation_analysis,
)
from eth_bollinger_compatibility import (
    create_eth_bollinger_compatibility_plots,
    run_eth_bollinger_compatibility_test,
)
from eth_cluster1_effort_result import (
    create_eth_cluster1_effort_result_plots,
    run_eth_cluster1_effort_result_analysis,
)
from eth_practical_imitation import create_eth_proxy_plots, run_eth_practical_imitation_model
from eth_short_proxy_validation import (
    create_eth_short_proxy_validation_plots,
    run_eth_short_proxy_validation,
)
from entry_quality import create_entry_quality_plots, run_entry_quality_analysis
from eth_cluster1_micro_trigger import create_eth_cluster1_micro_trigger_plots, run_eth_cluster1_micro_trigger_analysis
from eth_cluster_rule_extraction import create_eth_cluster_rule_plots, run_eth_cluster_rule_extraction
from eth_context_analysis import create_eth_context_plots, run_eth_context_analysis
from eth_filter_tightening import create_eth_tightening_plots, run_eth_filter_tightening
from eth_replication_hypothesis_test import create_eth_replication_plots, run_eth_replication_hypothesis_test
from eth_setup_clustering import create_eth_setup_cluster_plots, run_eth_setup_clustering
from market_context_at_entry import create_market_context_plot, run_market_context_analysis
from market import build_market_filter
from simulation import build_profile_stats, evaluate_edge_checks, simulate_monte_carlo
from utils import autodetect_input_file, load_trades, parse_mixed_datetime_to_utc, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trade history reconstruction / behavioral trading analysis for executed Binance copy-trading trades."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to trades.json or trades.csv. If omitted, the script auto-detects a known filename.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        help="Directory where cleaned CSVs, report, and plots will be saved.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging(verbose=args.verbose)

    try:
        work_dir = Path.cwd()
        input_path = Path(args.input).resolve() if args.input else autodetect_input_file(work_dir)
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / PLOTS_DIRNAME).mkdir(parents=True, exist_ok=True)

        raw_df = load_trades(input_path, logger)
        cleaned_df, quality = normalize_trades(raw_df, logger, parse_mixed_datetime_to_utc)

        logger.info("Calculating aggregate statistics")
        basic = basic_stats(cleaned_df)
        symbol_stats = build_symbol_stats(cleaned_df)
        side_stats = build_side_stats(cleaned_df)
        time_frames, time_stats = build_time_stats(cleaned_df)
        concentration = analyze_concentration(cleaned_df)
        profile_stats = build_profile_stats(cleaned_df, concentration, side_stats, symbol_stats)
        position_behavior = analyze_position_behavior(cleaned_df)
        holding_efficiency = analyze_holding_efficiency(cleaned_df, time_frames["holding_bucket"])
        style = classify_trader_style(cleaned_df, symbol_stats, side_stats, concentration)
        edge_checks = evaluate_edge_checks(cleaned_df)
        core_vs_noise = run_core_vs_noise_analysis(cleaned_df)
        entry_quality = run_entry_quality_analysis(cleaned_df, logger)
        market_context = run_market_context_analysis(cleaned_df, entry_quality["per_trade"], output_dir, logger)
        eth_context = run_eth_context_analysis(market_context["market_context_at_entry"], output_dir, logger)
        eth_setup_clusters = run_eth_setup_clustering(
            market_context["market_context_at_entry"],
            entry_quality["per_trade"],
            cleaned_df,
            output_dir,
            logger,
        )
        execution_mode_split = run_execution_mode_split_analysis(
            cleaned_df,
            entry_quality["per_trade"],
            eth_setup_clusters["clusters"],
            output_dir,
            logger,
        )
        eth_execution_aware = run_eth_execution_aware_archetype_analysis(
            eth_setup_clusters["clusters"],
            entry_quality["per_trade"],
            execution_mode_split["trade_split"],
            market_context["market_context_at_entry"],
            output_dir,
            logger,
        )
        eth_execution_reanchored = run_eth_execution_reanchored_entry_quality_analysis(
            eth_execution_aware["joined"],
            execution_mode_split["eth_trades"],
            output_dir,
            logger,
        )
        eth_cluster_rules = run_eth_cluster_rule_extraction(
            eth_setup_clusters["clusters"],
            eth_setup_clusters["cluster_summary"],
            output_dir,
            logger,
        )
        eth_replication = run_eth_replication_hypothesis_test(
            cleaned_df,
            eth_setup_clusters["clusters"],
            eth_setup_clusters["cluster_summary"],
            eth_cluster_rules["rules"],
            output_dir,
            logger,
        )
        eth_tightening = run_eth_filter_tightening(
            eth_setup_clusters["clusters"],
            entry_quality["per_trade"],
            eth_replication["signals_cluster1"],
            eth_replication["signals_cluster2"],
            output_dir,
            logger,
        )
        eth_bollinger = run_eth_bollinger_compatibility_test(
            eth_setup_clusters["clusters"],
            entry_quality["per_trade"],
            eth_replication["signals_cluster1"],
            eth_replication["signals_cluster2"],
            eth_tightening["summary"],
            output_dir,
            logger,
        )
        eth_cluster1_micro_trigger = run_eth_cluster1_micro_trigger_analysis(
            eth_setup_clusters["clusters"],
            entry_quality["per_trade"],
            eth_replication["signals_cluster1"],
            eth_tightening["summary"],
            output_dir,
            logger,
        )
        eth_cluster1_structural_confirmation = run_eth_cluster1_structural_confirmation_analysis(
            eth_setup_clusters["clusters"],
            entry_quality["per_trade"],
            eth_replication["signals_cluster1"],
            output_dir,
            logger,
        )
        eth_cluster1_effort_result = run_eth_cluster1_effort_result_analysis(
            eth_setup_clusters["clusters"],
            entry_quality["per_trade"],
            eth_replication["signals_cluster1"],
            eth_tightening["summary"],
            output_dir,
            logger,
        )
        eth_practical_imitation = run_eth_practical_imitation_model(
            cleaned_df,
            eth_setup_clusters["clusters"],
            output_dir,
            logger,
        )
        eth_short_proxy_validation = run_eth_short_proxy_validation(
            cleaned_df,
            output_dir,
            logger,
        )
        eth_cluster1_short_overlap = run_reanchored_cluster1_short_validation_analysis(
            cleaned_df,
            eth_execution_reanchored["per_trade"],
            eth_replication["signals_cluster1"],
            output_dir,
            logger,
        )
        eth_cluster1_short_refinement = run_cluster1_short_precision_refinement_analysis(
            cleaned_df,
            eth_cluster1_short_overlap["real_trades"],
            eth_cluster1_short_overlap["model_signals"],
            eth_replication["signals_cluster1"],
            output_dir,
            logger,
        )
        eth_cluster1_short_timing_error = run_cluster1_short_timing_error_anatomy_analysis(
            cleaned_df,
            eth_cluster1_short_overlap["overlap_matches"],
            eth_cluster1_short_refinement["results"],
            eth_cluster1_short_refinement["best_matches"],
            output_dir,
            logger,
        )
        eth_cluster1_short_final_timing = run_cluster1_short_final_timing_test_analysis(
            cleaned_df,
            eth_cluster1_short_overlap["real_trades"],
            eth_cluster1_short_overlap["model_signals"],
            eth_cluster1_short_timing_error["matches"],
            output_dir,
            logger,
        )

        logger.info("Running Monte Carlo behavioral replication")
        simulation_summary, simulation_paths = simulate_monte_carlo(
            cleaned_df,
            runs=SIMULATION_RUNS,
            seed=SIMULATION_SEED,
        )

        logger.info("Running optional EMA50 market filter check")
        market_filter_stats = None
        market_filter_note = "Market filter was not available."
        try:
            market_filter_result = build_market_filter(cleaned_df, logger)
            market_filter_stats = market_filter_result.stats
            market_filter_note = market_filter_result.note
        except Exception as market_exc:
            logger.warning("Market filter skipped: %s", market_exc)

        logger.info("Saving cleaned datasets")
        cleaned_df.to_csv(resolve_output_path(output_dir, "cleaned_trades"), index=False)
        symbol_stats.to_csv(resolve_output_path(output_dir, "symbol_stats"), index=False)
        side_stats.to_csv(resolve_output_path(output_dir, "side_stats"), index=False)
        time_stats.to_csv(resolve_output_path(output_dir, "time_stats"), index=False)
        profile_stats.to_csv(resolve_output_path(output_dir, "profile_stats"), index=False)
        simulation_summary.to_csv(resolve_output_path(output_dir, "simulation_summary"), index=False)
        edge_checks.to_csv(resolve_output_path(output_dir, "edge_checks"), index=False)
        core_vs_noise["summary"].to_csv(resolve_output_path(output_dir, "core_vs_noise_summary"), index=False)
        core_vs_noise["flagged_extreme_losers"].to_csv(resolve_output_path(output_dir, "flagged_extreme_losers"), index=False)
        core_vs_noise["flagged_faith_trades"].to_csv(resolve_output_path(output_dir, "flagged_faith_trades"), index=False)
        entry_quality["per_trade"].to_csv(resolve_output_path(output_dir, "entry_quality_per_trade"), index=False)
        entry_quality["excursion_metrics"].to_csv(resolve_output_path(output_dir, "excursion_metrics"), index=False)
        entry_quality["exit_efficiency"].to_csv(resolve_output_path(output_dir, "exit_efficiency"), index=False)
        entry_quality["entry_quality_summary"].to_csv(resolve_output_path(output_dir, "entry_quality_summary"), index=False)
        entry_quality["symbol_entry_quality"].to_csv(resolve_output_path(output_dir, "symbol_entry_quality"), index=False)
        entry_quality["side_entry_quality"].to_csv(resolve_output_path(output_dir, "side_entry_quality"), index=False)
        execution_mode_split["trade_split"].to_csv(resolve_output_path(output_dir, "execution_mode_split"), index=False)
        execution_mode_split["fill_delay_summary"].to_csv(resolve_output_path(output_dir, "execution_fill_delay_summary"), index=False)
        execution_mode_split["mode_comparison"].to_csv(resolve_output_path(output_dir, "execution_mode_comparison"), index=False)
        execution_mode_split["eth_trades"].to_csv(resolve_output_path(output_dir, "eth_execution_mode_trades"), index=False)
        eth_execution_aware["joined"].to_csv(resolve_output_path(output_dir, "eth_cluster_execution_joined"), index=False)
        eth_execution_aware["summary"].to_csv(resolve_output_path(output_dir, "eth_cluster_execution_summary"), index=False)
        eth_execution_aware["quality"].to_csv(resolve_output_path(output_dir, "eth_cluster_execution_quality"), index=False)
        eth_execution_aware["quality_two_bucket"].to_csv(resolve_output_path(output_dir, "eth_cluster_execution_quality_two_bucket"), index=False)
        eth_execution_reanchored["per_trade"].to_csv(resolve_output_path(output_dir, "eth_execution_reanchored_entry_quality"), index=False)
        eth_execution_reanchored["summary"].to_csv(resolve_output_path(output_dir, "eth_execution_reanchored_summary"), index=False)
        eth_execution_reanchored["cluster_quality"].to_csv(resolve_output_path(output_dir, "eth_execution_reanchored_cluster_quality"), index=False)
        eth_execution_reanchored["distortion"].to_csv(resolve_output_path(output_dir, "eth_execution_reanchored_distortion"), index=False)
        eth_cluster1_short_overlap["real_trades"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_real_trades"), index=False)
        eth_cluster1_short_overlap["model_signals"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_model_signals"), index=False)
        eth_cluster1_short_overlap["overlap_matches"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_overlap_matches"), index=False)
        eth_cluster1_short_overlap["overlap_summary"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_overlap_summary"), index=False)
        eth_cluster1_short_overlap["overlap_quality"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_overlap_quality"), index=False)
        eth_cluster1_short_refinement["variants"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_refinement_variants"), index=False)
        eth_cluster1_short_refinement["results"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_refinement_results"), index=False)
        eth_cluster1_short_refinement["best_matches"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_refinement_best_matches"), index=False)
        eth_cluster1_short_refinement["tradeoffs"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_refinement_tradeoffs"), index=False)
        eth_cluster1_short_timing_error["matches"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_timing_error_matches"), index=False)
        eth_cluster1_short_timing_error["summary"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_timing_error_summary"), index=False)
        eth_cluster1_short_timing_error["quality"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_timing_error_quality"), index=False)
        eth_cluster1_short_final_timing["variants"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_final_timing_variants"), index=False)
        eth_cluster1_short_final_timing["results"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_final_timing_results"), index=False)
        eth_cluster1_short_final_timing["tradeoffs"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_final_timing_tradeoffs"), index=False)
        eth_cluster1_short_final_timing["best_matches"].to_csv(resolve_output_path(output_dir, "eth_cluster1_short_final_timing_best_matches"), index=False)
        market_context["market_context_at_entry"].to_csv(resolve_output_path(output_dir, "market_context_at_entry"), index=False)
        market_context["strong_vs_weak_entry_comparison"].to_csv(resolve_output_path(output_dir, "strong_vs_weak_entry_comparison"), index=False)
        eth_context["eth_entry_context_comparison"].to_csv(resolve_output_path(output_dir, "eth_entry_context_comparison"), index=False)
        eth_context["eth_long_context_comparison"].to_csv(resolve_output_path(output_dir, "eth_long_context_comparison"), index=False)
        eth_context["eth_short_context_comparison"].to_csv(resolve_output_path(output_dir, "eth_short_context_comparison"), index=False)
        eth_bollinger["summary"].to_csv(resolve_output_path(output_dir, "eth_bollinger_compatibility_summary"), index=False)
        eth_bollinger["short_hypotheses"].to_csv(resolve_output_path(output_dir, "eth_bollinger_short_hypotheses"), index=False)
        eth_bollinger["long_hypotheses"].to_csv(resolve_output_path(output_dir, "eth_bollinger_long_hypotheses"), index=False)
        eth_setup_clusters["clusters"].to_csv(resolve_output_path(output_dir, "eth_setup_clusters"), index=False)
        eth_setup_clusters["cluster_summary"].to_csv(resolve_output_path(output_dir, "eth_cluster_summary"), index=False)
        eth_setup_clusters["best_cluster_profile"].to_csv(resolve_output_path(output_dir, "eth_best_cluster_profile"), index=False)
        eth_cluster_rules["feature_ranges"].to_csv(resolve_output_path(output_dir, "eth_cluster_feature_ranges"), index=False)
        eth_cluster_rules["rule_quality"].to_csv(resolve_output_path(output_dir, "eth_cluster_rule_quality"), index=False)
        eth_replication["signals_cluster1"].to_csv(resolve_output_path(output_dir, "eth_replication_signals_cluster1"), index=False)
        eth_replication["signals_cluster2"].to_csv(resolve_output_path(output_dir, "eth_replication_signals_cluster2"), index=False)
        eth_replication["summary"].to_csv(resolve_output_path(output_dir, "eth_replication_summary"), index=False)
        eth_replication["stage_comparison"].to_csv(resolve_output_path(output_dir, "eth_replication_stage_comparison"), index=False)
        eth_tightening["feature_ranking"].to_csv(resolve_output_path(output_dir, "eth_timing_feature_ranking"), index=False)
        eth_tightening["cluster1_candidates"].to_csv(resolve_output_path(output_dir, "eth_cluster1_tightened_candidates"), index=False)
        eth_tightening["cluster2_candidates"].to_csv(resolve_output_path(output_dir, "eth_cluster2_tightened_candidates"), index=False)
        eth_tightening["summary"].to_csv(resolve_output_path(output_dir, "eth_tightened_filter_summary"), index=False)
        eth_tightening["stage_comparison"].to_csv(resolve_output_path(output_dir, "eth_tightened_filter_stage_comparison"), index=False)
        eth_cluster1_micro_trigger["ranking"].to_csv(resolve_output_path(output_dir, "eth_cluster1_micro_trigger_ranking"), index=False)
        eth_cluster1_micro_trigger["candidates"].to_csv(resolve_output_path(output_dir, "eth_cluster1_micro_trigger_candidates"), index=False)
        eth_cluster1_micro_trigger["quality"].to_csv(resolve_output_path(output_dir, "eth_cluster1_micro_trigger_quality"), index=False)
        eth_cluster1_structural_confirmation["ranking"].to_csv(resolve_output_path(output_dir, "eth_cluster1_structural_confirmation_ranking"), index=False)
        eth_cluster1_structural_confirmation["candidates"].to_csv(resolve_output_path(output_dir, "eth_cluster1_final_trigger_candidates"), index=False)
        eth_cluster1_structural_confirmation["quality"].to_csv(resolve_output_path(output_dir, "eth_cluster1_final_trigger_quality"), index=False)
        eth_cluster1_effort_result["hypothesis"].to_csv(resolve_output_path(output_dir, "eth_cluster1_effort_result_hypothesis"), index=False)
        eth_cluster1_effort_result["ranking"].to_csv(resolve_output_path(output_dir, "eth_cluster1_effort_result_ranking"), index=False)
        eth_cluster1_effort_result["quality"].to_csv(resolve_output_path(output_dir, "eth_cluster1_effort_result_quality"), index=False)
        eth_practical_imitation["short_results"].to_csv(resolve_output_path(output_dir, "eth_proxy_short_results"), index=False)
        eth_practical_imitation["long_results"].to_csv(resolve_output_path(output_dir, "eth_proxy_long_results"), index=False)
        eth_practical_imitation["stage_comparison"].to_csv(resolve_output_path(output_dir, "eth_proxy_stage_comparison"), index=False)
        eth_practical_imitation["exit_comparison"].to_csv(resolve_output_path(output_dir, "eth_proxy_exit_comparison"), index=False)
        eth_short_proxy_validation["validation_summary"].to_csv(resolve_output_path(output_dir, "eth_short_proxy_validation_summary"), index=False)
        eth_short_proxy_validation["friction_comparison"].to_csv(resolve_output_path(output_dir, "eth_short_proxy_friction_comparison"), index=False)
        eth_short_proxy_validation["parameter_robustness"].to_csv(resolve_output_path(output_dir, "eth_short_proxy_parameter_robustness"), index=False)
        eth_short_proxy_validation["stage_consistency"].to_csv(resolve_output_path(output_dir, "eth_short_proxy_stage_consistency"), index=False)
        eth_short_proxy_validation["monthly_breakdown"].to_csv(resolve_output_path(output_dir, "eth_short_proxy_monthly_breakdown"), index=False)
        if market_filter_stats is not None:
            market_filter_stats.to_csv(resolve_output_path(output_dir, "market_filter_stats"), index=False)

        logger.info("Building text report")
        report_text = generate_summary_report(
            cleaned_df,
            quality,
            basic,
            profile_stats,
            symbol_stats,
            side_stats,
            time_frames,
            concentration,
            position_behavior,
            holding_efficiency,
            style,
            edge_checks,
            simulation_summary,
            market_filter_stats,
            market_filter_note,
            entry_quality["section_text"],
            execution_mode_split["section_text"],
            market_context["context_summary_text"],
            eth_context["summary_text"],
            eth_bollinger["section_text"],
            eth_setup_clusters["section_text"],
            eth_execution_aware["section_text"],
            eth_execution_reanchored["section_text"],
            eth_cluster1_short_overlap["section_text"],
            eth_cluster1_short_refinement["section_text"],
            eth_cluster1_short_timing_error["section_text"],
            eth_cluster1_short_final_timing["section_text"],
            eth_cluster_rules["section_text"],
            eth_replication["section_text"],
            eth_tightening["section_text"],
            eth_cluster1_micro_trigger["section_text"],
            eth_cluster1_structural_confirmation["section_text"],
            eth_cluster1_effort_result["section_text"],
            eth_practical_imitation["section_text"],
            eth_short_proxy_validation["section_text"],
            core_vs_noise["section_text"],
        )
        resolve_output_path(output_dir, "summary_report").write_text(report_text, encoding="utf-8")

        logger.info("Rendering plots with matplotlib")
        create_plots(cleaned_df, symbol_stats, side_stats, output_dir, logger)
        create_extended_plots(cleaned_df, simulation_summary, simulation_paths, output_dir, logger)
        create_core_vs_noise_plots(core_vs_noise["scenario_frames"], output_dir, logger)
        create_entry_quality_plots(entry_quality["per_trade"], output_dir, logger)
        create_execution_mode_split_plots(
            execution_mode_split["trade_split"],
            output_dir,
            logger,
        )
        create_market_context_plot(market_context["market_context_at_entry"], output_dir, logger)
        create_eth_context_plots(eth_context["eth_context_frame"], output_dir, logger)
        create_eth_bollinger_compatibility_plots(
            eth_bollinger["short_hypotheses"],
            eth_bollinger["long_hypotheses"],
            output_dir,
            logger,
        )
        create_eth_setup_cluster_plots(
            eth_setup_clusters["clusters"],
            eth_setup_clusters["cluster_summary"],
            output_dir,
            logger,
        )
        create_eth_execution_aware_archetype_plots(
            eth_execution_aware["summary"],
            eth_execution_aware["quality"],
            output_dir,
            logger,
        )
        create_eth_execution_reanchored_plots(
            eth_execution_reanchored["summary"],
            eth_execution_reanchored["cluster_quality"],
            output_dir,
            logger,
        )
        create_reanchored_cluster1_short_validation_plots(
            eth_cluster1_short_overlap["overlap_matches"],
            eth_cluster1_short_overlap["overlap_quality"],
            output_dir,
            logger,
        )
        create_cluster1_short_precision_refinement_plots(
            eth_cluster1_short_refinement["results"],
            output_dir,
            logger,
        )
        create_cluster1_short_timing_error_plots(
            eth_cluster1_short_timing_error["matches"],
            output_dir,
            logger,
        )
        create_cluster1_short_final_timing_test_plots(
            eth_cluster1_short_final_timing["results"],
            output_dir,
            logger,
        )
        create_eth_cluster_rule_plots(
            eth_cluster_rules["feature_ranges"],
            eth_cluster_rules["rule_quality"],
            output_dir,
            logger,
        )
        create_eth_replication_plots(
            eth_replication["summary"],
            eth_replication["combined_signals"],
            output_dir,
            logger,
        )
        create_eth_tightening_plots(
            eth_tightening["feature_ranking"],
            eth_tightening["summary"],
            output_dir,
            logger,
        )
        create_eth_cluster1_micro_trigger_plots(
            eth_cluster1_micro_trigger["ranking"],
            eth_cluster1_micro_trigger["quality"],
            output_dir,
            logger,
        )
        create_eth_cluster1_structural_confirmation_plots(
            eth_cluster1_structural_confirmation["ranking"],
            eth_cluster1_structural_confirmation["quality"],
            output_dir,
            logger,
        )
        create_eth_cluster1_effort_result_plots(
            eth_cluster1_effort_result["ranking"],
            eth_cluster1_effort_result["quality"],
            output_dir,
            logger,
        )
        create_eth_proxy_plots(
            eth_practical_imitation["short_results"],
            eth_practical_imitation["long_results"],
            eth_practical_imitation["exit_comparison"],
            output_dir,
            logger,
        )
        create_eth_short_proxy_validation_plots(
            eth_short_proxy_validation["validation_summary"],
            eth_short_proxy_validation["net_vs_gross_curves"],
            eth_short_proxy_validation["parameter_robustness"],
            output_dir,
            logger,
        )

        print("")
        print(console_summary(basic, symbol_stats, side_stats, concentration))
        print("")
        print(f"outputs saved to: {output_dir}")
        return 0
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
