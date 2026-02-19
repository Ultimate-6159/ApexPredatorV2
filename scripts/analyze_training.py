"""
Analyze Training Logs — View and analyze historical training data.

Usage:
    python -m scripts.analyze_training --regime trending_up
    python -m scripts.analyze_training --session 20240115_143052
    python -m scripts.analyze_training --all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TRAINING_LOG_DIR = Path("logs/training")


def list_sessions(regime: str | None = None) -> list[dict]:
    """List all training sessions, optionally filtered by regime."""
    sessions = []
    
    if not TRAINING_LOG_DIR.exists():
        logger.warning("No training logs found at %s", TRAINING_LOG_DIR)
        return sessions
    
    regimes = [regime.lower()] if regime else [d.name for d in TRAINING_LOG_DIR.iterdir() if d.is_dir()]
    
    for reg in regimes:
        regime_dir = TRAINING_LOG_DIR / reg
        if not regime_dir.exists():
            continue
        
        for session_dir in regime_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            summary_file = session_dir / "summary.json"
            config_file = session_dir / "config.json"
            
            session_info = {
                "regime": reg,
                "session_id": session_dir.name,
                "path": str(session_dir),
            }
            
            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    session_info.update(summary)
            elif config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    session_info.update(config)
            
            sessions.append(session_info)
    
    return sorted(sessions, key=lambda x: x.get("start_time", ""), reverse=True)


def print_sessions(sessions: list[dict]) -> None:
    """Print session list in a formatted table."""
    if not sessions:
        print("No training sessions found.")
        return
    
    print("\n" + "═" * 100)
    print(f"{'Regime':<20} {'Session ID':<20} {'Episodes':<10} {'Reward Mean':<15} {'Duration':<15}")
    print("═" * 100)
    
    for s in sessions:
        regime = s.get("regime", "N/A")
        session_id = s.get("session_id", "N/A")
        episodes = s.get("total_episodes", "N/A")
        reward_mean = s.get("final_reward_mean", "N/A")
        duration = s.get("duration_seconds", 0)
        
        if isinstance(reward_mean, float):
            reward_mean = f"{reward_mean:.4f}"
        if isinstance(duration, (int, float)):
            duration = f"{duration/60:.1f} min"
        
        print(f"{regime:<20} {session_id:<20} {str(episodes):<10} {str(reward_mean):<15} {str(duration):<15}")
    
    print("═" * 100)


def analyze_session(regime: str, session_id: str) -> None:
    """Analyze a specific training session."""
    session_dir = TRAINING_LOG_DIR / regime.lower() / session_id
    
    if not session_dir.exists():
        logger.error("Session not found: %s", session_dir)
        return
    
    print(f"\n{'═' * 80}")
    print(f"TRAINING SESSION ANALYSIS")
    print(f"{'═' * 80}")
    
    # Load summary
    summary_file = session_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        print(f"\n📊 Session Summary:")
        print(f"  Regime:           {summary.get('regime', 'N/A')}")
        print(f"  Start Time:       {summary.get('start_time', 'N/A')}")
        print(f"  End Time:         {summary.get('end_time', 'N/A')}")
        print(f"  Duration:         {summary.get('duration_seconds', 0)/60:.1f} minutes")
        print(f"  Total Timesteps:  {summary.get('total_timesteps', 'N/A'):,}")
        print(f"  Total Episodes:   {summary.get('total_episodes', 'N/A'):,}")
        print(f"\n📈 Performance:")
        print(f"  Final Reward Mean: {summary.get('final_reward_mean', 0):.4f}")
        print(f"  Final Reward Std:  {summary.get('final_reward_std', 0):.4f}")
        print(f"  Best Episode:      {summary.get('best_episode_reward', 0):.4f}")
        print(f"  Worst Episode:     {summary.get('worst_episode_reward', 0):.4f}")
    
    # Load episode data
    episodes_file = session_dir / "episodes.parquet"
    if episodes_file.exists():
        episodes_df = pd.read_parquet(episodes_file)
        
        print(f"\n📉 Episode Statistics:")
        print(f"  Total Episodes: {len(episodes_df)}")
        print(f"  Avg Reward:     {episodes_df['reward'].mean():.4f}")
        print(f"  Std Reward:     {episodes_df['reward'].std():.4f}")
        print(f"  Min Reward:     {episodes_df['reward'].min():.4f}")
        print(f"  Max Reward:     {episodes_df['reward'].max():.4f}")
        print(f"  Avg Length:     {episodes_df['length'].mean():.1f} steps")
        
        # Learning progress (first 10% vs last 10%)
        n = len(episodes_df)
        if n >= 20:
            first_10pct = episodes_df.head(n // 10)['reward'].mean()
            last_10pct = episodes_df.tail(n // 10)['reward'].mean()
            improvement = ((last_10pct - first_10pct) / abs(first_10pct) * 100) if first_10pct != 0 else 0
            
            print(f"\n🎯 Learning Progress:")
            print(f"  First 10% Avg:  {first_10pct:.4f}")
            print(f"  Last 10% Avg:   {last_10pct:.4f}")
            print(f"  Improvement:    {improvement:+.1f}%")
    
    # Load observation statistics
    obs_stats_file = session_dir / "obs_stats.json"
    if obs_stats_file.exists():
        with open(obs_stats_file, "r", encoding="utf-8") as f:
            obs_stats = json.load(f)
        
        print(f"\n📊 Observation Statistics (Feature Ranges):")
        feature_names = [
            "rsi_fast", "rsi_slow", "bb_width", "dist_ema50", "dist_ema200",
            "adx", "plus_di", "minus_di", "atr_norm", "volatility_ratio",
            "volume_zscore", "close_return", "ema_cross"
        ]
        
        means = obs_stats.get("mean", [])
        stds = obs_stats.get("std", [])
        mins = obs_stats.get("min", [])
        maxs = obs_stats.get("max", [])
        
        print(f"  {'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*55}")
        
        for i, name in enumerate(feature_names[:len(means)]):
            print(f"  {name:<15} {means[i]:>10.4f} {stds[i]:>10.4f} {mins[i]:>10.4f} {maxs[i]:>10.4f}")
    
    # Load timestep logs for action distribution
    timesteps_file = session_dir / "timesteps.parquet"
    if timesteps_file.exists():
        timesteps_df = pd.read_parquet(timesteps_file)
        
        # Get last action distribution
        action_cols = [col for col in timesteps_df.columns if col.startswith("action_") and col.endswith("_pct")]
        if action_cols and len(timesteps_df) > 0:
            last_row = timesteps_df.iloc[-1]
            
            print(f"\n🎮 Final Action Distribution:")
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            for col in sorted(action_cols):
                action_id = int(col.split("_")[1])
                action_name = action_names.get(action_id, f"Action {action_id}")
                print(f"  {action_name}: {last_row[col]*100:.1f}%")
    
    print(f"\n{'═' * 80}")
    print(f"Log files location: {session_dir}")
    print(f"{'═' * 80}\n")


def compare_sessions(regime: str) -> None:
    """Compare all sessions for a specific regime."""
    sessions = list_sessions(regime)
    
    if len(sessions) < 2:
        logger.info("Need at least 2 sessions to compare.")
        return
    
    print(f"\n{'═' * 100}")
    print(f"SESSION COMPARISON: {regime.upper()}")
    print(f"{'═' * 100}")
    
    # Create comparison DataFrame
    comparison_data = []
    for s in sessions:
        comparison_data.append({
            "Session": s.get("session_id", "N/A"),
            "Episodes": s.get("total_episodes", 0),
            "Timesteps": s.get("total_timesteps", 0),
            "Reward Mean": s.get("final_reward_mean", 0),
            "Reward Std": s.get("final_reward_std", 0),
            "Best": s.get("best_episode_reward", 0),
            "Duration (min)": s.get("duration_seconds", 0) / 60,
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Best session
    if len(df) > 0:
        best_idx = df["Reward Mean"].idxmax()
        print(f"\n🏆 Best performing session: {df.loc[best_idx, 'Session']}")
        print(f"   Reward Mean: {df.loc[best_idx, 'Reward Mean']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--regime", type=str, help="Filter by regime (e.g., trending_up)")
    parser.add_argument("--session", type=str, help="Specific session ID to analyze")
    parser.add_argument("--all", action="store_true", help="List all sessions")
    parser.add_argument("--compare", action="store_true", help="Compare sessions for a regime")
    args = parser.parse_args()
    
    if args.session and args.regime:
        analyze_session(args.regime, args.session)
    elif args.compare and args.regime:
        compare_sessions(args.regime)
    elif args.all or args.regime:
        sessions = list_sessions(args.regime)
        print_sessions(sessions)
    else:
        # Default: list all sessions
        sessions = list_sessions()
        print_sessions(sessions)
        
        if sessions:
            print("\nTip: Use --regime <name> --session <id> to analyze a specific session")
            print("     Use --regime <name> --compare to compare sessions")


if __name__ == "__main__":
    main()
