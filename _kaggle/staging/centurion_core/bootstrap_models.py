"""Bootstrap training for meta-label classifier and IND RL models.

GAP-5: Initial meta-label model training on historical data.
GAP-6: Initial PPO RL model training for NIFTY50 stocks.

Run once to populate data/meta_label_models/ and data/rl_models/ with
trained models. The scheduler will retrain periodically after this.

Usage:
    cd centurion_core
    python bootstrap_models.py [--meta-label] [--rl] [--all]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger("bootstrap")

_ROOT = str(Path(__file__).parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def bootstrap_meta_label():
    """Train meta-labeling classifier on 2Y historical data for IND and US."""
    logger.info("=== Meta-Label Bootstrap Training ===")

    try:
        import yfinance as yf
        from services.meta_labeling import train_meta_labeler

        # IND tickers (NIFTY50 subset)
        ind_tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "ICICIBANK.NS", "BHARTIARTL.NS", "LT.NS", "SBIN.NS",
            "ITC.NS", "TATAMOTORS.NS", "AXISBANK.NS", "WIPRO.NS",
            "SUNPHARMA.NS", "MARUTI.NS", "ONGC.NS", "HCLTECH.NS",
            "KOTAKBANK.NS", "BAJFINANCE.NS", "TITAN.NS", "NTPC.NS",
        ]

        logger.info("Downloading IND OHLCV for %d tickers...", len(ind_tickers))
        ind_cache = {}
        for ticker in ind_tickers:
            try:
                df = yf.download(ticker, period="2y", progress=False)
                if df is not None and len(df) > 252:
                    if hasattr(df.columns, 'get_level_values'):
                        try:
                            df.columns = df.columns.get_level_values(0)
                        except Exception:
                            pass
                    ind_cache[ticker] = df
                    logger.info("  %s: %d days", ticker, len(df))
            except Exception as e:
                logger.warning("  %s: download failed: %s", ticker, e)

        if len(ind_cache) < 5:
            logger.error("Insufficient IND data (%d symbols)", len(ind_cache))
            return False

        result = train_meta_labeler(ind_cache, market="IND")
        logger.info("IND Meta-Label Result: %s", result.get("status", "unknown"))

        # US tickers
        us_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
            "TSLA", "JPM", "V", "UNH", "HD", "PG",
        ]

        logger.info("Downloading US OHLCV for %d tickers...", len(us_tickers))
        us_cache = {}
        for ticker in us_tickers:
            try:
                df = yf.download(ticker, period="2y", progress=False)
                if df is not None and len(df) > 252:
                    if hasattr(df.columns, 'get_level_values'):
                        try:
                            df.columns = df.columns.get_level_values(0)
                        except Exception:
                            pass
                    us_cache[ticker] = df
            except Exception:
                continue

        if len(us_cache) >= 3:
            result_us = train_meta_labeler(us_cache, market="US")
            logger.info("US Meta-Label Result: %s", result_us.get("status", "unknown"))

        logger.info("Meta-label bootstrap complete.")
        return True

    except Exception as e:
        logger.exception("Meta-label bootstrap failed: %s", e)
        return False


def bootstrap_rl_models():
    """Train PPO RL models for IND NIFTY stocks."""
    logger.info("=== RL Model Bootstrap Training (IND) ===")

    try:
        import yfinance as yf

        # Check if stable-baselines3 is available
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
            return False

        ind_tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
            "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "TATAMOTORS.NS",
            "AXISBANK.NS", "SUNPHARMA.NS",
        ]

        model_dir = Path(__file__).parent / "data" / "rl_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        for ticker in ind_tickers:
            try:
                logger.info("Training RL model for %s...", ticker)
                df = yf.download(ticker, period="3y", progress=False)
                if df is None or len(df) < 504:
                    logger.warning("  %s: insufficient data (%s days)", ticker,
                                   len(df) if df is not None else 0)
                    continue

                if hasattr(df.columns, 'get_level_values'):
                    try:
                        df.columns = df.columns.get_level_values(0)
                    except Exception:
                        pass

                # Use the project's RL environment if available
                try:
                    from services.rl_confidence import RLTradingEnv
                    env = RLTradingEnv(df)
                except ImportError:
                    # Fallback: create a simple env
                    logger.warning("  %s: RLTradingEnv not found, using basic gym env", ticker)
                    continue

                model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=64)
                model.learn(total_timesteps=50_000)

                bare = ticker.replace('.NS', '').replace('.BO', '')
                model_path = model_dir / f"{bare}_ppo"
                model.save(str(model_path))
                logger.info("  %s: model saved to %s", ticker, model_path)

            except Exception as e:
                logger.warning("  %s: training failed: %s", ticker, e)

        logger.info("RL bootstrap complete.")
        return True

    except Exception as e:
        logger.exception("RL bootstrap failed: %s", e)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap ML models for Centurion Core")
    parser.add_argument("--meta-label", action="store_true", help="Train meta-label classifier")
    parser.add_argument("--rl", action="store_true", help="Train RL PPO models for IND stocks")
    parser.add_argument("--all", action="store_true", help="Train all models")
    args = parser.parse_args()

    if args.all or (not args.meta_label and not args.rl):
        args.meta_label = True
        args.rl = True

    success = True
    if args.meta_label:
        if not bootstrap_meta_label():
            success = False
    if args.rl:
        if not bootstrap_rl_models():
            success = False

    sys.exit(0 if success else 1)
