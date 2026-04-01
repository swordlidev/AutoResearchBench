#!/usr/bin/env python3
"""
Universal AI Autonomous Research Agent launch script

Usage:
    python run_universal_agent.py                             # Use default algorithm (nanochat)
    python run_universal_agent.py --algorithm nanochat
    python run_universal_agent.py --algorithm ViT --llm gpt-5
    python run_universal_agent.py --algorithm ViT --llm gemini-3-pro-preview --max-tool-calls 8

Notes:
    - Original code files are never modified
    - All generated code is saved in experiments/{algorithm_name}/{llm_name}/ directory
    - The entire research process is a continuous Agent Loop (auto baseline → LLM multi-turn tool calls)
"""

import sys
import argparse
from pathlib import Path

from autoresearch import UniversalAutoResearchAgent


def list_available_algorithms():
    """List all available algorithms"""
    algorithms_dir = Path("algorithms")
    if not algorithms_dir.exists():
        return []
    
    algorithms = []
    for algo_dir in algorithms_dir.iterdir():
        if algo_dir.is_dir() and (algo_dir / "train.py").exists():
            has_evaluator = (algo_dir / "evaluator.py").exists()
            algorithms.append({
                "name": algo_dir.name,
                "path": str(algo_dir),
                "has_evaluator": has_evaluator
            })
    return algorithms


def main():
    parser = argparse.ArgumentParser(
        description="Universal AI Autonomous Research Agent (Agentic Loop)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_universal_agent.py --algorithm nanochat
    python run_universal_agent.py --algorithm ViT --llm gpt-5
    python run_universal_agent.py --algorithm ViT --llm gemini-3-pro-preview --max-tool-calls 8
    python run_universal_agent.py --list
        """
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="nanochat",
        help="Algorithm name (directory name under algorithms/)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model name (e.g. gpt-5, gemini-3-pro-preview, claude-sonnet-4, etc.)"
    )
    parser.add_argument(
        "--max-tool-calls", "-m",
        type=int,
        default=5,
        help="Maximum tool calls in Agent Loop (default: 5)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available algorithms"
    )
    
    args = parser.parse_args()
    
    # List available algorithms
    if args.list:
        algorithms = list_available_algorithms()
        if not algorithms:
            print("❌ No available algorithms found")
            print("💡 Please create an algorithm directory under algorithms/ containing a train.py file")
            return
        
        print("📋 Available algorithms:")
        print("-" * 60)
        for algo in algorithms:
            evaluator_status = "✅ Has evaluator.py" if algo["has_evaluator"] else "⚠️ Uses default evaluator"
            print(f"  {algo['name']:20s} {evaluator_status}")
        print("-" * 60)
        return
    
    # Check algorithm directory
    algorithm_dir = Path("algorithms") / args.algorithm
    code_path = algorithm_dir / "train.py"
    
    if not algorithm_dir.exists():
        print(f"❌ Error: algorithm directory does not exist {algorithm_dir}")
        print("\n💡 Available algorithms:")
        for algo in list_available_algorithms():
            print(f"   - {algo['name']}")
        sys.exit(1)
    
    if not code_path.exists():
        print(f"❌ Error: training code not found {code_path}")
        sys.exit(1)
    
    # Create agent and run
    agent = UniversalAutoResearchAgent(
        algorithm_dir=str(algorithm_dir),
        max_tool_calls=args.max_tool_calls,
        model_name=args.llm,
    )
    
    try:
        agent.run(str(code_path))
        print("\n✅ Research complete!")
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
