#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test LLMClient class

Usage:
    python test_llm_client.py                          # Use default model
    python test_llm_client.py --llm gpt-5
    python test_llm_client.py --llm gemini-3-pro-preview
    python test_llm_client.py --llm claude-sonnet-4 --prompt "What is 1+1?"
"""

import argparse
from autoresearch import APIConfig, LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test LLMClient with different LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_llm_client.py
    python test_llm_client.py --llm gpt-5
    python test_llm_client.py --llm gemini-3-pro-preview
    python test_llm_client.py --llm claude-sonnet-4 --prompt "Explain transformers in one sentence."
        """
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model name (e.g. gpt-5, gemini-3-pro-preview, claude-sonnet-4). Default: use config default"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Introduce yourself in one sentence.",
        help="Test prompt to send to the LLM (default: 'Introduce yourself in one sentence.')"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="Temperature parameter (default: use config default)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens (default: use config default)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Test LLMClient")
    print("=" * 60)

    # Initialize config
    config = APIConfig()

    # Override model name if provided
    if args.llm:
        config.model_name = args.llm

    print(f"API URL: {config.url}")
    print(f"Model: {config.model_type}/{config.model_name}")
    print(f"Username: {config.username}")
    print(f"Temperature: {config.temperature}")
    print(f"Max Tokens: {config.max_tokens}")
    print("=" * 60)

    # Validate config
    try:
        config.validate()
        print("✅ Config validation passed")
    except ValueError as e:
        print(f"❌ Config validation failed: {e}")
        return

    # Initialize client
    client = LLMClient(config)

    # Test call (multi-turn conversation mode)
    print(f"\n📡 Sending test request to [{config.model_name}]...")
    print(f"📝 Prompt: {args.prompt}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]

    # Build optional overrides
    call_kwargs = {}
    if args.temperature is not None:
        call_kwargs["temperature"] = args.temperature
    if args.max_tokens is not None:
        call_kwargs["max_tokens"] = args.max_tokens

    try:
        response = client.call_with_messages(messages=messages, **call_kwargs)
        print("\n✅ Request successful!")
        print("=" * 60)
        print("📝 LLM Response:")
        print(response)
        print("=" * 60)

        # Print token usage stats
        print(f"\n{client.token_stats.summary()}")
    except Exception as e:
        print(f"\n❌ Request failed: {e}")


if __name__ == "__main__":
    main()
