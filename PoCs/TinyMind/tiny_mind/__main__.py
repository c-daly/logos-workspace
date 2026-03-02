#!/usr/bin/env python3
"""
TinyMind CLI - A baby intelligence that learns through conversation.

Usage:
    python -m tiny_mind chat                     # Interactive chat
    python -m tiny_mind read doc.pdf             # Read a PDF
    python -m tiny_mind know [topic]             # Query knowledge
    python -m tiny_mind viz                      # Visualize graph
    python -m tiny_mind forget <topic>           # Forget something
    python -m tiny_mind research <topic>         # Research a topic via web search
"""

import argparse
import os
import sys

from tiny_mind.conversation.mind import TinyMind, chat


def get_mind(args) -> TinyMind:
    """Create a TinyMind instance from CLI arguments."""
    return TinyMind(
        name=args.name,
        llm_provider=args.provider,
        model=args.model,
        critic_provider=args.critic_provider or args.provider,
        critic_model=args.critic_model,
        use_critic=not args.no_critic,
        save_path=args.save_path,
    )


def cmd_chat(args):
    """Run interactive chat session."""
    # Set environment variables from args so chat() picks them up
    if args.provider:
        os.environ["TINYMIND_PROVIDER"] = args.provider
    if args.model:
        os.environ["TINYMIND_MODEL"] = args.model
    if args.critic_model:
        os.environ["TINYMIND_CRITIC_MODEL"] = args.critic_model
    os.environ["TINYMIND_USE_CRITIC"] = "false" if args.no_critic else "true"
    
    chat()


def cmd_read(args):
    """Read a PDF file."""
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    if not args.file.lower().endswith('.pdf'):
        print("Error: Only PDF files are currently supported")
        sys.exit(1)
    
    mind = get_mind(args)
    
    # Parse page range if provided
    start_page = 0
    end_page = None
    if args.pages:
        try:
            if '-' in args.pages:
                start, end = args.pages.split('-')
                start_page = int(start) - 1  # Convert to 0-indexed
                end_page = int(end)
            else:
                # Single page
                start_page = int(args.pages) - 1
                end_page = int(args.pages)
        except ValueError:
            print(f"Error: Invalid page range: {args.pages}")
            print("Use format: 1-50 or just 10")
            sys.exit(1)
    
    print(f"Reading: {args.file}")
    if args.pages:
        print(f"Pages: {start_page + 1} to {end_page or 'end'}")
    
    result = mind.read_pdf(
        filepath=args.file,
        start_page=start_page,
        end_page=end_page,
        parallel=not args.sequential,
        batch_size=args.batch_size,
        max_workers=args.workers,
        max_chars_per_chunk=args.chunk_size,
        verbose=not args.quiet,
    )
    
    print(f"\nDone! Learned {result['nodes_created']} nodes, {result['edges_created']} edges")
    
    if args.save:
        mind.save()
        print(f"Saved to {mind.save_path}")


def cmd_know(args):
    """Query what the mind knows."""
    mind = get_mind(args)
    
    if len(mind.graph) == 0:
        print("I don't know anything yet. Try reading some documents or chatting with me!")
        return
    
    result = mind.know(args.topic)
    print(result)


def cmd_viz(args):
    """Visualize the knowledge graph."""
    mind = get_mind(args)

    if len(mind.graph) == 0:
        print("Nothing to visualize - the knowledge graph is empty.")
        return

    if args.html:
        output = args.output or f"./{mind.name.lower()}_graph.html"
        path = mind.visualize_interactive(
            output_path=output,
            open_browser=not args.no_open,
            show_edges=not args.clusters,
        )
        print(f"Generated: {path}")
    else:
        print(mind.visualize())


def cmd_forget(args):
    """Forget a topic."""
    mind = get_mind(args)
    result = mind.forget(args.topic)
    print(result)
    mind.save()


def cmd_reflect(args):
    """Reflect on knowledge."""
    mind = get_mind(args)
    print(mind.reflect())


def cmd_revise(args):
    """Run maintenance pass on knowledge graph."""
    mind = get_mind(args)

    if len(mind.graph) == 0:
        print("Nothing to revise - the knowledge graph is empty.")
        return

    result = mind.revise(verbose=not args.quiet)

    # Save by default unless --no-save is specified
    if not getattr(args, 'no_save', False):
        mind.save()
        print(f"\nSaved to {mind.save_path}")


def cmd_audit(args):
    """Deep audit to find misplaced nodes."""
    from tiny_mind.revision.reviser import Reviser

    mind = get_mind(args)

    if len(mind.graph) == 0:
        print("Nothing to audit - the knowledge graph is empty.")
        return

    reviser = Reviser()
    dry_run = not getattr(args, 'apply', False)

    results = reviser.deep_audit(
        mind.graph,
        verbose=not args.quiet,
        dry_run=dry_run,
    )

    if results and not dry_run:
        if not getattr(args, 'no_save', False):
            mind.save()
            print(f"\nSaved to {mind.save_path}")
    elif results and dry_run:
        print(f"\n(dry run - use --apply to make changes)")


def cmd_cluster(args):
    """Cluster nodes by embedding similarity."""
    from tiny_mind.revision.reviser import Reviser

    mind = get_mind(args)

    if len(mind.graph) == 0:
        print("Nothing to cluster - the knowledge graph is empty.")
        return

    reviser = Reviser()

    node_to_cluster = reviser.cluster_by_embeddings(
        mind.graph,
        verbose=not args.quiet,
        min_cluster_size=args.min_size,
        n_clusters=args.n_clusters,
        use_hdbscan=args.hdbscan,
    )

    if node_to_cluster:
        mind.save()
        print(f"\nSaved to {mind.save_path}")
        print("Run /viz-html to see clustered visualization")


def cmd_wonder(args):
    """Show what TinyMind is curious about."""
    mind = get_mind(args)
    
    if len(mind.graph) == 0:
        print("I don't know anything yet - nothing to be curious about!")
        return
    
    goals = mind.wonder(limit=args.limit)
    
    if not goals:
        print("I'm not particularly curious about anything right now.")
        return
    
    print("\nWhat I'm curious about:\n")
    for i, goal in enumerate(goals, 1):
        print(f"{i}. [{goal.type.value.upper()}] {goal.question}")
        print(f"   Priority: {goal.priority:.2f}")
        if goal.context:
            for k, v in list(goal.context.items())[:2]:
                print(f"   {k}: {v}")
        print()


def cmd_explore(args):
    """Investigate top curiosity goal."""
    mind = get_mind(args)
    
    if len(mind.graph) == 0:
        print("I don't know anything yet - nothing to explore!")
        return
    
    result = mind.explore(verbose=not args.quiet)
    
    if result.success:
        if args.save:
            mind.save()
            print(f"\nSaved to {mind.save_path}")
    else:
        print(f"Investigation failed: {result.errors}")


def cmd_ponder(args):
    """Autonomous curiosity-driven exploration."""
    mind = get_mind(args)

    if len(mind.graph) == 0:
        print("I don't know anything yet - nothing to ponder!")
        return

    results = mind.ponder(cycles=args.cycles, verbose=not args.quiet)

    if args.save:
        mind.save()
        print(f"\nSaved to {mind.save_path}")


def cmd_research(args):
    """Research a specific topic."""
    mind = get_mind(args)

    result = mind.research(args.topic, verbose=not args.quiet)

    if result.success:
        if args.save:
            mind.save()
            print(f"\nSaved to {mind.save_path}")
    else:
        print(f"Research failed: {result.errors}")


def main():
    parser = argparse.ArgumentParser(
        prog="tinymind",
        description="TinyMind - A baby intelligence that learns through conversation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tiny_mind chat                        Interactive chat
  python -m tiny_mind read book.pdf               Read entire PDF
  python -m tiny_mind read book.pdf --pages 1-50  Read pages 1-50
  python -m tiny_mind know matrix                 What do I know about matrices?
  python -m tiny_mind research "quantum computing" --save   Research a topic
  python -m tiny_mind viz                         Text visualization
  python -m tiny_mind viz --html                  Interactive HTML visualization
  
Environment variables:
  OPENAI_API_KEY          Your OpenAI API key
  TINYMIND_PROVIDER       LLM provider (openai, anthropic)
  TINYMIND_MODEL          Model for extraction
  TINYMIND_CRITIC_MODEL   Model for critique
""",
    )
    
    # Global options
    parser.add_argument(
        "--name", "-n",
        default="Tiny",
        help="Name for the mind (default: Tiny)"
    )
    parser.add_argument(
        "--provider", "-p",
        default=os.environ.get("TINYMIND_PROVIDER", "openai"),
        help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("TINYMIND_MODEL", "gpt-4o"),
        help="Model for extraction (default: gpt-4o)"
    )
    parser.add_argument(
        "--critic-provider",
        help="Provider for critic (default: same as --provider)"
    )
    parser.add_argument(
        "--critic-model",
        default=os.environ.get("TINYMIND_CRITIC_MODEL", "gpt-4o-mini"),
        help="Model for critique (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--no-critic",
        action="store_true",
        help="Disable the critic (faster but less accurate)"
    )
    parser.add_argument(
        "--save-path", "-s",
        help="Path to save/load mind state (default: ./tiny_mind.json)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat session")
    chat_parser.set_defaults(func=cmd_chat)
    
    # read command
    read_parser = subparsers.add_parser("read", help="Read a PDF file")
    read_parser.add_argument("file", help="Path to PDF file")
    read_parser.add_argument(
        "--pages",
        help="Page range to read (e.g., 1-50 or just 10)"
    )
    read_parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process sequentially (slower but uses less memory)"
    )
    read_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Pages per batch for hybrid processing (default: 10)"
    )
    read_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    read_parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Max characters per chunk (default: 2000)"
    )
    read_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    read_parser.add_argument(
        "--save",
        action="store_true",
        help="Save after reading"
    )
    read_parser.set_defaults(func=cmd_read)
    
    # know command
    know_parser = subparsers.add_parser("know", help="Query what the mind knows")
    know_parser.add_argument(
        "topic",
        nargs="?",
        help="Topic to query (optional, shows all if not specified)"
    )
    know_parser.set_defaults(func=cmd_know)
    
    # viz command
    viz_parser = subparsers.add_parser("viz", help="Visualize knowledge graph")
    viz_parser.add_argument(
        "--html",
        action="store_true",
        help="Generate interactive HTML visualization (requires pyvis)"
    )
    viz_parser.add_argument(
        "--output", "-o",
        help="Output path for HTML file (default: ./{name}_graph.html)"
    )
    viz_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open the HTML file in browser"
    )
    viz_parser.add_argument(
        "--clusters",
        action="store_true",
        help="Cluster view only (no edges)"
    )
    viz_parser.set_defaults(func=cmd_viz)
    
    # forget command
    forget_parser = subparsers.add_parser("forget", help="Forget a topic")
    forget_parser.add_argument("topic", help="Topic to forget")
    forget_parser.set_defaults(func=cmd_forget)
    
    # reflect command
    reflect_parser = subparsers.add_parser("reflect", help="Reflect on knowledge")
    reflect_parser.set_defaults(func=cmd_reflect)
    
    # revise command
    revise_parser = subparsers.add_parser("revise", help="Run maintenance pass on knowledge")
    revise_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    revise_parser.add_argument(
        "--no-save",
        action="store_true",
        dest="no_save",
        help="Don't save after revision (default: save)"
    )
    revise_parser.set_defaults(func=cmd_revise)

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Deep audit to find misplaced nodes")
    audit_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    audit_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply suggested relocations (default: dry run)"
    )
    audit_parser.add_argument(
        "--no-save",
        action="store_true",
        dest="no_save",
        help="Don't save after audit (default: save if --apply)"
    )
    audit_parser.set_defaults(func=cmd_audit)

    # cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Cluster nodes by embedding similarity")
    cluster_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    cluster_parser.add_argument(
        "--min-size",
        type=int,
        default=3,
        dest="min_size",
        help="Minimum cluster size for HDBSCAN (default: 3)"
    )
    cluster_parser.add_argument(
        "-k", "--n-clusters",
        type=int,
        default=None,
        dest="n_clusters",
        help="Use KMeans with fixed k (default: auto-detect optimal k)"
    )
    cluster_parser.add_argument(
        "--hdbscan",
        action="store_true",
        help="Use HDBSCAN instead of KMeans (auto-detects clusters)"
    )
    cluster_parser.set_defaults(func=cmd_cluster)

    # wonder command
    wonder_parser = subparsers.add_parser("wonder", help="What am I curious about?")
    wonder_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Number of goals to show (default: 5)"
    )
    wonder_parser.set_defaults(func=cmd_wonder)
    
    # explore command
    explore_parser = subparsers.add_parser("explore", help="Investigate top curiosity")
    explore_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    explore_parser.add_argument(
        "--save",
        action="store_true",
        help="Save after exploration"
    )
    explore_parser.set_defaults(func=cmd_explore)
    
    # ponder command
    ponder_parser = subparsers.add_parser("ponder", help="Autonomous exploration")
    ponder_parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=3,
        help="Number of exploration cycles (default: 3)"
    )
    ponder_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    ponder_parser.add_argument(
        "--save",
        action="store_true",
        help="Save after pondering"
    )
    ponder_parser.set_defaults(func=cmd_ponder)

    # research command
    research_parser = subparsers.add_parser("research", help="Research a specific topic")
    research_parser.add_argument(
        "topic",
        help="Topic to research (e.g., 'quantum computing')"
    )
    research_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    research_parser.add_argument(
        "--save",
        action="store_true",
        help="Save after research"
    )
    research_parser.set_defaults(func=cmd_research)

    args = parser.parse_args()
    
    if args.command is None:
        # Default to chat if no command specified
        args.func = cmd_chat
        cmd_chat(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
