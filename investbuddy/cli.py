"""
CLI entry point for InvestBuddy.

Commands:
    ingest          Fetch, parse, chunk, embed, and index all configured filings.
    ask <question>  Ask a question grounded in the indexed filings.
"""

import logging
import click
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
def cli():
    """InvestBuddy — RAG system for SEC 10-K and 10-Q filings."""


@cli.command()
def ingest():
    """Fetch, parse, chunk, embed, and store all configured filings in Chroma."""
    from investbuddy.ingest import run_ingest
    run_ingest()
    click.echo("\nIngest complete.")


@cli.command()
@click.argument("question")
@click.option("--k", default=6, show_default=True, help="Number of chunks to retrieve.")
def ask(question: str, k: int):
    """Ask QUESTION, grounded in the indexed SEC filings."""
    from investbuddy.answer import answer_question

    click.echo(f"\nSearching filings for: {question!r}\n")
    result = answer_question(question, k=k)

    click.echo("─" * 70)
    click.echo(result["answer"])
    click.echo("─" * 70)

    if result["sources"]:
        click.echo(f"\nCited sources ({len(result['sources'])}):")
        for meta in result["sources"]:
            click.echo(
                f"\n  [{meta['chunk_id']}]\n"
                f"    Company : {meta['company']}\n"
                f"    Form    : {meta['form']}  |  Filed: {meta['filing_date']}"
                f"  |  Period: {meta['fiscal_period']}\n"
                f"    Section : {meta['section']}\n"
                f"    URL     : {meta['source_url']}"
            )
    else:
        click.echo("\nNo specific chunks were cited in the answer.")

    click.echo(
        f"\n({len(result['retrieved'])} chunks retrieved, "
        f"{len(result['sources'])} cited)\n"
    )


if __name__ == "__main__":
    cli()
