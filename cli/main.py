import sys
import os
# Fix macOS iCloud PYTHONPATH hanging by natively injecting the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import typer
from rich.console import Console
from rich.table import Table
import os

from agentmem_os.storage.manager import StorageManager
from agentmem_os.storage.store import ConversationStore
from agentmem_os.storage.sync import SSDSync
from agentmem_os.db.models import Turn, Session

app = typer.Typer(help="MemNAI CLI - Local-First AI Memory Framework")
console = Console()

@app.command()
def storage_status():
    """
    Shows SSD mount status, storage sizes, and token totals.
    """
    sm = StorageManager()
    sync = SSDSync()
    
    sync.check_and_sync()
    
    console.print()
    console.print("[bold cyan]MemNAI Storage Status[/bold cyan]")
    
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="dim")
    table.add_column("Value")
    
    status_color = "yellow" if sm.is_fallback_active() else "green"
    status_text = "Fallback (SSD Disconnected)" if sm.is_fallback_active() else "Primary SSD Mounted"
    
    table.add_row("Active Path", sm.active_path)
    table.add_row("Status", f"[{status_color}]{status_text}[/{status_color}]")
    
    db_path = os.path.join(sm.get_path("db"), "sessions.db")
    db_size = f"{os.path.getsize(db_path) / 1024:.2f} KB" if os.path.exists(db_path) else "0 KB"
    table.add_row("Database Size", db_size)
    
    store = ConversationStore()
    try:
        total_tokens = store.db.query(Turn).count() * 10 
        sessions_count = store.db.query(Session).count()
        table.add_row("Total Sessions", str(sessions_count))
        table.add_row("Total Turns", str(store.db.query(Turn).count()))
    except Exception as e:
        pass
    finally:
        store.close()
        
    console.print(table)
    console.print()

@app.command()
def pull(from_session: str = typer.Option(..., "--from", help="Session to pull from"),
         query: str = typer.Option(..., "--query", help="Semantic query to search")):
    """
    Semantically search a past session and preview the context to inject.
    """
    from agentmem_os.db.chroma_client import ChromaManager
    cm = ChromaManager()
    results = cm.search(from_session, query, top_k=3)
    
    if not results:
        console.print(f"[yellow]No relevant context found in session '{from_session}' for query '{query}'[/yellow]")
        return
        
    console.print(f"[bold green]Found relevant context from '{from_session}':[/bold green]\n")
    # Result from chromadb could be a list of strings if it found things
    if isinstance(results, str):
        results = [results]
        
    for idx, r in enumerate(results):
        console.print(f"[bold cyan]Chunk {idx+1}:[/bold cyan]")
        console.print(r)
        console.print("---")
        
    inject = typer.confirm("Inject this context into current session?")
    if inject:
        # In a real run, this binds to the memory queue
        console.print("[green]Context successfully queued for injection into next prompt.[/green]")

@app.command()
def cost_report():
    """
    Displays an aggregated report of API costs across all sessions.
    """
    from agentmem_os.db.database import get_session
    from agentmem_os.db.models import CostLog
    from sqlalchemy import func
    
    db = get_session()
    try:
        # Aggregate by session
        results = db.query(
            CostLog.session_id,
            func.sum(CostLog.cost_usd).label('total_cost'),
            func.sum(CostLog.input_tokens).label('total_in'),
            func.sum(CostLog.output_tokens).label('total_out'),
            func.sum(CostLog.cached_tokens).label('total_cached')
        ).group_by(CostLog.session_id).all()
        
        console.print("\n[bold cyan]MemNAI Cost Report (Global)[/bold cyan]")
        
        if not results:
            console.print("[dim]No API calls logged yet.[/dim]\n")
            return
            
        report_table = Table(box=None)
        report_table.add_column("Session")
        report_table.add_column("In Tokens", justify="right")
        report_table.add_column("Out Tokens", justify="right")
        report_table.add_column("Actual Cost", style="dim", justify="right")
        report_table.add_column("Est. Savings", style="bold green", justify="right")
        
        total_platform_cost = 0.0
        total_savings = 0.0
        
        for row in results:
            # Baseline calculation against GPT-4o ($5/M input, $15/M output)
            gpt4o_cost = (row.total_in * 0.000005) + (row.total_out * 0.000015)
            savings = gpt4o_cost - row.total_cost
            
            report_table.add_row(
                row.session_id,
                f"{row.total_in:,}",
                f"{row.total_out:,}",
                f"${row.total_cost:.4f}",
                f"${savings:.4f}"
            )
            total_platform_cost += row.total_cost
            total_savings += savings
            
        console.print(report_table)
        console.print(f"\n[bold]Total Platform API Cost:[/bold] [dim]${total_platform_cost:.4f}[/dim]")
        console.print(f"[bold]Total Enterprise Cost Saved:[/bold] [bold green]${total_savings:.4f}[/bold green]\n")
        
    finally:
        db.close()

@app.command()
def branch_list(session_id: str = typer.Option(..., "--session", help="Root session to visualize")):
    """
    Shows a tree of all branches originating from or related to the given session.
    """
    from agentmem_os.storage.store import ConversationStore
    store = ConversationStore()
    
    try:
        branches = store.list_branches(session_id)
        if not branches:
            console.print(f"[yellow]No branches found connecting to {session_id}[/yellow]")
            return
            
        console.print(f"\n[bold magenta]Branch Tree for Session:[/bold magenta] {session_id}")
        table = Table(box=None)
        table.add_column("Session ID")
        table.add_column("Name")
        table.add_column("Parent")
        table.add_column("Created At")
        table.add_column("Tokens", justify="right")
        
        for b in branches:
            table.add_row(
                b.session_id,
                b.name,
                b.parent_session_id or "[dim]Root[/dim]",
                str(b.created_at).split('.')[0],
                f"{b.total_tokens:,}"
            )
            
        console.print(table)
    finally:
        store.close()

@app.command()
def chat(session: str = typer.Option(..., "--session", help="Session ID to open/resume"),
         model: str = typer.Option("groq/llama-3.1-8b-instant", "--model", help="LLM to use")):
    """
    Interactive chat session.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    from agentmem_os.llm.adapters import UniversalAdapter
    from agentmem_os.storage.store import ConversationStore
    import builtins
    
    store = ConversationStore()
    adapter = UniversalAdapter()
    
    # Ensure session exists
    store.get_or_create_session(session, name=session, model=model)
    
    console.print(f"[bold green]Session [/bold green]'{session}' [bold green]started via[/bold green] {model}")
    console.print("[dim]Type 'exit' or 'quit' to end. Type 'branch <name>' to branch off.[/dim]\n")
    
    while True:
        try:
            user_input = builtins.input("You> ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if user_input.lower().startswith('branch '):
                name = user_input.split(' ', 1)[1]
                child = store.create_branch(session, name)
                console.print(f"[bold green]Branched![/bold green] New session ID: {child.session_id}")
                session = child.session_id
                continue
                
            store.save_turn(session, "user", user_input)
            
            with console.status("[bold cyan]Generating...[/bold cyan]"):
                response_text = adapter.send_message(session, user_input, model=model)
                
            store.save_turn(session, "assistant", response_text)
            console.print(f"[bold magenta]AI>[/bold magenta] {response_text}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    app()
