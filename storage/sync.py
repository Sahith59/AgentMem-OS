import os
import shutil
import logging
from agentmem_os.storage.manager import StorageManager
from rich.console import Console

console = Console()

class SSDSync:
    """
    Handles syncing logic when an external SSD is reconnected.
    Moves turns saved to the fallback path over to the primary path.
    """
    def __init__(self):
        self.storage_manager = StorageManager()

    def check_and_sync(self):
        if self.storage_manager.is_fallback_active():
            return
            
        fallback_db = os.path.join(self.storage_manager.fallback_path, "db", "sessions.db")
        if not os.path.exists(fallback_db):
            return 

        primary_db_dir = self.storage_manager.get_path("db")
        primary_db = os.path.join(primary_db_dir, "sessions.db")
        
        console.print("[yellow]Notice: Data exists in your fallback path from a previous offline session.[/yellow]")
        
        if not os.path.exists(primary_db) or os.path.getsize(primary_db) < 10000:
            console.print("[green]Syncing fallback data to primary SSD...[/green]")
            shutil.copy2(fallback_db, primary_db)
            console.print("[green]Sync complete.[/green]")
        else:
            console.print("[yellow]Manual merge required. The primary SSD already contains data.[/yellow]")
