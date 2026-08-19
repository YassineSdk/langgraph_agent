from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

console = Console()


def show_banner():
    from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def show_banner():

    # LEFT COLUMN
    logo = Text()
    logo.append("        ████\n")
    logo.append("      ██    ██\n")
    logo.append("     ██      ██\n")
    logo.append("      ██    ██\n")
    logo.append("        ████\n")

    info = Text()
    info.append("MyAgent CLI\n", style="bold")
    info.append("LangGraph Agent\n")
    info.append("Model: llama-3.1-8b-instant\n")
    info.append("Working directory: ./")

    left = Table.grid(padding=(0, 2))
    left.add_column()
    left.add_column()

    left.add_row(logo, info)

    # RIGHT COLUMN
    tools = Text()

    tools.append("Tools\n\n", style="bold")

    tools.append("● chat\n", style="bold")
    tools.append("  General conversation and Q&A\n\n")

    tools.append("● rag\n", style="bold")
    tools.append("  Retrieve information from documents\n")
    tools.append("  and answer using relevant context\n\n")

    tools.append("● Web search\n", style="bold")
    tools.append("  Performs websearch \n" 
                "   answer using relevant context\n\n")

    # MAIN TWO-COLUMN TABLE
    content = Table.grid(
        padding=(0, 3),
        expand=True
    )

    content.add_column(ratio=1)
    content.add_column(ratio=1)

    # This creates the vertical separator
    content.add_row(left, tools)

    # PANEL
    panel = Panel(
        content,
        border_style="bright_black",
        padding=(1, 2)
    )

    console.print(panel)