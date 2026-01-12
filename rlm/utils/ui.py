"""
Agentic CLI UI - Rich terminal interface for RLM.

Provides Claude Code / Cursor-style UI with:
- Collapsible steps that fold when complete
- Spinners for active operations
- Real-time streaming output
- Colored panels and status indicators
"""

import time
from contextlib import contextmanager
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()


class StepUI:
    """
    Manages a single step in the agentic workflow.
    Shows spinner while running, collapses to summary when done.
    """

    def __init__(self, title: str, step_number: int):
        self.title = title
        self.step_number = step_number
        self.status = "pending"  # pending, running, success, error
        self.content_lines: list[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.summary: Optional[str] = None

    def start(self):
        self.status = "running"
        self.start_time = time.time()

    def add_content(self, text: str):
        """Add a line of content to the step."""
        self.content_lines.append(text)

    def complete(self, summary: Optional[str] = None):
        """Mark step as complete."""
        self.status = "success"
        self.end_time = time.time()
        self.summary = summary

    def error(self, message: str):
        """Mark step as failed."""
        self.status = "error"
        self.end_time = time.time()
        self.summary = message

    @property
    def duration(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def render_collapsed(self) -> Text:
        """Render collapsed view (just title + duration)."""
        duration_str = f"({self.duration:.1f}s)"

        if self.status == "success":
            icon = "✓"
            style = "green"
        elif self.status == "error":
            icon = "✗"
            style = "red"
        else:
            icon = "○"
            style = "dim"

        text = Text()
        text.append(f"{icon} ", style=style)
        text.append(f"Step {self.step_number}: ", style="bold")
        text.append(self.title, style=style)
        text.append(f" {duration_str}", style="dim")

        if self.summary:
            text.append(f"\n  └─ {self.summary[:80]}", style="dim italic")

        return text

    def render_expanded(self) -> Panel:
        """Render expanded view with spinner and content."""
        # Build content
        content_parts = []

        # Add spinner for running status
        if self.status == "running":
            spinner_text = Text()
            spinner_text.append("⠋ ", style="cyan")
            spinner_text.append("Processing...", style="cyan italic")
            content_parts.append(spinner_text)

        # Add content lines (last 5 for brevity)
        recent_lines = self.content_lines[-5:] if self.content_lines else []
        for line in recent_lines:
            line_text = Text(f"  {line[:100]}", style="dim")
            content_parts.append(line_text)

        if not content_parts:
            content_parts.append(Text("  Initializing...", style="dim italic"))

        # Create panel
        if self.status == "running":
            border_style = "cyan"
        elif self.status == "success":
            border_style = "green"
        elif self.status == "error":
            border_style = "red"
        else:
            border_style = "dim"

        title = f"Step {self.step_number}: {self.title}"

        return Panel(
            Group(*content_parts),
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
        )


class AgenticUI:
    """
    Main UI controller for agentic workflow display.
    Manages multiple steps with live updating.
    """

    def __init__(self, title: str = "RLM Agent"):
        self.title = title
        self.steps: list[StepUI] = []
        self.current_step: Optional[StepUI] = None
        self.live: Optional[Live] = None
        self._enabled = True

    def disable(self):
        """Disable UI (for non-interactive mode)."""
        self._enabled = False

    def enable(self):
        """Enable UI."""
        self._enabled = True

    def _render(self) -> Group:
        """Render the full UI."""
        parts = []

        # Header
        header = Text()
        header.append("🤖 ", style="bold")
        header.append(self.title, style="bold cyan")
        parts.append(header)
        parts.append(Text())  # Empty line

        # Completed steps (collapsed)
        for step in self.steps:
            if step.status in ("success", "error"):
                parts.append(step.render_collapsed())

        # Current step (expanded)
        if self.current_step and self.current_step.status == "running":
            parts.append(Text())  # Spacing
            parts.append(self.current_step.render_expanded())

        return Group(*parts)

    def start(self):
        """Start the live display."""
        if not self._enabled:
            return

        self.live = Live(
            self._render(),
            console=console,
            refresh_per_second=10,
            transient=False,
        )
        self.live.start()

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None

    def update(self):
        """Update the display."""
        if self.live and self._enabled:
            self.live.update(self._render())

    @contextmanager
    def step(self, title: str):
        """
        Context manager for a step.

        Usage:
            with ui.step("Exploring context"):
                ui.log("Found 3 documents")
                # ... do work ...
        """
        step_number = len(self.steps) + 1
        step = StepUI(title, step_number)
        self.steps.append(step)
        self.current_step = step

        step.start()
        self.update()

        try:
            yield step
            step.complete()
        except Exception as e:
            step.error(str(e))
            raise
        finally:
            self.current_step = None
            self.update()

    def log(self, message: str):
        """Add a log message to the current step."""
        if self.current_step:
            self.current_step.add_content(message)
            self.update()
        elif self._enabled:
            console.print(f"  {message}", style="dim")

    def print_final(self, title: str, content: str):
        """Print final result panel - prominent answer display."""
        if self.live:
            self.live.stop()

        console.print()
        console.print()

        # Main answer panel - large and prominent
        if "**" in content or "#" in content or "\n" in content:
            answer_content = Markdown(content)
        else:
            answer_content = Text(content, style="white")

        console.print(
            Panel(
                answer_content,
                title=f"[bold green]📌 {title}[/bold green]",
                title_align="left",
                border_style="green bold",
                padding=(1, 2),
                expand=True,
            )
        )

    def print_stats(self, stats: dict, log_file: str = None):
        """Print statistics in a subtle, compact format."""
        console.print()

        # Build compact stats line
        stat_items = []
        if "duration" in stats:
            stat_items.append(f"⏱️ {stats['duration']:.1f}s")
        if "tokens" in stats:
            in_tok = stats.get("input_tokens", 0)
            out_tok = stats.get("output_tokens", 0)
            if in_tok and out_tok:
                stat_items.append(
                    f"📊 {stats['tokens']:,} tokens (↑{in_tok:,} ↓{out_tok:,})"
                )
            else:
                stat_items.append(f"📊 {stats['tokens']:,} tokens")
        if "cost" in stats:
            stat_items.append(f"💰 ${stats['cost']:.4f}")
        if "llm_calls" in stats:
            stat_items.append(f"🔄 {stats['llm_calls']} calls")

        # Print as subtle footer
        if stat_items:
            stats_text = "  │  ".join(stat_items)
            console.print(
                "[dim]─────────────────────────────────────────────────────────────[/dim]"
            )
            console.print(f"[dim]{stats_text}[/dim]")

        # Log file info - very subtle
        if log_file:
            console.print(f"[dim italic]📁 Log: {log_file}[/dim italic]")

        console.print()

    def print_error(self, message: str):
        """Print error message."""
        console.print()
        console.print(
            Panel(
                Text(message, style="red"),
                title="[bold red]❌ Error[/bold red]",
                border_style="red",
                padding=(0, 1),
            )
        )

    def print_header(self, title: str, subtitle: str = None):
        """Print application header."""
        console.print()
        console.print(f"[bold cyan]{'═' * 60}[/bold cyan]")
        console.print(f"[bold cyan]  🤖 {title}[/bold cyan]")
        if subtitle:
            console.print(f"[dim]  {subtitle}[/dim]")
        console.print(f"[bold cyan]{'═' * 60}[/bold cyan]")
        console.print()


# Singleton instance for easy import
ui = AgenticUI()


# Helper function for simple usage
def with_status(message: str):
    """Simple status context manager."""
    return console.status(message, spinner="dots")


if __name__ == "__main__":
    # Demo
    console.print(Rule("Agentic UI Demo", style="cyan"))
    console.print()

    ui.start()

    with ui.step("Exploring context"):
        ui.log("Loading document...")
        time.sleep(0.5)
        ui.log("Found 3 contracts")
        time.sleep(0.5)
        ui.log("Analyzing structure...")
        time.sleep(0.5)

    with ui.step("Extracting definitions"):
        ui.log("Searching for 'Definitions' section...")
        time.sleep(0.5)
        ui.log("Found at position 12998")
        time.sleep(0.5)
        ui.log("Calling sub-LLM...")
        time.sleep(1)

    with ui.step("Synthesizing answer"):
        ui.log("Combining results...")
        time.sleep(0.5)

    ui.stop()

    # Example final answer (like the user showed)
    ui.print_final(
        "Final Answer",
        """**Contract 1: SYNDICATED LOAN AGREEMENT (ZEMENIK - VTB BANK)**

'Definitions' 섹션에서 정의된 주요 용어 5개:

1. **보증인 (Guarantor)** - 계약의 보증을 담당하는 당사자
2. **질권설정계약 (Pledge Agreement)** - 담보 설정에 관한 계약
3. **채권자 과반수 (Majority of Creditors)** - 75% 이상의 채권자
4. **연결 EBITDA (Consolidated EBITDA)** - 그룹의 연결 세전이익
5. **자회사 (Subsidiary)** - 그룹에 속한 종속 회사""",
    )

    ui.print_stats(
        {
            "duration": 48.21,
            "tokens": 10660,
            "input_tokens": 9192,
            "output_tokens": 1468,
            "cost": 0.0360,
            "llm_calls": 4,
        },
        log_file="logs/rlm_run_20260112_095838.log",
    )
