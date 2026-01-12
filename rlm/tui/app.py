"""
RLM TUI Application - Toad-style Terminal UI

A full-screen terminal interface for the Recursive Language Model.
Features:
- Dataset selection sidebar
- Query presets
- Interactive chat
- Command palette (Ctrl+P)
- Real-time status updates
"""

import os

from dotenv import load_dotenv
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Input, Markdown, Static
from textual.worker import Worker, WorkerState

load_dotenv(dotenv_path=".env.local")

from rlm.data import TEST_QUERIES  # noqa: E402

# ============================================================================
# Custom Messages
# ============================================================================


class StepUpdate(Message):
    """Message to update a thinking step from worker thread."""

    def __init__(
        self,
        step_num: int,
        title: str,
        content: str,
        is_complete: bool,
        duration: float = 0.0,
        description: str = None,
    ):
        super().__init__()
        self.step_num = step_num
        self.title = title
        self.content = content
        self.is_complete = is_complete
        self.duration = duration  # Elapsed time in seconds
        self.description = description  # Human-readable intent/purpose


class SubStepUpdate(Message):
    """Message to update a batch sub-step from worker thread."""

    def __init__(
        self, step_num: int, sub_id: str, title: str, content: str, is_complete: bool
    ):
        super().__init__()
        self.step_num = step_num  # Parent step number
        self.sub_id = sub_id  # e.g., "Batch[1]" or "Sub-LLM"
        self.title = title
        self.content = content
        self.is_complete = is_complete


# ============================================================================
# Custom Widgets
# ============================================================================


class ChatMessage(Static):
    """A single chat message with visual styling."""

    def __init__(self, role: str, content: str, **kwargs):
        # Pass empty string to Static to avoid rendering duplicate content
        super().__init__("", **kwargs)
        self.role = role
        self._message_content = (
            content  # Use different name to avoid Static.content collision
        )

    def compose(self) -> ComposeResult:
        if self.role == "user":
            # User message: cyan left border, bold
            yield Static(
                f"[bold cyan]💬 You[/bold cyan]\n{self._message_content}",
                classes="user-message",
            )
        elif self.role == "assistant":
            # Assistant answer: green border, markdown for proper formatting
            yield Static("[bold green]🤖 Answer[/bold green]", classes="answer-header")
            yield Markdown(self._message_content, classes="assistant-message")
        elif self.role == "status":
            yield Static(
                f"[dim]{self._message_content}[/dim]", classes="status-message"
            )
        elif self.role == "error":
            yield Static(
                f"[bold red]❌ Error:[/bold red] {self._message_content}",
                classes="error-message",
            )
        elif self.role == "thinking-start":
            # Visual separator before thinking section
            yield Static(
                "[bold yellow]🧠 Thinking...[/bold yellow]", classes="thinking-header"
            )


class ThinkingContainer(Static):
    """Container for thinking steps with visual separation."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.steps: list = []

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold yellow]🧠 Thinking[/bold yellow]", classes="thinking-header"
        )


class ThinkingStep(Container):
    """A collapsible thinking step with scrollable content."""

    # Spinner frames for animation
    SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    DEFAULT_CSS = """
    ThinkingStep {
        width: 100%;
        height: auto;
        max-height: 25;
        margin-bottom: 0;
        padding: 0 1;
        border-left: solid $warning;
        margin-left: 1;
    }
    
    ThinkingStep .step-header {
        width: 100%;
        height: auto;
    }
    
    ThinkingStep .step-content-scroll {
        width: 100%;
        max-height: 20;
        height: auto;
        background: $surface-darken-2;
        border: solid $primary-darken-2;
        padding: 1;
        margin-left: 2;
    }
    
    ThinkingStep .step-content-scroll:focus-within {
        border: solid $primary;
    }
    
    ThinkingStep .sub-step-item {
        margin-left: 2;
    }
    """

    def __init__(self, step_num: int, title: str, **kwargs):
        import time

        super().__init__(**kwargs)
        self.step_num = step_num
        self._title = title
        self._content = ""
        self._description = None
        self._is_complete = False
        self._is_expanded = True
        self._spinner_idx = 0
        self._start_time = time.time()
        self._duration = 0.0
        self._sub_steps: dict[str, dict] = {}

    def compose(self) -> ComposeResult:
        """Compose the step with header and collapsible content."""
        yield Static(self._build_header(), id="step-header", classes="step-header")

    def _get_spinner(self) -> str:
        frame = self.SPINNERS[self._spinner_idx % len(self.SPINNERS)]
        self._spinner_idx += 1
        return frame

    def _build_header(self) -> str:
        """Build the header text with status, title, duration."""
        import time

        if self._is_complete:
            elapsed = self._duration
        else:
            elapsed = time.time() - self._start_time
        time_str = f"[dim][{elapsed:.1f}s][/dim]" if elapsed > 0.5 else ""

        if self._is_complete:
            arrow = "▼" if self._is_expanded else "▶"
            sub_count = len(self._sub_steps)
            suffix = f" [dim]({sub_count} batch)[/dim]" if sub_count > 0 else ""
            return f"[green]{arrow}[/green] [bold green]✓ {self._title}[/bold green] {time_str}{suffix}"
        else:
            spinner = self._get_spinner()
            # Show sub-step summary if active
            active_subs = [s for s in self._sub_steps.values() if not s["is_complete"]]
            sub_info = (
                f" [dim]({len(self._sub_steps)} batch, {len(active_subs)} active)[/dim]"
                if self._sub_steps
                else ""
            )
            return f"[cyan]{spinner}[/cyan] [bold cyan]{self._title}[/bold cyan] {time_str}{sub_info}"

    def _build_content(self) -> str:
        """Build the full scrollable content."""
        lines = []

        # Main content
        if self._content:
            for line in self._content.split("\n"):
                if line.startswith("Response:"):
                    lines.append(f"[bold magenta]📝 {line}[/bold magenta]")
                elif line.startswith("Code:"):
                    lines.append(f"[bold cyan]💻 {line}[/bold cyan]")
                elif line.startswith("Output:"):
                    lines.append(f"[bold green]📤 {line}[/bold green]")
                elif line.strip().startswith("```"):
                    lines.append(f"[magenta]{line}[/magenta]")
                elif line.strip():
                    lines.append(f"[dim]{line}[/dim]")

        # Sub-steps
        for sub_id, sub_info in self._sub_steps.items():
            if sub_info["is_complete"]:
                content_preview = (
                    sub_info["content"][:100].replace("\n", " ")
                    if sub_info["content"]
                    else ""
                )
                lines.append(
                    f"  [green]✓[/green] [dim]{sub_id}:[/dim] {content_preview}..."
                )
            else:
                spinner = self._get_spinner()
                lines.append(
                    f"  [cyan]{spinner}[/cyan] [yellow]{sub_id}:[/yellow] {sub_info['title']}"
                )

        return "\n".join(lines)

    def update_sub_step(self, sub_id: str, title: str, content: str, is_complete: bool):
        """Update or add a sub-step."""
        if sub_id not in self._sub_steps:
            self._sub_steps[sub_id] = {
                "title": title,
                "content": content,
                "is_complete": is_complete,
            }
        else:
            self._sub_steps[sub_id].update(
                {"title": title, "content": content, "is_complete": is_complete}
            )
        self._refresh_display()

    def update_step(
        self,
        title: str,
        content: str,
        is_complete: bool,
        duration: float = 0.0,
        description: str = None,
    ):
        """Update step content and status."""
        import time

        self._title = title
        self._content = content
        was_complete = self._is_complete
        self._is_complete = is_complete

        if is_complete and not was_complete:
            self._duration = time.time() - self._start_time
            self._is_expanded = False  # Auto-collapse on completion
        elif duration > 0:
            self._duration = duration

        if description:
            self._description = description

        self._refresh_display()

    def _refresh_display(self):
        """Refresh the display based on current state."""
        try:
            header = self.query_one("#step-header", Static)
            header.update(self._build_header())

            # Handle content area - determine if should be shown
            should_show_content = (
                not self._is_complete
            ) or (  # Always show for active steps
                self._is_complete and self._is_expanded
            )  # Show for completed+expanded

            try:
                content_scroll = self.query_one("#step-content-scroll", VerticalScroll)
                content_scroll.display = should_show_content

                if should_show_content:
                    content_static = content_scroll.query_one(
                        "#step-content-static", Static
                    )
                    if not self._is_complete:
                        # Active steps: show last few lines
                        lines = self._build_content().split("\n")
                        recent = (
                            "\n".join(lines[-8:])
                            if len(lines) > 8
                            else "\n".join(lines)
                        )
                        content_static.update(recent)
                    else:
                        # Complete+expanded: show all content
                        content_static.update(self._build_content())
            except Exception:
                # Content area doesn't exist yet, create it if needed
                if should_show_content:
                    content_scroll = VerticalScroll(
                        Static(self._build_content(), id="step-content-static"),
                        id="step-content-scroll",
                        classes="step-content-scroll",
                    )
                    self.mount(content_scroll)
        except Exception:
            pass

    def toggle_expand(self):
        """Toggle expanded state for completed steps."""
        if self._is_complete:
            self._is_expanded = not self._is_expanded
            self._refresh_display()

    def on_click(self) -> None:
        """Handle click to toggle expand/collapse."""
        self.toggle_expand()


class StatsBar(Static):
    """Statistics bar showing tokens, cost, duration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stats = {}

    def update_stats(self, stats: dict):
        self.stats = stats
        parts = []
        if "duration" in stats:
            parts.append(f"⏱️ {stats['duration']:.1f}s")
        if "tokens" in stats:
            parts.append(f"📊 {stats['tokens']:,} tokens")
        if "cost" in stats:
            parts.append(f"💰 ${stats['cost']:.4f}")
        if "llm_calls" in stats:
            parts.append(f"🔄 {stats['llm_calls']} calls")

        self.update("  │  ".join(parts) if parts else "")


# ============================================================================
# Command Palette Commands
# ============================================================================

# Default models (can be overridden by RLM_AVAILABLE_MODELS env var)
DEFAULT_MODELS = [
    ("gemini-3-flash-preview", "Gemini 3 Flash (추천)"),
    ("gemini-3-pro-preview", "Gemini 3.0 Pro"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite (빠름)"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
]


def get_available_models():
    """Get available models from env var or use defaults."""
    env_models = os.getenv("RLM_AVAILABLE_MODELS")
    if env_models:
        # Parse comma-separated: "model1,model2,model3"
        models = []
        for m in env_models.split(","):
            m = m.strip()
            if m:
                models.append((m, m))  # (id, display_name)
        if models:
            return models
    return DEFAULT_MODELS


def get_default_model():
    """Get default model from GEMINI_MODEL_NAME or use first available."""
    env_default = os.getenv("GEMINI_MODEL_NAME")
    if env_default:
        return env_default.strip()
    models = get_available_models()
    return models[0][0] if models else "gemini-3-flash-preview"


# For backwards compatibility
AVAILABLE_MODELS = get_available_models()


class RLMCommands(Provider):
    """Command palette commands for RLM."""

    async def search(self, query: str) -> Hits:
        app = self.app

        # Import get_all_datasets dynamically
        from rlm.data import get_all_datasets

        all_datasets = get_all_datasets()

        # Dataset commands
        for key, info in all_datasets.items():
            name = f"Switch to {info['name']}"
            if query.lower() in name.lower():
                yield Hit(
                    score=1,
                    match_display=name,
                    command=lambda k=key: app.switch_dataset(k),
                    help=f"Load project {key}",
                )

        # Model selection commands
        for model_id, model_name in AVAILABLE_MODELS:
            name = f"Model: {model_name}"
            if query.lower() in name.lower() or query.lower() in model_id.lower():
                yield Hit(
                    score=2,
                    match_display=name,
                    command=lambda m=model_id: app.switch_model(m),
                    help=f"Switch to {model_id}",
                )

        # Query commands
        for key, info in TEST_QUERIES.items():
            if info["query"]:
                name = f"Run: {info['name']}"
                if query.lower() in name.lower():
                    yield Hit(
                        score=1,
                        match_display=name,
                        command=lambda q=info["query"]: app.run_query(q),
                        help=info["description"],
                    )

        # Other commands
        commands = [
            ("Clear chat", app.clear_chat, "Clear all messages"),
            ("Show help", app.show_help, "Display keyboard shortcuts"),
        ]
        for name, cmd, help_text in commands:
            if query.lower() in name.lower():
                yield Hit(score=1, match_display=name, command=cmd, help=help_text)


# ============================================================================
# Main Application
# ============================================================================


class RLMApp(App):
    """RLM Terminal UI Application."""

    TITLE = "🤖 RLM Agent"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    COMMANDS = {RLMCommands}

    def __init__(self):
        super().__init__()
        self.current_dataset = "1"
        self.project_path = ""  # Path to project folder (dynamic file access)
        self.current_model = get_default_model()  # From env or default
        self._query_running = False
        self.thinking_steps: dict[int, ThinkingStep] = {}
        self._thinking_start_time: float = 0
        self._thinking_timer = None
        self._ignore_step_updates = False  # Flag to ignore late step updates

    def compose(self) -> ComposeResult:
        yield Header()

        # Clean chat-only layout (no sidebar)
        with Vertical(id="chat-area"):
            yield VerticalScroll(id="chat-log")
            yield StatsBar(id="stats-bar")
            yield Input(placeholder="/help 입력 or 질문 입력 (Enter)", id="chat-input")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.query_one("#chat-input", Input).focus()

        # Welcome message
        self.add_message("status", "👋 RLM Agent에 오신 것을 환영합니다!")
        self.add_message(
            "status",
            "💡 data/projects/ 폴더에 PDF, DOCX, TXT 등 문서를 넣고 질문하세요.",
        )
        self.add_message("status", "⌨️  Ctrl+P: 명령어 팔레트 | Ctrl+Q: 종료")

        # Load first available project folder
        from rlm.data import get_all_datasets

        all_datasets = get_all_datasets()
        if all_datasets:
            first_key = list(all_datasets.keys())[0]
            self.switch_dataset(first_key)

    def add_message(self, role: str, content: str):
        """Add a message to the chat log."""
        chat_log = self.query_one("#chat-log", VerticalScroll)
        message = ChatMessage(role, content)
        chat_log.mount(message)
        chat_log.scroll_end()

    def add_or_update_step(
        self,
        step_num: int,
        title: str,
        content: str,
        is_complete: bool,
        duration: float = 0.0,
        description: str = None,
    ):
        """Add or update a thinking step in the chat log."""
        chat_log = self.query_one("#chat-log", VerticalScroll)

        if step_num in self.thinking_steps:
            # Update existing step
            step_widget = self.thinking_steps[step_num]
            step_widget.update_step(title, content, is_complete, duration, description)
        else:
            # Create new step
            step_widget = ThinkingStep(step_num, title, id=f"step-{step_num}")
            step_widget.update_step(title, content, is_complete, duration, description)
            self.thinking_steps[step_num] = step_widget
            chat_log.mount(step_widget)

        chat_log.scroll_end()

    def clear_thinking_steps(self):
        """Clear all thinking steps (before new query)."""
        for step_widget in self.thinking_steps.values():
            step_widget.remove()
        self.thinking_steps.clear()

    def clear_chat(self):
        """Clear all messages."""
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.remove_children()
        self.thinking_steps.clear()
        self.add_message("status", "💬 채팅이 초기화되었습니다.")

    def show_help(self):
        """Show help message."""
        help_text = """
**📋 슬래시 명령어:**
- `/project` - 프로젝트 목록 보기
- `/project <N>` - 프로젝트 선택 (예: /project 1)
- `/model` - 모델 목록 보기
- `/model <name>` - 모델 선택 (예: /model gemini-2.5-flash)
- `/clear` - 채팅 초기화
- `/help` - 도움말

**⌨️ 단축키:**
- `Ctrl+P`: 명령어 팔레트
- `Ctrl+L`: 채팅 초기화
- `Ctrl+Q`: 종료
"""
        self.add_message("assistant", help_text)

    def switch_dataset(self, key: str):
        """Switch to a different project folder (just stores path, no loading)."""
        from rlm.data import get_all_datasets

        all_datasets = get_all_datasets()

        if not all_datasets:
            self.add_message(
                "error",
                "프로젝트 폴더가 없습니다. data/projects/ 에 폴더를 생성하세요.",
            )
            return

        if key not in all_datasets:
            self.add_message("error", f"Unknown dataset: {key}")
            return

        self.current_dataset = key
        info = all_datasets[key]
        self.project_path = info["path"]

        # Just show info, no loading (LLM will explore dynamically)
        self.add_message("status", f"✅ Selected: {info['name']}")
        self.add_message("status", f"📂 Path: {info['path']}")

    def switch_model(self, model_name: str):
        """Switch to a different LLM model."""
        self.current_model = model_name
        self.add_message("status", f"🤖 Model switched to: {model_name}")

    def show_models(self):
        """Show available models."""
        lines = ["**Available models:**"]
        for model_id, model_name in AVAILABLE_MODELS:
            marker = "→" if model_id == self.current_model else " "
            lines.append(f"{marker} `/model {model_id}` - {model_name}")
        lines.append("\nUse Ctrl+P and type 'model' to select, or type `/model <name>`")
        self.add_message("assistant", "\n".join(lines))

    def handle_slash_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        text = text.strip()

        if text == "/model" or text == "/models":
            self.show_models()
            return True

        if text.startswith("/model "):
            model_name = text[7:].strip()
            # Check if valid model
            valid_models = [m[0] for m in AVAILABLE_MODELS]
            if model_name in valid_models:
                self.switch_model(model_name)
            else:
                self.add_message("error", f"Unknown model: {model_name}")
                self.show_models()
            return True

        if text == "/project" or text == "/projects":
            self.show_projects()
            return True

        if text.startswith("/project "):
            project_key = text[9:].strip()
            self.switch_dataset(project_key)
            return True

        if text == "/help":
            self.show_help()
            return True

        if text == "/clear":
            self.clear_chat()
            return True

        return False

    def show_projects(self):
        """Show available projects."""
        from rlm.data import get_all_datasets

        all_datasets = get_all_datasets()

        if not all_datasets:
            self.add_message(
                "error", "프로젝트가 없습니다. data/projects/ 에 폴더를 추가하세요."
            )
            return

        lines = ["**Available projects:**"]
        for key, info in all_datasets.items():
            marker = "→" if info["path"] == self.project_path else " "
            lines.append(f"{marker} `/project {key}` - {info['name']}")
        lines.append("\nUse `/project <number>` to select")
        self.add_message("assistant", "\n".join(lines))

    def run_query(self, query: str):
        """Run a query through RLM."""
        if self._query_running:
            self.add_message("status", "⏳ 이전 쿼리가 아직 실행 중입니다...")
            return

        if not self.project_path:
            self.add_message("error", "먼저 프로젝트 폴더를 선택해주세요.")
            return

        # Clear previous thinking steps
        self.clear_thinking_steps()
        self._ignore_step_updates = False  # Allow step updates for new query

        # Add user message and thinking header
        self.add_message("user", query)
        self.add_message("thinking-start", "")

        # Start thinking timer
        import time

        self._thinking_start_time = time.time()
        self._start_thinking_timer()

        self.run_rlm_worker(query)

    def _start_thinking_timer(self):
        """Start timer to update thinking duration."""
        self._thinking_timer = self.set_interval(0.5, self._update_thinking_timer)

    def _stop_thinking_timer(self):
        """Stop the thinking timer."""
        if self._thinking_timer:
            self._thinking_timer.stop()
            self._thinking_timer = None

    def _update_thinking_timer(self):
        """Update thinking duration and refresh spinners."""
        if not self._query_running:
            self._stop_thinking_timer()
            return

        import time

        _ = int(time.time() - self._thinking_start_time)  # Track elapsed time

        # Update active step spinner
        for step_widget in self.thinking_steps.values():
            if not step_widget._is_complete:
                step_widget._refresh_display()

    def _step_callback(
        self,
        step_num: int,
        title: str,
        content: str,
        is_complete: bool,
        duration: float = 0.0,
        description: str = None,
    ):
        """Callback for RLM to report step progress."""
        # Post message for thread-safe UI update
        self.post_message(
            StepUpdate(step_num, title, content, is_complete, duration, description)
        )

    def _sub_step_callback(
        self, sub_id: str, title: str, content: str, is_complete: bool
    ):
        """Callback for RLM to report sub-step (batch) progress."""
        # Get current step number from agent
        step_num = getattr(self, "_current_step_num", 1)
        self.post_message(SubStepUpdate(step_num, sub_id, title, content, is_complete))

    def on_step_update(self, message: StepUpdate) -> None:
        """Handle step update message from worker thread."""
        # Ignore late step updates after completion
        if self._ignore_step_updates:
            return
        self._current_step_num = message.step_num  # Track for sub-steps
        self.add_or_update_step(
            message.step_num,
            message.title,
            message.content,
            message.is_complete,
            message.duration,
            message.description,
        )

    def on_sub_step_update(self, message: SubStepUpdate) -> None:
        """Handle sub-step (batch) update message from worker thread."""
        if self._ignore_step_updates:
            return
        step_num = message.step_num
        if step_num in self.thinking_steps:
            step_widget = self.thinking_steps[step_num]
            step_widget.update_sub_step(
                message.sub_id, message.title, message.content, message.is_complete
            )

    @work(exclusive=True, thread=True)
    def run_rlm_worker(self, query: str) -> dict:
        """Run RLM in background worker."""
        self._query_running = True
        self.call_from_thread(self._show_loading, True)

        try:
            from rlm.core import RLMAgent

            # Create agent with selected model and callbacks
            agent = RLMAgent(
                model_name=self.current_model,
                step_callback=self._step_callback,
                sub_step_callback=self._sub_step_callback,
            )
            answer = agent.run(self.project_path, query)

            duration = agent.stats.get("end_time", 0) - agent.stats.get("start_time", 0)
            stats = {
                "duration": duration,
                "tokens": agent.stats.get("total_tokens", 0),
                "cost": agent.stats.get("estimated_cost", 0),
                "llm_calls": agent.stats.get("llm_calls", 0),
            }

            return {"answer": answer, "stats": stats}

        except Exception as e:
            return {"error": str(e)}

        finally:
            self._query_running = False
            self.call_from_thread(self._show_loading, False)

    def _show_loading(self, show: bool):
        """Show/hide loading indicator (no-op, using step display instead)."""
        pass  # LoadingIndicator removed, using ThinkingStep display instead

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion."""
        if event.state == WorkerState.SUCCESS:
            # Stop the thinking timer
            self._stop_thinking_timer()

            result = event.worker.result
            if result:
                if "error" in result:
                    self.add_message("error", result["error"])
                else:
                    answer = result.get("answer", "")
                    # Debug: check if answer is empty or looks like a placeholder
                    if not answer or answer == "[LLM response]":
                        self.add_message(
                            "error",
                            f"Empty or placeholder answer received: '{answer[:100]}'",
                        )
                    else:
                        # IMPORTANT: Set flag to ignore any late step updates
                        self._ignore_step_updates = True

                        # Collapse all thinking steps (keep them visible but collapsed)
                        for step_widget in self.thinking_steps.values():
                            step_widget._is_expanded = False
                            step_widget._is_complete = True
                            step_widget._refresh_display()

                        # Add answer
                        self.add_message("assistant", answer)
                    stats_bar = self.query_one("#stats-bar", StatsBar)
                    stats_bar.update_stats(result["stats"])

    @on(Input.Submitted, "#chat-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        query = event.value.strip()
        if query:
            event.input.value = ""

            # Handle slash commands first
            if query.startswith("/"):
                if self.handle_slash_command(query):
                    return

            self.run_query(query)

    # Key bindings
    def action_clear_chat(self):
        self.clear_chat()

    def action_focus_input(self):
        self.query_one("#chat-input", Input).focus()


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Run the TUI application."""
    app = RLMApp()
    app.run()


if __name__ == "__main__":
    main()
