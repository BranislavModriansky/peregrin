from shiny import reactive, render, ui
from src.code import LogQueue, get_logger

_log = get_logger(__name__)


# CSS classes applied per level
_LEVEL_CSS = {
    "warn":  "color:#efc015;",
    "error": "color:#ee643a;",
}


def mount_logger():
    """Reactive poll that drains the LogQueue and appends entries to #log-output."""

    # Accumulate rendered entries so expanding the panel shows history
    _history: list[ui.Tag] = []

    @reactive.poll(lambda: LogQueue().count, interval_secs=0.5)
    def _drain_log():
        entries = LogQueue().drain()
        for e in entries:
            style = _LEVEL_CSS.get(e.level, "")
            _history.append(
                ui.tags.span(
                    ui.tags.span(e.timestamp, style="color:#47b90e; margin-right:8px;"),
                    e.message,
                    style=f"display:block;{style}",
                )
            )

    @render.ui
    def log_output_content():
        _drain_log()
        if not _history:
            _log.warning("I See Boats Moving (Fernando Pessoa, transl. by Johnathan Griffin)")
            _log.info("I see boats moving on the sea.")
            _log.info("Their sails, like wings of what I see,")
            _log.info("Bring me a vague inner desire to be")
            _log.info("Who I was without knowing what it was.")
            _log.info("So all recalls my home self, and, because")
            _log.info("It recalls that, what I am aches in me.")
            _log.info("")
            return None
        return ui.TagList(*_history)