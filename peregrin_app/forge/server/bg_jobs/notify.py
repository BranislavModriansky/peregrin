import time
from operator import eq
from shiny import reactive, ui, req

from src.code import Level

def mount_notifier(noticequeue):

    """
    Polls the noticequeue every 2 seconds and shows notifications.
    """

    # @reactive.Calc
    def show_notifications():
        """Helper function to display notifications from data"""
        data = noticequeue.Emit()
        if not data:
            return

        # default
        if data["default"]["count"] > 0:
            ui.notification_show(
                ui=ui.tags.div(
                    ui.tags.strong(data["default"]["note"] + ":\n"),
                    ui.tags.pre(
                        [
                            "".join(data["default"]["messages"]),
                            [
                                "\n",
                                ui.tags.details("\n".join(data["default"]["details"]))
                            ] if data["default"]["details"] else ""
                        ],
                        style="border: none; background: none; white-space: pre-wrap; word-wrap: break-word;",
                    ),
                ),
                type="default",
                duration=data["default"]["duration"][0] if data["default"]["details"] else data["default"]["duration"][1],
            )
            noticequeue.Cleanse(Level.info)

        # warning
        if data["warning"]["count"] > 0:
            ui.notification_show(
                ui=ui.tags.div(
                    ui.tags.strong(data["warning"]["note"] + ":"),
                    ui.tags.pre(
                        [
                            "\n".join(data["warning"]["messages"]),
                            [
                                "\n", 
                                ui.tags.details("\n".join(data["warning"]["details"]))
                            ] if data["warning"]["details"] else ""
                        ],
                        style="border: none; background: none; white-space: pre-wrap; break-word;",
                    ),
                ),
                type="warning",
                duration=data["warning"]["duration"][0] if data["warning"]["details"] else data["warning"]["duration"][1],
            )
            noticequeue.Cleanse(Level.warning)

        # error
        if data["error"]["count"] > 0:
            ui.notification_show(
                ui=ui.tags.div(
                    ui.tags.strong(data["error"]["note"] + ":"),
                    ui.tags.pre(
                        [
                            "\n".join(data["error"]["messages"]),
                            [
                                "\n",
                                ui.tags.details("\n".join(data["error"]["details"]))
                            ] if data["error"]["details"] else ""
                        ],
                        style="border: none; background: none; white-space: pre-wrap; word-wrap: break-word;",
                    ),
                ),
                type="error",
                duration=data["error"]["duration"],
            )
            noticequeue.Cleanse(Level.error)

        
    @reactive.poll(show_notifications, interval_secs=1.5)
    def read_queue():
        """Reads data from the noticequeue"""
        show_notifications()
        return noticequeue.Emit()
