from typing import *

class Level:

    info = "default"
    warning = "warning"
    error = "error"


class NoticeQueue:

    def __init__(self):
        self.cache = {
            "default": {
                "note": "Info",
                "messages": [],
                "details": [],
                "count": 0,
                "duration": [10, 15]
            },
            "warning": {
                "note": "Warning",
                "messages": [],
                "details": [],
                "count": 0,
                "duration": [15, 25]
            },
            "error": {
                "note": "Error",
                "messages": [],
                "details": [],
                "count": 0,
                "duration": None
            }
        }


    def Cleanse(self, level: str):
        if level in self.cache:
            self.cache[level] = {**self.cache[level], "messages": [], "details": [], "count": 0}
        if level == "all":
            for lvl in self.cache.keys():
                self.cache[lvl] = {**self.cache[lvl], "messages": [], "details": [], "count": 0}


    def Report(self, level: str, entry: str, entry_detail: str = None):
        if level in self.cache:
            message = list(self.cache[level].get("messages", []))
            detail = list(self.cache[level].get("details", []))
            count = int(self.cache[level].get("count", 0)) + 1

            message.append(f'[{count}] {entry}')
            if entry_detail:
                detail.append(f'[{count}] {entry_detail}')
            self.cache[level] = {**self.cache[level], "messages": message, "details": detail, "count": count}


    def Emit(self):
        return self.cache
    


def Reporter(level: str, message: str, details: str = None, noticequeue: Callable[[Level, str, str], None] = None) -> None:

    program_messages = {
        "default":  "PROGRAM INFO:     ",
        "warning":  "PROGRAM WARNING:  ",
        "error":    "PROGRAM ERROR:    "
    }

    if noticequeue is None:
        print("")
        print(f"{program_messages.get(level, 'UNKNOWN')}: {message}")
        if details:
            print(f"{18*' '}--> Details: {details}")
        
    else:
        noticequeue.Report(level, message, details)
