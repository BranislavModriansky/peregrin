
# TODO: make the Level class have subclasses with often used messages

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
            msgs = list(self.cache[level].get("messages", []))
            dtl = list(self.cache[level].get("details", []))
            cnt = int(self.cache[level].get("count", 0)) + 1

            msgs.append(f'[{cnt}] {entry}')
            if entry_detail:
                dtl.append(f'[{cnt}] {entry_detail}')
            self.cache[level] = {**self.cache[level], "messages": msgs, "details": dtl, "count": cnt}


    def Emit(self):
        return self.cache