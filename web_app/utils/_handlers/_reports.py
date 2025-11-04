from shiny import reactive

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
                "duration": 5
            },
            "warning": {
                "note": "Warning",
                "messages": [],
                "details": [],
                "count": 0,
                "duration": 10
            },
            "error": {
                "note": "Error",
                "messages": [],
                "details": [],
                "count": 0,
                "duration": 15
            }
        }

    def Cleanse(self, level: str):
        # data = {**self.cache}
        if level in self.cache:
            self.cache[level] = {**self.cache[level], "messages": [], "details": [], "count": 0}
            # self.cache = data
        if level == "all":
            for lvl in self.cache.keys():
                self.cache[lvl] = {**self.cache[lvl], "messages": [], "details": [], "count": 0}
        print("Cache after cleanse:", self.cache)
            # self.cache = data

    def Report(self, level: str, entry: str, entry_detail: str = None):
        if level in self.cache:
            msgs = list(self.cache[level].get("messages", []))
            dtl = list(self.cache[level].get("details", []))
            cnt = int(self.cache[level].get("count", 0)) + 1

            msgs.append(f'[{cnt}] {entry}')
            if entry_detail:
                dtl.append(f'[{cnt}] {entry_detail}')
            self.cache[level] = {**self.cache[level], "messages": msgs, "details": dtl, "count": cnt}
            # self.cache = data

    def Emit(self):
        # Return reactively so dependents re-run when cache changes.
        return self.cache