# TODO: make the Level class have subclasses with often used messages

class Level:

    info = "default"
    warning = "warning"
    error = "error"

class Messages:

    _inform = {}

    def __init__(self, message: str):

        self.message = message
        
        self.info = Level.info
        self.warning = Level.warning
        self.error = Level.error

        


    # def _inform(self):
    #     infos = {}
    #     for key in infos:
    #         infos[key]["level"] = self.info
    #     return infos

    # def _warn(self):
    #     warnings = {
    #         "replicate colors": {
    #             "message": "Missing replicates",
    #             "detail": (
    #                 "No technical or biological replicates were detected for the current "
    #                 "sample; downstream statistics may be unreliable."
    #             ),
    #         }
    #     }
    #     for key in warnings:
    #         warnings[key]["level"] = self.warning
    #     return warnings

    # def _err(self):
    #     errors = {}
    #     for key in errors:
    #         errors[key]["level"] = self.error
    #     return errors


    def _messages(self):
        self.messages = {
            **self._inform(),
            **self._warn(),
            **self._err(),
        }

    
    def __call__(self):
        self._messages()
        return self.messages[self.message]



    
    


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

    def Exert(self, key: str, ):
        """
        Use a pre-defined report by key, e.g.:

            Report.Exert("missing replicates")
        """
        # normalize key: "Missing Replicates" -> "missing_replicates"
        normalized = key.strip().lower().replace(" ", "_")
        preset = Level.PRESETS.get(normalized)
        if not preset:
            # you can choose to silently ignore or log instead of raising
            raise KeyError(f"Unknown preset report: {key!r}")

        self.Report(
            preset["level"],
            preset["message"],
            preset.get("detail"),
        )

    def Emit(self):
        return self.cache
    

