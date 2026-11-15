import inspect
import json
from warnings import warn
import time
from pathlib import Path
from typing import Literal
# from dataclasses import classmethod

from ._pckg_exceptions import PackageError, PackageWarning


LOG_FILE = Path(__file__).with_name('log.json')



class TerminalFormatter:

    # Source - https://stackoverflow.com/a/287944
    # Posted by joeld, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-11-15, License - CC BY-SA 4.0
    BLACK   = '\033[30m'
    GREY    = '\033[90m'
    PURPLE  = '\033[95m'
    BLUE    = '\033[94m'
    DIMBLUE = '\033[34m'
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    DIMGREEN = '\033[32m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    ITALIC  = '\033[3m'
    BOLD    = '\033[1m'
    ENDC    = '\033[0m'
    # UNDERLINE = '\033[4m',
    # STRIKE    = '\033[9m',
    # BGGRE     = '\033[100m',
    # BGRED     = '\033[101m',
    # BGGRE     = '\033[102m',
    # BGYEL     = '\033[103m',
    # BGBLU     = '\033[104m',
    # BGPUR     = '\033[105m',
    # BGCYA     = '\033[106m',
    # BGWHI     = '\033[107m',

    def __self__(): pass

    def _black(self, text: str) -> str:
        return f'{self.BLACK}{text}{self.ENDC}'

    def _grey(self, text: str) -> str:
        return f'{self.GREY}{text}{self.ENDC}'
    
    def _purple(self, text: str) -> str:
        return f'{self.PURPLE}{text}{self.ENDC}'

    def _blue(self, text: str) -> str:
        return f'{self.BLUE}{text}{self.ENDC}'

    def _dimblue(self, text: str) -> str:
        return f'{self.DIMBLUE}{text}{self.ENDC}'

    def _cyan(self, text: str) -> str:
        return f'{self.CYAN}{text}{self.ENDC}'

    def _green(self, text: str) -> str:
        return f'{self.GREEN}{text}{self.ENDC}'

    def _dimgreen(self, text: str) -> str:
        return f'{self.DIMGREEN}{text}{self.ENDC}'

    def _yellow(self, text: str) -> str:
        return f'{self.YELLOW}{text}{self.ENDC}'

    def _red(self, text: str) -> str:
        return f'{self.RED}{text}{self.ENDC}'

    def _italic(self, text: str) -> str:
        return f'{self.ITALIC}{text}{self.ENDC}'

    def _bold(self, text: str) -> str:
        return f'{self.BOLD}{text}{self.ENDC}'

    _formats = {
        'black': lambda self, text: self._black(text),
        'grey': lambda self, text: self._grey(text),
        'purple': lambda self, text: self._purple(text),
        'blue': lambda self, text: self._blue(text),
        'dimblue': lambda self, text: self._dimblue(text),
        'cyan': lambda self, text: self._cyan(text),
        'green': lambda self, text: self._green(text),
        'dimgreen': lambda self, text: self._dimgreen(text),
        'yellow': lambda self, text: self._yellow(text),
        'red': lambda self, text: self._red(text),
        'bold': lambda self, text: self._bold(text),
        'italic': lambda self, text: self._italic(text),
    }
    _formats['debug']   = _formats['grey']
    _formats['info']    = _formats['blue']
    _formats['success'] = _formats['green']
    _formats['warning'] = _formats['yellow']
    _formats['success'] = _formats['green']
    _formats['error']   = _formats['red']

    def format_text(self, text: str, *styles: object) -> str:
        for style in styles:
            if style not in self._formats:
                raise ValueError(f'Unknown style: {style}')
            text = self._formats[style](self, text)
        return text

format_text = TerminalFormatter().format_text


def log(
    *msg: object,
    ctx: object = None,
    lvl: Literal['debug', 'info', 'success', 'warning', 'error', 'none'] = 'none',
    raise_: bool = True,
    **kw,
) -> None:

    frame = inspect.currentframe().f_back
    call_source = frame.f_code.co_filename
    call_line = frame.f_lineno
    local_t = time.localtime()

    timestamp = f"{local_t.tm_hour:02}:{local_t.tm_min:02}:{local_t.tm_sec:02}"
    msg_type  = f"[{lvl.upper()}]"
    msg_text  = kw.get('sep', ' ').join(map(str, msg))
    msg_src   = f"{call_source}:{call_line}"
    
    fmt_timestamp = format_text(timestamp, 'blue')
    fmt_msg_type  = format_text(msg_type, lvl, 'bold')
    fmt_msg_text  = format_text(msg_text, 'bold')
    fmt_msg_src   = format_text(msg_src, 'italic')

    if raise_:
        match lvl:
            case 'error':
                if ctx is None:
                    ctx = PackageError
                raise ctx(f'{fmt_msg_text} {fmt_msg_src} {fmt_timestamp}')
            case 'warning':
                if ctx is None:
                    ctx = PackageWarning
                warn(f'{fmt_msg_text} {fmt_timestamp}', ctx, stacklevel=2)
            case _:
                print(f'{fmt_timestamp} {fmt_msg_type} {fmt_msg_text} {fmt_msg_src}')
        return



    message = {
        'timestamp': timestamp,
        'level': lvl,
        'message': msg_text,
        'source': call_source,
        'line': call_line,
    }

    log = LOG_FILE.read_text(encoding='utf-8').strip() if LOG_FILE.exists() else '[]'
    entries = json.loads(log) if log else []
    if not isinstance(entries, list):
        raise ValueError(f'{LOG_FILE} must contain a JSON array')
    entries.insert(0, message)
    if len(entries) > 100:
        entries = entries[:100]
    with LOG_FILE.open('w', encoding='utf-8') as file:
        json.dump(entries, file, ensure_ascii=False, indent=2, default=str)
        file.write('\n')



def clear_log() -> None:
    if LOG_FILE.exists():
        LOG_FILE.write_text('[]', encoding='utf-8')
    else:
        LOG_FILE.touch(exist_ok=True)
