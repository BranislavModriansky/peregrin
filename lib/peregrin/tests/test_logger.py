import inspect
import json
import traceback
from datetime import datetime

import pytest

from src._pckg_log import log
from src._pckg_log import logger as logger_module
from src._pckg_log._pckg_exceptions import PackageError, PackageWarning


def test_log_error_includes_call_site():
    with pytest.raises(PackageError) as error:
        call_line = inspect.currentframe().f_lineno + 1
        log('operation failed', lvl='error')

    assert f'{__file__}:{call_line}' in str(error.value)
    assert any(
        frame.filename == __file__ and frame.lineno == call_line
        for frame in traceback.extract_tb(error.value.__traceback__)
    )


def test_log_warning_points_to_call_site():
    with pytest.warns(PackageWarning) as warning:
        call_line = inspect.currentframe().f_lineno + 1
        log('check this value', lvl='warning')

    assert warning[0].filename == __file__
    assert warning[0].lineno == call_line


def test_log_caches_timestamped_entries_without_printing(tmp_path, monkeypatch, capsys):
    log_file = tmp_path / 'log.json'
    log_file.write_text('', encoding='utf-8')
    monkeypatch.setattr(logger_module, 'LOG_FILE', log_file)

    call_line = inspect.currentframe().f_lineno + 1
    log('saved', lvl='info', raise_=True, request_id=7)
    log('again', lvl='debug', raise_=True)

    entries = json.loads(log_file.read_text(encoding='utf-8'))
    assert len(entries) == 2
    assert entries[0]['message'] == 'saved'
    assert entries[0]['source'] == __file__
    assert entries[0]['line'] == call_line
    assert entries[0]['context'] == {'request_id': 7}
    assert datetime.fromisoformat(entries[0]['timestamp']).tzinfo is not None
    assert capsys.readouterr().out == ''


def test_log_info_prints_formatted_context(capsys):
    log('ready', lvl='info', request_id=7)

    output = capsys.readouterr().out
    assert 'INFO' in output
    assert 'ready' in output
    assert 'context={"request_id": 7}' in output
