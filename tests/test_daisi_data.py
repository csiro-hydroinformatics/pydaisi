from pathlib import Path
import pandas as pd
import re
import pytest

from timeit import Timer
import time

import numpy as np
from pydaisi import daisi_data

import warnings

source_file = Path(__file__).resolve()
FTEST = source_file.parent


def test_get_sites():
    sites = daisi_data.get_sites()
    assert sites.shape == (201, 28)


def test_get_data():
    df = daisi_data.get_data("410734")
    assert df.shape == (588, 3)


def test_periods():
    periods = daisi_data.Periods()

    for per in ["per1", "per2", "per3"]:
        assert per in periods.periods
        p = periods.get_periodset(per)
        assert hasattr(p, "active")
        assert hasattr(p, "total")

        tot = p.total
        assert hasattr(tot, "start")
        assert hasattr(tot, "end")

        act = p.active
        assert hasattr(act, "start")
        assert hasattr(act, "end")

    p1 = periods.get_periodset("per1").active
    p1e = pd.to_datetime(p1.end)
    assert str(p1e.date()) == "1999-06-01"

    p2 = periods.get_periodset("per2").active
    p2s = pd.to_datetime(p2.start)
    assert str(p2s.date()) == "1999-07-01"

    time = pd.date_range("1990-01-01", "2000-02-02")
    idx = p1.select_index(time)
    assert idx.sum() == 3439



