import io
import asyncio
import warnings

from datetime import date
import pandas as pd

from shiny import render, reactive, ui, req
from src.code import MSD




def mount_plot_msd(input, output, session, S, noticequeue):

    @ui.bind_task_button(button_id="generate_msd")
    @reactive.extended_task
    async def output_plot_msd(
        data,
        conditions,
        repliacates,
        group_replicates,
        c_mode,

    ):
        pass