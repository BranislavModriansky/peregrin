import asyncio
import shiny.ui as ui
from shiny import reactive

def mount_loaders(input, output, session, S):

    # _ _ _ _ RUN PEREGRIN LOADER _ _ _ _
    @reactive.extended_task
    async def loader1():
        with ui.Progress(min=0, max=12) as p:
            p.set(message="Initialization in progress")

            for i in range(0, 10):
                p.set(i, message="Initializing Peregrin...")
                await asyncio.sleep(0.04)
        pass

    @reactive.effect
    @reactive.event(input.run, ignore_none=True)
    def initialize_loader1():
        return loader1()
        
    # _ _ _ _ PROCESSED DATA INPUT LOADER _ _ _ _
    @reactive.extended_task
    async def loader2():
        with ui.Progress(min=0, max=20) as p:
            p.set(message="Initialization in progress")

            for i in range(1, 12):
                p.set(i, message="Initializing Peregrin...")
                await asyncio.sleep(0.12)
        pass

    @reactive.effect
    @reactive.event(input.already_processed_input, ignore_none=True)
    def initialize_loader2():
        return loader2()