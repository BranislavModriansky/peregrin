from shiny import App, ui, render, Inputs, Outputs, Session

import sys
from pathlib import Path

class ImportMyModules:

    def __init__(self) -> None:
        self.cwd = Path.cwd().resolve()

    def _get_repo_root(self, folder: str) -> Path:
        return next(
            parent for parent in [self.cwd, *self.cwd.parents]
            if (parent / "lib" / "peregrin" / folder).exists()
        )

    def insert_path(self, folder: str) -> None:
        repo_root = self._get_repo_root("src")

        package_root = repo_root / "lib"
        print(f"Adding {package_root} to sys.path")
        sys.path.insert(0, str(package_root))

importer = ImportMyModules()

importer.insert_path("peregrin")


from peregrin.src.data_handler.data_loader import load_data
from peregrin.src.object_interface import create_object

# print(print(Path.cwd().resolve()))

data_object = create_object( load_data(Path.cwd().resolve() / "lib" / "peregrin" / "tests" / "dummy_data" / "tree", {'id': 'track_id', 't': 'time_point', 'x': 'x_coordinate', 'y': 'y_coordinate'}, spatialunits = 'um', timeunits='s') )

tracks_graph = data_object.plot_tracks(color_by='distance').__repr_html__
tracks_graph2 = data_object.plot_tracks(color='tab:blue', align_start=True).__repr_html__
msd_graph = data_object.msd().__repr_html__

app_ui = ui.page_fluid(
    ui.row(
        ui.output_ui("tracks"),
        ui.output_ui("tracks2"),
        ui.output_ui("msd")
    )
)

def server(input: Inputs, output: Outputs, session: Session):

    @render.ui
    def tracks():
        return ui.HTML(
            tracks_graph
        )

    @render.ui
    def tracks2():
        return ui.HTML(
            tracks_graph2
        )

    @render.ui
    def msd():
        return ui.HTML(
            msd_graph
        )


app = App(app_ui, server)
