from shiny import Inputs, Outputs, Session

from utils import NoticeQueue

from .state import build_state
from .data.input import mount_data_input
from .data.display import mount_data_display
from .data.labeling import mount_data_labeling
from .filters.threshold1d.build_thresholds import mount_thresholds_build
from .filters.threshold1d.calc_thresholds import mount_thresholds_calc
from .filters.threshold1d.export_info import mount_thresholds_info_export

from .viz.tracks import MountTracks
from .viz.superplots import mount_superplots

from .bg_jobs.loaders import mount_loaders
from .bg_jobs.bckgrnd_tasks import mount_tasks

from server.bg_jobs.notify import mount_notifier

    




def Server(input: Inputs, output: Outputs, session: Session):
    
    S = build_state()
    queue = NoticeQueue()
    
    args = (input, output, session, S)
    kwargs = {'queue': queue}
    
    mount_notifier(queue)

    mount_data_input(*args)
    mount_data_display(*args)
    mount_data_labeling(*args)
    mount_thresholds_build(*args)
    mount_thresholds_calc(*args)
    mount_thresholds_info_export(*args)


    MountTracks.realistic_reconstruction(*args)
    MountTracks.polar_reconstruction(*args)
    MountTracks.animated_reconstruction(*args)
    MountTracks.lut_map(*args)
    mount_superplots(*args, **kwargs)


    mount_loaders(*args)
    mount_tasks(S)

    # session.on_ended(queue.Cleanse('all'))

    



