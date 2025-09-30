from shiny import ui
import pandas as pd


class Format:

    Accordion = """
        #sidebar, #sidebar > *, #sidebar > div, #sidebar .accordion,
        #sidebar .accordion .accordion-header {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        #sidebar .accordion .accordion-item .accordion-header .accordion-button {
            font-size: 0.6em !important;
            /* font-weight: bold !important; */
            padding-left: 10px !important;
            padding-right: 0 !important;
        }
        #sidebar .accordion .accordion-collapse {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        #sidebar .accordion .accordion-body {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            box-sizing: border-box !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        #sidebar .accordion .accordion-item {
            border-left: 0 !important;
            border-right: 0 !important;
        }
    """

    AccordionDraggable = """
        .custom-accordion .accordion-item {
            background-color: #f8f8f8; /* light grey background */
        }
        .custom-accordion .accordion-button {
            background-color: #f8f8f8; /* dalightrk grey for header */
        }
    """

    Link1 = """
        .plain-link {
            color: #006ea9"";        /* blue color */
            text-decoration: none; /* remove underline */
            cursor: pointer;       /* keep it clickable */
            font-style: italic;    /* italic by default */
        }
        .plain-link:hover {
            text-decoration: underline; /* optional subtle hover effect */
        }
    """

    FilenameFormatExample = pd.DataFrame({
        "Filename": [
            "Date#CellType3#Treatment7#abc.csv", 
            "A#49;10.7728#20;5.2821#.csv", 
            "#ArkadyI.S_hotel#Bern.Rieux#zápalky.csv"
        ],
        "Condition": ["CellType3", "49;10.7728", "ArkadyI.S_hotel"],
        "Replicate": ["Treatment7", "20;5.2821", "Bern.Rieux"]
    })