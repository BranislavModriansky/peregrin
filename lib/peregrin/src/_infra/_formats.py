from shiny import ui
import pandas as pd

FilenameFormatExample = pd.DataFrame({
    "Filename": [
        "CellType3#Treatment7#ExperimentNumber4#Date.csv", 
        "A#49;10.7728#20;5.2821#.csv", 
        "#ArkadyI.S_hotel#Bern.Rieux#zápalky.csv"
    ],
    "Condition": ["Treatment7", "49;10.7728", "ArkadyI.S_hotel"],
    "Replicate": ["ExperimentNumber4", "20;5.2821", "Bern.Rieux"]
})