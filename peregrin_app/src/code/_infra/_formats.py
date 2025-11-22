from shiny import ui
import pandas as pd

FilenameFormatExample = pd.DataFrame({
    "Filename": [
        "Date#CellType3#Treatment7#abc.csv", 
        "A#49;10.7728#20;5.2821#.csv", 
        "#ArkadyI.S_hotel#Bern.Rieux#zápalky.csv"
    ],
    "Condition": ["CellType3", "49;10.7728", "ArkadyI.S_hotel"],
    "Replicate": ["Treatment7", "20;5.2821", "Bern.Rieux"]
})