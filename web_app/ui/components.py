import shiny.ui as ui
import shiny_sortable as sortable

def make_sortable_ui(inputID: str, items: list[str]):
    
    # _ _  CUSTOM SORTABLE UI COMPONENT _ _

    @sortable.make(updatable=True)
    def ladder(inputID: str = inputID, items: list[str] = items):
        lis = [
            ui.tags.li(label, **{"data-id": label}, class_="p-2 mb-2 border rounded")
            for label in items
        ]
        return ui.tags.ul(*lis, id=inputID)
