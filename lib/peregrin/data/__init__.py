from ..src.data_handler.directory_reader import make_tree
from ..src.data_handler.data_loader import load_data


b_cells_data = r"https://github.com/BranislavModriansky/peregrin/tree/main/cell-tracking-data/01"

b_cells_tree = make_tree(b_cells_data)

b_cells = load_data(b_cells_tree)



__all__ = ["b_cells"]