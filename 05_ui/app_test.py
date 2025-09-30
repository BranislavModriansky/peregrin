from shiny import App, ui, render
import numpy as np
import matplotlib.pyplot as plt

# --- UI ---
app_ui = ui.page_fillable(
    ui.head_content(
        ui.tags.style(
            """
            /* A scrollable container that lives inside a card */
            .scroll-panel {
                height: 360px;          /* Visible height of the panel */
                overflow: auto;          /* Enable vertical + horizontal scrollbars */
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                background: white;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            """
        )
    ),
    ui.layout_column_wrap(
        ui.card(
            ui.card_header("Scrollable big plot"),
            ui.div(
                # Make the *plot image* larger than the panel so scrolling kicks in
                ui.output_plot("bigplot", width="1400px", height="1000px"),
                class_="scroll-panel",
            ),
        ),
        width=1,
    ),
)

# --- Server ---
def server(input, output, session):
    @output
    @render.plot
    def bigplot():
        # Create a large/complex plot
        x = np.linspace(0, 200, 50_000)
        y = np.sin(x) + 0.15 * np.random.randn(x.size)

        fig, ax = plt.subplots(figsize=(18, 12), dpi=100)  # big figure
        ax.plot(x, y, linewidth=1)
        ax.set_title("Huge plot inside a scrollable panel")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        return fig

app = App(app_ui, server)
