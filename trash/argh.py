from shiny import App, ui, reactive, render

N = 6  # however many buttons you currently have (can change)

app_ui = ui.page_fluid(
    ui.h3("Click any button"),
    ui.div(
        *[
            ui.input_action_button(
                f"pass_selected_{i}",
                f"Select {i}",
                # send a single value to the server with the index that was clicked
                onclick=f"Shiny.setInputValue('pass_selected', {i}, {{priority: 'event'}});"
            )
            for i in range(N)
        ],
        class_="space-x-2"
    ),
    ui.output_text("last_clicked"),
)

def server(input, output, session):
    # React to *any* of the buttons via one input
    @reactive.Effect
    @reactive.event(input.pass_selected)
    def _on_any_click():
        i = input.pass_selected()
        print("Clicked:", i)  # do your work here

    @output
    @render.text
    def last_clicked():
        val = input.pass_selected()
        return "Nothing yet" if val is None else f"Last clicked index: {val}"

app = App(app_ui, server)
