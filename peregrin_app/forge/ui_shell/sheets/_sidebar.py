from shiny import ui


sidebar = ui.sidebar(    
    ui.markdown("""  <p>  """),
    ui.markdown(f""" <h2> <b> <p style="margin-left: 10px;">  Trajectory thresholds </p> </b> </h2> """),

    ui.input_action_button(id="append_threshold", label="Add threshold", class_="btn-primary", width="100%", disabled=True),
    ui.input_action_button(id="remove_threshold", label="Remove threshold", class_="btn-primary", width="100%", disabled=True),
    
    ui.output_ui(id="sidebar_accordion_placeholder"),
    ui.input_task_button(id="set_threshold", label="Set threshold", label_busy="Applying...", class_="btn-secondary task-btn", disabled=True),
    ui.markdown("<p style='line-height:0.1;'> <br> </p>"),
    ui.output_ui(id="threshold_info"),
    # ui.download_button(id="download_threshold_info", label="Info SVG", width="100%", _class="space-x-2"),
    ui.output_ui(id="threshold_settings_imex"),

    id="sidebar", open="closed", position="right", width="300px"
)