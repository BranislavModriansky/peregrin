from shiny import ui


navpanel_about = ui.nav_panel(
    '&',
    ui.br(),
    ui.markdown(
        """ 
            ### About Peregrin
            ##### A tool designed for processing and interpreting tracking data, offering a user-friendly interface built with <a href="https://shiny.posit.co/py/" target="_blank" rel="noopener noreferrer">Py-Shiny</a>.
            *Import raw or processed data.* <br>
            *Explore them by applying filters and generating insightful visualizations.* <br>
            *Export results.* <br>

            ___
            ### References
        """
    ),
    ui.row(
        ui.column(6,
            ui.tags.details(
                ui.tags.summary(
                    "Qualitative colormaps: ",
                    ui.tags.a(
                        "https://matplotlib.org/stable/tutorials/colors/colormaps.html",
                        href="https://matplotlib.org/stable/tutorials/colors/colormaps.html",
                        target="_blank",
                        rel="noopener noreferrer"
                    )
                ),
                ui.div(
                    ui.output_image("colormap_showcase")
                ) 
            )
        ),
        ui.column(6,
            ui.tags.details(
                ui.tags.summary(
                    "Colors: ",
                    ui.tags.a(
                        "https://xkcd.com/color/rgb/",
                        href="https://xkcd.com/color/rgb/",
                        target="_blank",
                        rel="noopener noreferrer"
                    )   
                ),
                ui.div(
                    ui.output_image("colors_showcase")
                ) 
            )
        )
    )
)