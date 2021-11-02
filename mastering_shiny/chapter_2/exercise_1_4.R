library(shiny)

x <- list(
    alpha_1 = letters[1:8],
    alpha_2 = letters[9:16],
    alpha_3 = letters[17:24]
    )

ui <- fluidPage(
  selectInput('long_list', label='Datasets', choices=x)
)

server <- function(input, output, session) {
  
}

shinyApp(ui, server)