library(shiny)

ui <- fluidPage(
  sliderInput(
    'deliver', 
    'When should we deliver?', 
    min = as.POSIXct('2020-09-16'), 
    max = as.POSIXct('2020-09-23'), 
    value = as.POSIXct('2020-09-17'), 
    step=86400,
    timeFormat = '%F'
    ),
)

server <- function(input, output, session) {
  
}

shinyApp(ui, server)