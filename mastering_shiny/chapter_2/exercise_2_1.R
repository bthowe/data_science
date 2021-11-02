library(shiny)

ui <- fluidPage(
  textOutput('ayo')
  # verbatimTextOutput('ayo')
)

server <- function(input, output, session) {
  output$ayo <- renderText('Good morning!')
}

shinyApp(ui, server)