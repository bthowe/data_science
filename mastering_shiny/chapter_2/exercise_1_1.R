library(shiny)

ui <- fluidPage(
  textInput('name', label='Cheese', placeholder='Your name'),
  textOutput('name_output')
)

server <- function(input, output, session) {
  output$name_output <- renderText(input$name)
}

shinyApp(ui, server)