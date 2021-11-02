library(shiny)

ui <- fluidPage(
  plotOutput("plot", width = "700px", height = '300px')
)
server <- function(input, output, session) {
  output$plot <- renderPlot(plot(1:5), res = 96, alt='Scatter thing')
}
shinyApp(ui, server)
