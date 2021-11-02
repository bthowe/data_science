library(shiny)
library(reactable)

ui <- fluidPage(
  reactableOutput("table")
)
server <- function(input, output, session) {
  output$table <- renderReactable({
    reactable(iris)
  })
  # https://glin.github.io/reactable/
}

shinyApp(ui, server)