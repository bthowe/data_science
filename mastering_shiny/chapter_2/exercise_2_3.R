library(shiny)

ui <- fluidPage(
  dataTableOutput("table")
)
server <- function(input, output, session) {
  output$table <- renderDataTable(mtcars, options = list(searching = FALSE, ordering=FALSE, paging=FALSE))
  # https://datatables.net/reference/option/
}

shinyApp(ui, server)