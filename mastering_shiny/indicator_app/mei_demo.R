library(shiny)
library(tidyverse)

df_bfs <- read.csv('/Users/thowe/Projects/data_science/mastering_shiny/indicator_app/bfs_us.csv')
df_bfs$time <- as.Date(df_bfs$time)
print(head(df_bfs, 20))


# df_bfs <- df %>% filter(naics %in% c('44-45', '31-33'))
# print(head(df_bfs, 20))

# df_bfs %>%
#   ggplot(aes(time, BA_BA, colour=industry)) +
#   geom_line() +
#   labs(y = "Business Applications")

df_bds <- read.csv('/Users/thowe/Projects/data_science/mastering_shiny/indicator_app/bds_us.csv') %>%
  filter(FAGE == 10)
  # filter(naics %in% c('44-45', '31-33')) %>%
print(head(df_bds, 20))

# df_bds %>%
#   ggplot(aes(time, FIRM, colour=industry)) +
#   geom_line() +
#   labs(y = "Startups")


naics_unique <- unique(df[ , 3:4 ] )
naics_codes <- setNames(naics_unique$naics, naics_unique$industry)

ui <- fluidPage(
  fluidRow(
    column(6, selectInput("naics", "NAICS Code", choices = naics_codes))
  ),
  fluidRow(
    column(12, plotOutput("bfs"))
  ),
  fluidRow(
    column(12, plotOutput("bds"))
  )
)
server <- function(input, output, session) {
  selected_bfs <- reactive(df_bfs %>% filter(naics == input$naics))
  selected_bds <- reactive(df_bds %>% filter(naics == input$naics))
  
  # output$diag <- renderTable(
  #   selected() %>% count(diag, wt = weight, sort = TRUE)
  # )
  # output$body_part <- renderTable(
  #   selected() %>% count(body_part, wt = weight, sort = TRUE)
  # )
  # output$location <- renderTable(
  #   selected() %>% count(location, wt = weight, sort = TRUE)
  # )
  
  # summary <- reactive({
  #   selected() %>%
  #     count(age, sex, wt = weight) %>%
  #     left_join(population, by = c("age", "sex")) %>%
  #     mutate(rate = n / population * 1e4)
  # })
  
  output$bfs <- renderPlot({
    selected_bfs() %>%
      ggplot(aes(time, BA_BA)) +
      geom_line() +
      labs(y = "Business Applications")
  }, res = 96)
  output$bds <- renderPlot({
    selected_bds() %>%
      ggplot(aes(time, FIRM)) +
      geom_line() +
      labs(y = "Startups")
  }, res = 96)
}
shinyApp(ui, server)


# todo: slider underneath to change years
# todo: right panel with industry
