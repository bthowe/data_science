library(shiny)
library(tidyverse)
library(lubridate)

df_bfs <- read.csv('/Users/thowe/Projects/data_science/mastering_shiny/indicator_app/bfs_us.csv')
df_bfs$time <- as.Date(df_bfs$time)
df_bfs$year <- year(df_bfs$time)
print(head(df_bfs, 20))

df_bfs_grouped <- df_bfs  %>%
                  group_by(fips, naics, year) %>%
                  summarise(BA_BA=sum(BA_BA))
print(as.data.frame(df_bfs_grouped))
# stop()

df_bds <- read.csv('/Users/thowe/Projects/data_science/mastering_shiny/indicator_app/bds_us.csv') %>%
  filter(FAGE == 10)
print(head(df_bds, 20))

# naics_unique <- unique(df[ , 3:4 ] )
# naics_codes <- setNames(naics_unique$naics, naics_unique$industry)
naics_codes <- setNames(
  c('00', '21', '22', '23', '31-33', '42', '44-45', '48-49', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81'),
  c('All', 'Mining', 'Utilities', 'Construction', 'Manufacturing', 'Wholesale', 'Retail', 'Transportation', 'Information', 'Finance', 'Real Estate', 'Pro. Services', 'Management', 'Admin. Services', 'Educational', 'Health Care', 'Arts', 'Food Services', 'Other')
)

bfs_time_range <- c(year(min(df_bfs$time)), year(max(df_bfs$time)))
bds_time_range <- c(min(df_bds$time), max(df_bds$time))

ui <- fluidPage(
  sidebarLayout(
    mainPanel(
      sidebarLayout(
        mainPanel(fluidRow(column(12, plotOutput("bfs"))), width=10),
        sidebarPanel(
          fluidRow(sliderInput("range_bfs", "Years", min=bfs_time_range[1], max=bfs_time_range[2], value = bfs_time_range, step=1, sep='')),
          fluidRow(checkboxInput("annualize", "Annual")),
          fluidRow(span('Tags: input, bfs', style='font-size:12px')),
          fluidRow(span('Source: ', a(href='https://www.census.gov/econ/currentdata/dbsearch?program=BFS&startYear=2004&endYear=2021&categories=TOTAL&dataType=BA_BA&geoLevel=US&adjusted=1&notAdjusted=1&errorData=0', 'Business Formation Statistics'), style='font-size:8px')),
          width=2),
      ),
      sidebarLayout(
        mainPanel(fluidRow(column(12, plotOutput("bds"))), width=10),
        sidebarPanel(
          fluidRow(sliderInput("range_bds", "Years", min=bds_time_range[1], max=bds_time_range[2], value=bds_time_range, step=1, sep='')),
          fluidRow(span('Tags: intermediate, bds', style='font-size:12px')),
          fluidRow(span('Source: ', a(href='https://www.census.gov/programs-surveys/bds/data.html', 'Business Dynamics Statistics'), style='font-size:8px')),
          width=2)
      ),
      width=10
    ),
    sidebarPanel(
      checkboxGroupInput('naics', 'NAICS Codes', choices=naics_codes, selected='00'),
      width=2
    )
  )
)
server <- function(input, output, session) {
  selected_bfs <- reactive(df_bfs %>% filter((naics %in% input$naics) & between(time, as.Date(paste0(input$range_bfs[1], '-01-01')), as.Date(paste0(input$range_bfs[2], '-12-31')))))
  selected_bds <- reactive(df_bds %>% filter((naics %in% input$naics) & (input$range_bds[1] <= time) & (input$range_bds[2] >= time)))
  
  annual_bfs <- reactive({
      selected_bfs() %>% 
      group_by(fips, naics, industry, year) %>% 
      summarise(BA_BA=sum(BA_BA))
  }
  )
  
  output$bfs <- renderPlot({
    if (input$annualize == FALSE) {
        selected_bfs() %>%
        ggplot(aes(time, BA_BA, colour=industry)) +
        geom_line() +
        labs(title = "Business Applications", y=NULL, x=NULL) + 
        theme(legend.position="bottom", legend.title=element_blank())
    }
    else {
        annual_bfs() %>%
        ggplot(aes(year, BA_BA, colour=industry)) +
        geom_line() +
        labs(title = "Business Applications", y=NULL, x=NULL) + 
        theme(legend.position="bottom", legend.title=element_blank())
    }
  }, res = 96)
  output$bds <- renderPlot({
    selected_bds() %>%
      ggplot(aes(time, FIRM, colour=industry)) +
      geom_line() +
      labs(title = "Startups", y=NULL, x=NULL) + 
      theme(legend.position="bottom", legend.title=element_blank())
  }, res = 96)
}
shinyApp(ui, server)


# todo: text that is characteristics of data
# todo: annualize
# todo: list of datasets to choose


