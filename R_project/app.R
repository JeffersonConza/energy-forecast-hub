# R_project/app.R
library(shiny)
library(httr)
library(jsonlite)
library(ggplot2)
library(plotly)

ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "flatly"),
  titlePanel("⚡ Energy Forecast Hub (R Edition)"),
  
  sidebarLayout(
    sidebarPanel(
      dateInput("date_input", "Select Date", value = Sys.Date() + 1, min = "2020-01-01", max = "2030-12-31"),
      actionButton("predict_btn", "Get Forecast", class = "btn-success", width = "100%"),
      hr(),
      helpText("Backend: R Plumber"),
      helpText("Model: Auto-Selected Best Model")
    ),
    mainPanel(
      uiOutput("result_box"),
      plotlyOutput("trend_plot")
    )
  )
)

server <- function(input, output) {
  
  # Reactive calculation
  prediction <- eventReactive(input$predict_btn, {
    # Call the LOCAL Plumber API
    # Note: You must run api.R in a separate R terminal on port 8000
    res <- POST(paste0("http://127.0.0.1:8000/predict?date=", input$date_input))
    
    if (status_code(res) == 200) {
      fromJSON(content(res, "text"))
    } else {
      list(error = "API Error")
    }
  })
  
  output$result_box <- renderUI({
    res <- prediction()
    if (!is.null(res$error)) return(h4("Error connecting to API"))
    
    wellPanel(
      h2(paste(res$predicted_consumption_kw, "kW"), style = "color: #2c3e50; font-weight: bold;"),
      p(paste("Forecast for:", res$date))
    )
  })
  
  output$trend_plot <- renderPlotly({
    res <- prediction()
    if (is.null(res$predicted_consumption_kw)) return(NULL)
    
    # Simple visualization
    df <- data.frame(Type = "Forecast", Value = res$predicted_consumption_kw)
    
    p <- ggplot(df, aes(x = Type, y = Value)) +
      geom_col(fill = "#18bc9c", width = 0.4) +
      ylim(0, max(res$predicted_consumption_kw) * 1.5) +
      labs(y = "Consumption (kW)", title = "Predicted Usage") +
      theme_minimal()
    
    ggplotly(p)
  })
}

shinyApp(ui, server)