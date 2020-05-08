# Load packages
library(shiny)
library(shinythemes)
library(shinyWidgets)
library(dplyr)
library(ggplot2)
options(max.print=9999)
options(shiny.port = 7775)

# Get and check the input file
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("At least one argument must be supplied (input file).\n", call.=FALSE)
}

# Read file
to.read = file(args[1], "rb")

# Get file size
fileSize <- ((file.info(args[1])$size)/8)

# Initialize the matrix for the loaded data
result <- matrix(1:fileSize, ncol = 3, byrow=TRUE)
  
# Load data
for ( i in 1:(fileSize/3)){
    result[i,] = c(readBin(to.read, numeric(), 3))
}
close(to.read)


# Define UI
ui <- fluidPage(theme = shinytheme("lumen"),
  titlePanel("Metric Evaluation"),
  sidebarLayout(
    sidebarPanel(

      # Select date range to be plotted
      numericRangeInput(inputId = "number", label = strong("Observation Window Number"), value = c( 1, (fileSize/3) )),

      # Select whether to overlay smooth trend line
      checkboxInput(inputId = "smoother", label = strong("Overlay smooth trend line"), value = FALSE),

      # Display only if the smoother is checked
      conditionalPanel(condition = "input.smoother == true",
                       sliderInput(inputId = "f", label = "Smoother span:",
                                   min = 0.01, max = 1, value = 0.67, step = 0.01,
                                   animate = animationOptions(interval = 100)),
                       HTML("Higher values give more smoothness.")
      )
    ),

    # Output: Description, lineplot, and reference
    mainPanel(
      plotOutput(outputId = "lineplot", height = "300px"),
      plotOutput(outputId = "lineplot0", height = "300px"),
      plotOutput(outputId = "lineplot1", height = "300px")            
    )
  )
)


# Define server function
server <- shinyServer(function(input, output) {
  
  # Create scatterplot object the plotOutput function is expecting
  output$lineplot <- renderPlot({
    color = "#434343"
    plot(x = seq(input$number[1],input$number[2],1), y = result[,1][input$number[1]:input$number[2]], type = "l",
          xlab = "Observation Window Number", ylab = "Metric Values",main="Mean", col = color, fg = color, col.lab = color, col.axis = color)
    # Display only if smoother is checked
    if(input$smoother){
      smooth_curve <- lowess(x = seq(input$number[1],input$number[2],1), y = result[,1][input$number[1]:input$number[2]], f = input$f)
      lines(smooth_curve, col = "#E6553A", lwd = 3)
    }
  })

  output$lineplot0 <- renderPlot({
    color = "#434343"
    plot(x = seq(input$number[1],input$number[2],1), y = result[,2][input$number[1]:input$number[2]], type = "l",
        xlab = "Observation Window Number", ylab = "Metric Values",main="Variance", col = color, fg = color, col.lab = color, col.axis = color)
    # Display only if smoother is checked
    if(input$smoother){
      smooth_curve <- lowess(x = seq(input$number[1],input$number[2],1), y = result[,2][input$number[1]:input$number[2]], f = input$f)
      lines(smooth_curve, col = "#E6553A", lwd = 3)
    }   
  })

  output$lineplot1 <- renderPlot({
    color = "#434343"
    plot(x = seq(input$number[1],input$number[2],1), y = result[,3][input$number[1]:input$number[2]], type = "l",
        xlab = "Observation Window Number", ylab = "Metric Values",main="Silences", col = color, fg = color, col.lab = color, col.axis = color)
    # Display only if smoother is checked
    if(input$smoother){
      smooth_curve <- lowess(x = seq(input$number[1],input$number[2],1), y = result[,3][input$number[1]:input$number[2]], f = input$f)
      lines(smooth_curve, col = "#E6553A", lwd = 3)
    }   
  })

})

# Create Shiny object
shinyApp(ui = ui, server = server)



