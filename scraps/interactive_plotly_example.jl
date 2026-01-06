using Plots
plotly()  # Set Plotly as the backend for interactive plotting

# Create a simple interactive 3D plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
z = [1, 3, 5, 7, 9]

scatter(x, y, z, 
    marker = :circle,
    markersize = 5,
    xlabel = "X Axis",
    ylabel = "Y Axis",
    zlabel = "Z Axis",
    title = "Interactive 3D Scatter Plot",
    legend = false)