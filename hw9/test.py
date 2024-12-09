import matplotlib.pyplot as plt

# Define the x and y coordinates
x = [1, 2]
y = [2, 3]

# Create a plot
plt.plot(x, y, label='Line', color='b', marker='o', linestyle='-', linewidth=2)

# Add labels and title
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Line Plot')

# Show the legend
plt.legend()

# Display the plot
plt.show()
