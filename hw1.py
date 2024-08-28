import tkinter as tk
import pyautogui
import time
import numpy as np
import random

# Create the main window
root = tk.Tk()
root.title("Cursor Tracker")

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the center of the screen
center_x = screen_width // 2
center_y = screen_height // 2

v1_bounds = [[-400, 400], [1, center_y - 200]]
v2_adder_bounds = [100, 200]
v3_adder_bounds = [100, 200]

v1_before_transform = [random.randint(v1_bounds[0][0], v1_bounds[0][1]), random.randint(v1_bounds[1][0], v1_bounds[1][1])]
v2_before_transform = [v1_before_transform[0] + random.randint(v2_adder_bounds[0], v2_adder_bounds[1]), v1_before_transform[1]]
v3_before_transform = [random.randint(v1_before_transform[0], v2_before_transform[0]), v1_before_transform[1] + random.randint(v3_adder_bounds[0], v3_adder_bounds[1])]


root_x = 0
root_y = 0

# Create a canvas that covers the entire screen
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

# Draw the coordinate axes
canvas.create_line(center_x, 0, center_x, screen_height, fill="black", width=2)  # Y-axis
canvas.create_line(0, center_y, screen_width, center_y, fill="black", width=2)   # X-axis

# Create a small point (a circle) that will follow the cursor
cursor_point = canvas.create_oval(center_x - 5, center_y - 5, center_x + 5, center_y + 5, fill="red")

# Function to update the cursor position and draw a line from the origin
def update_cursor_position():
    global root_x, root_y
    x, y = pyautogui.position()
    # Adjust for window position
    root_x, root_y = root.winfo_rootx(), root.winfo_rooty()
    x_adjusted, y_adjusted = x - root_x - center_x, y - root_y - center_y
    
    direction_x = x_adjusted
    direction_y = y_adjusted
    length = np.sqrt(direction_x**2 + direction_y**2)
    
    # Normalize the direction vector and scale it to 100 pixels
    if length > 0:
        scale_factor = 100 / length
        direction_x *= scale_factor
        direction_y *= scale_factor
    
    # Calculate the end points of the line
    end_x = center_x + direction_x
    end_y = center_y + direction_y

    # Move the point to the cursor's position
    canvas.coords(cursor_point, end_x - 5, end_y - 5, end_x + 5, end_y + 5)

    # Delete any previous line
    canvas.delete("line")
    # Draw a line from the origin (center of the screen) to the cursor's position
    canvas.create_line(center_x, center_y, end_x, end_y, fill="blue", width=2, tags="line")
    return(x_adjusted, -y_adjusted)

# Main loop to keep updating the cursor position
while True:

    hit = False

    cursor_point_x, cursor_point_y = update_cursor_position()


    v1 = np.array([v1_before_transform[0], -v1_before_transform[1] + center_y, 1])
    v2 = np.array([v2_before_transform[0], -v2_before_transform[1] + center_y, 1])
    v3 = np.array([v3_before_transform[0], -v3_before_transform[1] + center_y, 1])
    p = np.array([cursor_point_x, cursor_point_y, 1])
    origin = np.array([0, 0, 1])

    l_cursor = np.cross(origin, p)
    l12 = np.cross(v1, v2)
    l23 = np.cross(v2, v3)
    l31 = np.cross(v3, v1)

    intersection12 = np.cross(l_cursor, l12)
    intersection23 = np.cross(l_cursor, l23)
    intersection31 = np.cross(l_cursor, l31)

    if(intersection12[2] != 0):
        p12 = [intersection12[0] / intersection12[2], intersection12[1] / intersection12[2]]
        if(v1[0] <= p12[0] and p12[0] <= v2[0] and v1[1] <= p12[1] and p12[1] <= v2[1]):
            hit = True
    if(intersection23[2] != 0):
        p23 = [intersection23[0] / intersection23[2], intersection23[1] / intersection23[2]]
        if(v3[0] <= p23[0] and p23[0] <= v2[0] and v3[1] <= p23[1] and p23[1] <= v2[1]):
            hit = True
    if(intersection31[2] != 0):
        p31 = [intersection31[0] / intersection31[2], intersection31[1] / intersection31[2]]
        if(v1[0] <= p31[0] and p31[0] <= v3[0] and v3[1] <= p31[1] and p31[1] <= v1[1]):
            hit = True


    if(not hit):
        canvas.create_line(v1_before_transform[0] + center_x, v1_before_transform[1], v2_before_transform[0] + center_x, v2_before_transform[1], fill="red", width=2)
        canvas.create_line(v2_before_transform[0] + center_x, v2_before_transform[1], v3_before_transform[0] + center_x, v3_before_transform[1], fill="red", width=2)
        canvas.create_line(v3_before_transform[0] + center_x, v3_before_transform[1], v1_before_transform[0] + center_x, v1_before_transform[1], fill="red", width=2)
    else:
        canvas.create_line(v1_before_transform[0] + center_x, v1_before_transform[1], v2_before_transform[0] + center_x, v2_before_transform[1], fill="green", width=2)
        canvas.create_line(v2_before_transform[0] + center_x, v2_before_transform[1], v3_before_transform[0] + center_x, v3_before_transform[1], fill="green", width=2)
        canvas.create_line(v3_before_transform[0] + center_x, v3_before_transform[1], v1_before_transform[0] + center_x, v1_before_transform[1], fill="green", width=2)

    root.update_idletasks()  # Update tasks in tkinter
    root.update()            # Update the tkinter window
    time.sleep(0.01)         # Sleep for 10 milliseconds to control update rate









