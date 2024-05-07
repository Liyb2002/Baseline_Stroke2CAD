import numpy as np
import json

def generate_random_rectangle(perpendicular_axis):
    # Generate a random center point for the rectangle
    center = np.random.uniform(-10, 10, 3)
    
    # Generate random length and width
    length = np.random.uniform(1, 5)
    width = np.random.uniform(1, 5)
    
    # Calculate the half lengths
    half_length = length / 2
    half_width = width / 2

    # Initialize points array
    points = np.zeros((4, 3))

    # Randomly choose a direction for the normal vector (+1 or -1)
    normal_direction = np.random.choice([-1, 1])

    if perpendicular_axis == 'x':
        # Points in the YZ plane, X is constant
        constant_value = center[0]
        normal_vector = [normal_direction, 0, 0]  # Normal vector along x-axis
        points[:, 0] = constant_value
        points[:, 1] = [center[1] - half_length, center[1] - half_length, center[1] + half_length, center[1] + half_length]
        points[:, 2] = [center[2] - half_width, center[2] + half_width, center[2] + half_width, center[2] - half_width]
    elif perpendicular_axis == 'y':
        # Points in the XZ plane, Y is constant
        constant_value = center[1]
        normal_vector = [0, normal_direction, 0]  # Normal vector along y-axis
        points[:, 1] = constant_value
        points[:, 0] = [center[0] - half_length, center[0] - half_length, center[0] + half_length, center[0] + half_length]
        points[:, 2] = [center[2] - half_width, center[2] + half_width, center[2] + half_width, center[2] - half_width]
    elif perpendicular_axis == 'z':
        # Points in the XY plane, Z is constant
        constant_value = center[2]
        normal_vector = [0, 0, normal_direction]  # Normal vector along z-axis
        points[:, 2] = constant_value
        points[:, 0] = [center[0] - half_length, center[0] - half_length, center[0] + half_length, center[0] + half_length]
        points[:, 1] = [center[1] - half_width, center[1] + half_width, center[1] + half_width, center[1] - half_width]
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    return points, normal_vector

def save_points_to_json(points, index, filename='canvas/Program.json'):
    # Create a dictionary to store the operation data
    operation = {
        'operation': 'sketch',
        'vertices': []
    }
    
    # Add each point with an ID to the vertices list
    for count, point in enumerate(points):
        vertex_id = f"v_{index}_{count}"
        vertex = {
            'id': vertex_id,
            'coordinates': point.tolist()  # Convert numpy array to list for JSON serialization
        }
        operation['vertices'].append(vertex)
    
    # Load existing data from the file or create a new list if the file doesn't exist
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            # Ensure the data is a list to store multiple operations
            if not isinstance(data, list):
                data = []
    except FileNotFoundError:
        data = []
    
    # Append the new operation to the list of operations
    data.append(operation)
    
    # Write the updated data to a JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Data saved to {filename}")
