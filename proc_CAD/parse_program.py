import json
import build123.protocol

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for Op in data:
            operation = Op['operation']
            
            if operation == 'sketch':
                parse_sketch(Op)

    return data

def parse_sketch(Op):
    point_list = [vert['coordinates'] for vert in Op['vertices']]
    
    new_point_list = [point_list[0]]  # Start with the first point
    for i in range(1, len(point_list)):
        # Append each subsequent point twice
        new_point_list.append(point_list[i])
        new_point_list.append(point_list[i])
    
    # Add the first point again at the end to close the loop
    new_point_list.append(point_list[0])

    build123.protocol.build_sketch(0, new_point_list)



# Example usage:
file_path = './canvas/Program.json'
data = read_json_file(file_path)
