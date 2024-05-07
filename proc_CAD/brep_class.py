

import json
import numpy as np


class Brep:
    def __init__(self):
        self.Faces = []
        self.Edges = []
        self.Vertices = []

        self.op = []
        self.idx = 0
        
    
    def add_sketch_op(self, points, normal):

        vertex_list = []
        for i, point in enumerate(points):
            vertex_id = f"vertex_{self.idx}_{i}"
            vertex = Vertex(vertex_id, point)
            self.Vertices.append(vertex)
            vertex_list.append(point)

        num_vertices = len(vertex_list)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [vertex_list[i], vertex_list[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            self.Edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        face = Face(face_id, vertex_list, normal)
        self.Faces.append(face)
        
        self.idx += 1
        self.op.append('sketch')

    def write_to_json(self):
        for count in range(0, self.idx):
            op = self.op[count]
            if op == 'sketch':
                self.write_sketch(count)
                

    def write_sketch(self, index, filename='./canvas/Program.json'):
        operation = {
            'operation': 'sketch',
            'faces': [],
            'edges': [],
            'vertices': []
        }

                # Add each point with an ID to the vertices list
        for face in self.Faces:
            if face.id.split('_')[1] == str(index):
                face = {
                    'id': face.id,
                    'vertices': [vertex.tolist() for vertex in face.vertices],
                    'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                }
                operation['faces'].append(face)

        for edge in self.Edges:
            if edge.id.split('_')[1] == str(index):
                
                edge = {
                    'id': edge.id,
                    'vertices': [vertex.tolist() for vertex in edge.vertices],
                }
                operation['edges'].append(edge)


        
        for vertex in self.Vertices:
            if vertex.id.split('_')[1] == str(index):
                vertex = {
                    'id': vertex.id,
                    'coordinates': vertex.position.tolist()  # Convert numpy array to list for JSON serialization
                }
                operation['vertices'].append(vertex)
        



        # Load existing data from the file or create a new list if the file doesn't exist        
        # Append the new operation to the list of operations
        data = operation
        
        # Write the updated data to a JSON file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data saved to {filename}")





class Face:
    def __init__(self, id, vertices, normal):
        print(f"An Face is created with ID: {id}")
        self.id = id
        self.vertices = vertices
        self.normal = normal


class Edge:
    def __init__(self, id, vertices):
        print(f"An edge is created with ID: {id}")
        self.id = id
        self.vertices = vertices

class Vertex:
    def __init__(self, id, vertices):
        print(f"A vertex is created with ID: {id}")
        self.id = id
        self.position = vertices

    