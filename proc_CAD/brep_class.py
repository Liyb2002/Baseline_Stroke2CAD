

import json
import numpy as np
import helper

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
            vertex = Vertex(vertex_id, point.tolist())
            self.Vertices.append(vertex)
            vertex_list.append(vertex)

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

    def add_extrude_add_op(self, amount):
        target_face = self.Faces[-1]

        new_vertices = []
        new_edges = []
        new_faces = []

        for i, vertex in enumerate(target_face.vertices):

            new_pos = [vertex.position[j] + target_face.normal[j] * amount for j in range(3)]
            vertex_id = f"vertex_{self.idx}_{i}"
            new_vertex = Vertex(vertex_id, new_pos)
            self.Vertices.append(new_vertex)
            new_vertices.append(new_vertex)

        num_vertices = len(new_vertices)
        for i in range(num_vertices):
            edge_id = f"edge_{self.idx}_{i}"
            edge = Edge(edge_id, [new_vertices[i], new_vertices[(i+1) % num_vertices]])  # Loop back to first vertex to close the shape
            self.Edges.append(edge)
            new_edges.append(edge)

        face_id = f"face_{self.idx}_{0}"
        new_face = Face(face_id, new_vertices, [-x for x in target_face.normal])
        self.Faces.append(new_face)
        new_faces.append(new_face)
        
        
        #create side edges and faces
        for i in range(num_vertices):
            # Vertical edges from old vertices to new vertices
            vertical_edge_id = f"edge_{self.idx}_{i+num_vertices}"
            vertical_edge = Edge(vertical_edge_id, [target_face.vertices[i], new_vertices[i]])
            self.Edges.append(vertical_edge)

            # Side faces formed between pairs of old and new vertices
            side_face_id = f"face_{self.idx}_{i}"
            side_face_vertices = [
                target_face.vertices[i], new_vertices[i],
                new_vertices[(i + 1) % num_vertices], target_face.vertices[(i + 1) % num_vertices]
            ]
            normal = helper.compute_normal(side_face_vertices, new_vertices[(i + 2) % num_vertices])
            side_face = Face(side_face_id, side_face_vertices, normal)
            self.Faces.append(side_face)

        self.idx += 1
        self.op.append('extrude_addition')



    def write_to_json(self, filename='./canvas/Program.json'):
        data = []
        for count in range(0, self.idx):
            op = self.op[count]
            self.write_Op(op, count, data)
                
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data saved to {filename}")


    def write_Op(self, Op, index, data):
        operation = {
            'operation': Op,
            'faces': [],
            'edges': [],
            'vertices': []
        }

                # Add each point with an ID to the vertices list
        for face in self.Faces:
            if face.id.split('_')[1] == str(index):
                face = {
                    'id': face.id,
                    'vertices': [vertex.id for vertex in face.vertices],
                    'normal': [float(n) if isinstance(n, np.floating) else int(n) for n in face.normal]
                }
                operation['faces'].append(face)

        for edge in self.Edges:
            if edge.id.split('_')[1] == str(index):
                
                edge = {
                    'id': edge.id,
                    'vertices': [vertex.id for vertex in edge.vertices]
                }
                operation['edges'].append(edge)


        
        for vertex in self.Vertices:
            if vertex.id.split('_')[1] == str(index):
                vertex = {
                    'id': vertex.id,
                    'coordinates': vertex.position  # Convert numpy array to list for JSON serialization
                }
                operation['vertices'].append(vertex)
        

        data.append(operation)

        return data
        





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

    