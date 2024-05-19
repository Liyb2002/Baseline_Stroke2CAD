import json
import build123.protocol
from basic_class import Face, Edge, Vertex
import helper


class create_stroke_cloud():
    def __init__(self, file_path, output = True):
        self.file_path = file_path

        self.id = 0
        self.faces = {}
        self.edges = {}
        self.vertices = {}
        
    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            for Op in data:
                operation = Op['operation']
                
                if operation[0] == 'sketch':
                    self.parse_op(Op)
                
                if operation[0] == 'extrude_addition' or operation[0] == 'extrude_substraction':
                    self.parse_op(Op)
                
                if operation[0] == 'fillet':
                    self.parse_op(Op)

        return


    def output(self):
        print("Outputting details of all components...")

        # Output vertices
        print("\nVertices:")
        for vertex_id, vertex in self.vertices.items():
            print(f"Vertex ID: {vertex_id}, Position: {vertex.position}")

        # Output edges
        print("\nEdges:")
        for edge_id, edge in self.edges.items():
            vertex_ids = [vertex.id for vertex in edge.vertices]
            print(f"Edge ID: {edge_id}, Vertices: {vertex_ids}")

        # Output faces
        print("\nFaces:")
        for face_id, face in self.faces.items():
            vertex_ids = [vertex.id for vertex in face.vertices]
            normal = face.normal
            print(f"Face ID: {face_id}, Vertices: {vertex_ids}, Normal: {normal}")

        
        
    def parse_op(self, Op):
        if len(Op['faces']) > 0 and 'radius' in Op['faces'][0]:
            print("parse circle")
            return


        op = Op['operation'][0]
        print("op", op)

        for vertex_data in Op['vertices']:
            vertex = Vertex(id=vertex_data['id'], position=vertex_data['coordinates'])
            self.vertices[vertex.id] = vertex

        for edge_data in Op['edges']:
            vertices = [self.vertices[v_id] for v_id in edge_data['vertices']]
            edge = Edge(id=edge_data['id'], vertices=vertices)
            self.edges[edge.id] = edge

        for face_data in Op['faces']:
            vertices = [self.vertices[v_id] for v_id in face_data['vertices']]
            normal = face_data['normal']
            face = Face(id=face_data['id'], vertices=vertices, normal=normal)
            self.faces[face.id] = face  



# Example usage:

def run():
    file_path = './canvas/Program.json'
    parsed_program_class = create_stroke_cloud(file_path)
    parsed_program_class.read_json_file()
    parsed_program_class.output()

run()