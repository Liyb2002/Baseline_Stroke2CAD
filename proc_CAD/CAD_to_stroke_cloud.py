import json
import build123.protocol
from basic_class import Face, Edge, Vertex
import helper

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class create_stroke_cloud():
    def __init__(self, file_path, output = True):
        self.file_path = file_path

        self.order_count = 0
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
            # Adding checks if 'Op' and 'order_count' are attributes of edge
            ops = getattr(edge, 'Op', 'No operations')
            order_count = getattr(edge, 'order_count', 'No order count')
            print(f"Edge ID: {edge_id}, Vertices: {vertex_ids}, Operations: {ops}, Order Count: {order_count}")

        # Output faces
        print("\nFaces:")
        for face_id, face in self.faces.items():
            vertex_ids = [vertex.id for vertex in face.vertices]
            normal = face.normal
            print(f"Face ID: {face_id}, Vertices: {vertex_ids}, Normal: {normal}")

    
    def vis_stroke_cloud(self, target_Op = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        
        for _, edge in self.edges.items():

            line_color = 'blue'

            if target_Op is not None and target_Op in edge.Op:
                line_color = 'red'

            
            points = [vertex.position for vertex in edge.vertices]
            if len(points) == 2:
                x_values = [points[0][0], points[1][0]]
                y_values = [points[0][1], points[1][1]]
                z_values = [points[0][2], points[1][2]]
                ax.plot(x_values, y_values, z_values, marker='o', color=line_color)  # Line plot connecting the vertices

        plt.show()

        
    def parse_op(self, Op):
        if len(Op['faces']) > 0 and 'radius' in Op['faces'][0]:
            print("parse circle")
            return


        op = Op['operation'][0]

        for vertex_data in Op['vertices']:
            vertex = Vertex(id=vertex_data['id'], position=vertex_data['coordinates'])
            self.vertices[vertex.id] = vertex


        cur_op_vertex_ids = []
        for edge_data in Op['edges']:
            vertices = [self.vertices[v_id] for v_id in edge_data['vertices']]

            for v_id in edge_data['vertices']:
                cur_op_vertex_ids.append(v_id)

            edge = Edge(id=edge_data['id'], vertices=vertices)
            edge.set_Op(op)
            edge.set_order_count(self.order_count)
            self.order_count += 1
            self.edges[edge.id] = edge
        
        #find the edges that has the current operation 
        #but not created by the current operation
        self.find_unwritten_edges(cur_op_vertex_ids, op)

        for face_data in Op['faces']:
            vertices = [self.vertices[v_id] for v_id in face_data['vertices']]
            normal = face_data['normal']
            face = Face(id=face_data['id'], vertices=vertices, normal=normal)
            self.faces[face.id] = face  

    def find_unwritten_edges(self, cur_op_vertex_ids, op):
        vertex_id_set = set(cur_op_vertex_ids)

        for edge_id, edge in self.edges.items():
            if all(vertex.id in vertex_id_set for vertex in edge.vertices):
                edge.set_Op(op)



# Example usage:

def run():
    file_path = './canvas/Program.json'
    parsed_program_class = create_stroke_cloud(file_path)
    parsed_program_class.read_json_file()
    # parsed_program_class.output()
    parsed_program_class.vis_stroke_cloud('sketch')

run()