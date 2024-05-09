import json
import build123.protocol
from basic_class import Face, Edge, Vertex
import helper


class parsed_program():
    def __init__(self, file_path):
        self.file_path = file_path

        self.canvas = None
        self.prev_sketch = None
        self.Op_idx = 0
        
    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
            for Op in data:
                operation = Op['operation']
                
                if operation[0] == 'sketch':
                    self.parse_sketch(Op)
                
                if operation[0] == 'extrude_addition':
                    self.parse_extrude_addition(Op)
                
                if operation[0] == 'fillet':
                    self.parse_fillet(Op)

        return

    def parse_sketch(self, Op):
        point_list = [vert['coordinates'] for vert in Op['vertices']]
        
        new_point_list = [point_list[0]]  # Start with the first point
        for i in range(1, len(point_list)):
            # Append each subsequent point twice
            new_point_list.append(point_list[i])
            new_point_list.append(point_list[i])
        
        # Add the first point again at the end to close the loop
        new_point_list.append(point_list[0])

        self.prev_sketch = build123.protocol.build_sketch(self.Op_idx, new_point_list)
        self.Op_idx += 1

    def parse_extrude_addition(self, Op):
        extrude_amount = Op['operation'][2]
        self.canvas = build123.protocol.build_extrude(self.Op_idx, self.canvas, self.prev_sketch, extrude_amount)
        self.Op_idx += 1
        
    def parse_fillet(self, Op):
        fillet_amount = Op['operation'][2]['amount']
        verts = Op['operation'][3]['verts']

        # print("fillet_amount", fillet_amount)
        # print("verts", verts)

        helper.find_target_verts(verts, self.canvas.edges())


        # existing_edges = self.canvas.edges()
        # t_edge = existing_edges[0]
        # print("vert", t_edge.vertices())

# Example usage:
file_path = './canvas/Program.json'
parsed_program_class = parsed_program(file_path)
parsed_program_class.read_json_file()