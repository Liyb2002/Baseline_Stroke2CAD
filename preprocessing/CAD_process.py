import io_utils
import brep_class
import build123.protocol
import random
import build123.helper

class Single_CAD_Simulation():
    def __init__ (self, CAD_program_path):
        self.parsed_CAD_program = io_utils.read_json_file(CAD_program_path)
        self.faces = {}
        self.edges = {}
        self.vertices = {}

        self.count = 0
        self.keys = list(self.parsed_CAD_program['entities'].keys())

        self.canvas = None
        self.targetFace = []
        self.targetFace_normals = []

    def CAD_process(self):

        entity_key = self.keys[self.count]
        self.count += 1

        entity_type = self.parsed_CAD_program['entities'][entity_key]['type']

        if entity_type == 'Sketch':
            self.process_sketch(self.parsed_CAD_program['entities'][entity_key])
        
        if entity_type == 'extrude':
            self.process_extrude(self.parsed_CAD_program['entities'][entity_key])


    def process_sketch(self, data):
        self.targetFace = []
        new_faces = brep_class.build_face(data['profiles']['faces'])
        new_edges = brep_class.build_edge(data['profiles']['edges'])
        new_vertices = brep_class.build_vertex(data['profiles']['vertices'])

        #update the dict
        for face in new_faces:
            if face.id not in self.faces:
                self.faces[face.id] = face
        
        for edge in new_edges:
            if edge.id not in self.edges:
                self.edges[edge.id] = edge

        for vertex in new_vertices:
            if vertex.id not in self.vertices:
                self.vertices[vertex.id] = vertex


        #get the vertices for current face to build
        whole_sketch_translation = data['transform']['origin']
        whole_sketch_rotation_raw = data['transform']
        whole_sketch_rotation = build123.helper.Compute_angle(whole_sketch_rotation_raw)

        for face in new_faces:
            face_translation = face.param['origin']
            per_face_rotation = face.param['x']
            vertex_ids = []

            for edge_id in face.loops_edge_ids[0]:
                edge = self.edges.get(edge_id)
                if edge:
                    vertex_ids.append(edge.vertices)
                else:
                    print(f"Edge ID {edge_id} not found in the edge dictionary.")
            

            vertex_ids = io_utils.ensure_sequential_vertex_order(vertex_ids)
            
            
            #build the sketch
            total_translation = build123.helper.combine_translations(face_translation, whole_sketch_translation)
            point_list = [self.vertices[vertex_id].vector for pair in vertex_ids for vertex_id in pair]

            self.targetFace_normals.append(build123.helper.check_normal_direction(face.param['normal']))
            self.targetFace.append(build123.protocol.build_sketch(
                        self.count,self.canvas, point_list, 
                        total_translation, 
                        whole_sketch_rotation, per_face_rotation))
            


    def process_extrude(self, data):

        count = 0
        for target_face in self.targetFace:
            if count == 0 and data['extent_one'] != 0:
                self.canvas = build123.protocol.build_extrude(self.count, self.canvas, target_face, data['extent_one'] * self.targetFace_normals[count])
            else:
                if data['extent_two'] != 0:
                    self.canvas = build123.protocol.build_extrude(self.count, self.canvas, target_face, data['extent_two'] * self.targetFace_normal)
            count += 1
            


def Process_CAD_example():
    CAD_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/193/parsed_features.json'

    simulation = Single_CAD_Simulation(CAD_path)
    simulation.CAD_process()

    print("-----")
    simulation.CAD_process()
    print("-----")
    simulation.CAD_process()
    print("-----")
    simulation.CAD_process()


Process_CAD_example()