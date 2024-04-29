import io_utils
import brep_class

class Single_CAD_Simulation():
    def __init__ (self, CAD_program_path):
        self.parsed_CAD_program = io_utils.read_json_file(CAD_program_path)
        self.faces = []
        self.edges = []
        self.vertrices = []

        self.count = 0
        self.keys = list(self.parsed_CAD_program['entities'].keys())

    def CAD_process(self):

        entity_key = self.keys[self.count]
        self.count += 1

        print("entity_key", entity_key)
        entity_type = self.parsed_CAD_program['entities'][entity_key]['type']

        if entity_type == 'Sketch':
            self.process_sketch(self.parsed_CAD_program['entities'][entity_key])
        
        if entity_type == 'extrude':
            self.process_extrude(self.parsed_CAD_program['entities'][entity_key])


    def process_sketch(self, data):
        self.faces += brep_class.build_face(data['profiles']['faces'])
        self.edges += brep_class.build_edge(data['profiles']['edges'])
        self.vertices += brep_class.build_vertex(data['profiles']['vertices'])

    def process_extrude(self, data):
        print("extrude", data)



def Process_CAD_example():
    CAD_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/193/parsed_features.json'

    simulation = Single_CAD_Simulation(CAD_path)
    simulation.CAD_process()

    # print("-----")
    # simulation.CAD_process()
    # print("-----")
    # simulation.CAD_process()
    # print("-----")
    # simulation.CAD_process()


Process_CAD_example()