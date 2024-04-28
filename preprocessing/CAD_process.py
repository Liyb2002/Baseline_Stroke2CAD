import io_utils
import brep_class

def CAD_process(CAD_program):

    parsed_CAD_program = io_utils.read_json_file(CAD_program)
    for entity_key in parsed_CAD_program['entities']:
        entity_type = parsed_CAD_program['entities'][entity_key]['type']

        if entity_type == 'Sketch':
            process_sketch(parsed_CAD_program['entities'][entity_key])
        
        if entity_type == 'extrude':
            process_extrude(parsed_CAD_program['entities'][entity_key])


def process_sketch(data):
    faces = brep_class.build_face(data['profiles']['faces'])
    edges = brep_class.build_edge(data['profiles']['edges'])
    vertices = brep_class.build_vertex(data['profiles']['vertices'])

def process_extrude(data):
    print("extrude", data)

def Process_CAD_example():
    CAD_path = '/Users/yuanboli/Documents/GitHub/Baseline_Stroke2CAD/dataset/CAD2Sketch/193/parsed_features.json'

    CAD_process(CAD_path)

Process_CAD_example()