import json

def parseCAD(CAD_file_paths):
    result = []

    for file_path in CAD_file_paths:
        with open(file_path) as file:
            data = json.load(file)

        file_info = {
            'file_path': file_path,
            'entities': data['entities'],
            'bounding_box': {},
            'sequence': []
        }

        # Access the properties
        # properties = data['properties']
        # bounding_box = properties['bounding_box']
        # max_point = bounding_box['max_point']
        # min_point = bounding_box['min_point']

        # file_info['bounding_box'] = {
        #     'max_point': max_point,
        #     'min_point': min_point
        # }

        # Access the sequence
        sequence = data['sequence']

        for step in sequence:
            step_info = {
                'index': step['index'],
                'type': step['type'],
                'entity': step['entity']
            }
            file_info['sequence'].append(step_info)

        result.append(file_info)

    return result


def operation_to_id(string):
    operation_dict = {
        "sketch": 0,
        "extrude": 1,
        "fillet": 2
    }
    return operation_dict.get(string.lower(), None)

def sketch_entity(entities):

    entities_list = []

    for entity in entities.items():
        #this is not sketch operation
        if 'entities' in entity[1]:
            singele_entity_info = {
            'entities': entity[1]['entities'],
            }

        #this is sketch operation
        if 'profiles' in entity[1]:
            profiles = entity[1]['profiles']
            edges = profiles['edges']

            singele_entity_info = {
            'edges': [],
            }


            for edge in edges.items():
                edge_id = edge[1]['id']
                edge_type = edge[1]['param']['type']

                singe_edge_info = {
                'edge_id' : edge_id,
                'edge_type' : edge_type,
                'edge_direction': [],
                'edge_origin' : []
                }

                if edge_type == 'Line':
                    edge_direction = edge[1]['param']['direction']
                    edge_origin = edge[1]['param']['origin']
                    singe_edge_info['edge_direction'] = edge_direction
                    singe_edge_info['edge_origin'] = edge_origin

                singele_entity_info['edges'].append(singe_edge_info)
        
        entities_list.append(singele_entity_info)
    
    return entities_list

