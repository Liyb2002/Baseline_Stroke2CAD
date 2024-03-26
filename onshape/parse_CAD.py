import json

def parseCAD(CAD_file_paths):
    result = []

    for file_path in CAD_file_paths:
        with open(file_path) as file:
            data = json.load(file)

        file_info = {
            'file_path': file_path,
            'entities': [],
            'bounding_box': {},
            'sequence': []
        }

        # # Access the entities
        # entities = data['entities']

        # for entity_id, entity_data in entities.items():
        #     entity_info = {
        #         'id': entity_id,
        #         'name': entity_data['name'],
        #         'type': entity_data['type']
        #     }
        #     file_info['entities'].append(entity_info)

        # # Access the properties
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
