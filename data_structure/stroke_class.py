import torch


class StraightLine3D:
    def __init__(self, data_block):
        self.clean_data(data_block)
        self.type = 'straight_stroke'
        self.to_device('cpu')

    def __repr__(self):
        return f"StraightLine3D({self.point0}, {self.point1}, '{self.operation_type}', {self.line_id})"

    def clean_data(self,data_block):

        keys = []
        for key in data_block:
            keys.append(key)

        points = data_block[keys[0]]
        self.point0 = points[0]
        self.point1 = points[1]

        self.operation_type = line_feature_to_id(data_block[keys[1]])
        if self.operation_type == None:
            print('data_block[keys[1]', data_block[keys[1]])

        self.feature_id = data_block[keys[2]]

        self.other_features = data_block[keys[3]]

        self.visibility_score = data_block[keys[4]]

        self.line_id = data_block[keys[5]]

    def to_device(self, device):
        self.point0 = torch.tensor(self.point0, dtype=torch.float32)
        self.point1 = torch.tensor(self.point1, dtype=torch.float32)
        self.point0 = self.point0.to(device)
        self.point1 = self.point1.to(device)


class CurveLine3D:
    def __init__(self, data_block):
        self.clean_data(data_block)
        self.type = 'curve_stroke'
        self.to_device('cpu')

    def __repr__(self):
        return f"CurveLine3D({self.points} '{self.operation_type}', {self.line_id})"

    def clean_data(self,data_block):

        keys = []
        for key in data_block:
            keys.append(key)

        self.points = data_block[keys[0]]

        self.operation_type = line_feature_to_id(data_block[keys[1]])
        if self.operation_type == None:
            print("data_block[keys[1]]", data_block[keys[1]])

        self.feature_id = data_block[keys[2]]

        self.other_features = data_block[keys[3]]

        self.visibility_score = data_block[keys[4]]

        self.line_id = data_block[keys[5]]

    def to_device(self, device):
        self.points = torch.tensor(self.points, dtype=torch.float32)
        self.points = self.points.to(device)



def line_feature_to_id(string):
    line_feature_dict = {
        "extrude_line": 0,
        "grid_lines": 1,
        "section_lines": 2,
        "feature_line": 3,
        "sketch": 4,
        "fillet_line": 5,
        "silhouette_line": 6
    }
    return line_feature_dict.get(string.lower(), None)


def id_to_line_feature(id):
    line_feature_dict = {
        "extrude_line": 0,
        "grid_lines": 1,
        "section_lines": 2,
        "feature_line": 3,
        "sketch": 4,
        "fillet_line": 5,
        "silhouette_line": 6
    }
    inverted_dict = {v: k for k, v in line_feature_dict.items()}
    return inverted_dict.get(id, None)
