
class StraightLine3D:
    def __init__(self, data_block):
        self.clean_data(data_block)

    def __repr__(self):
        return f"StraightLine3D({self.point1}, {self.point2}, '{self.operation_type}', {self.line_id})"

    def clean_data(self,data_block):

        keys = []
        for key in data_block:
            keys.append(key)

        points = data_block[keys[0]]
        self.point0 = points[0]
        self.point1 = points[1]

        self.operation_type = data_block[keys[1]]

        self.feature_id = data_block[keys[2]]

        self.other_features = data_block[keys[3]]

        self.visibility_score = data_block[keys[4]]

        self.line_id = data_block[keys[5]]


class CurveLine3D:
    def __init__(self, data_block):
        self.clean_data(data_block)

    def __repr__(self):
        return f"CurveLine3D({self.points} '{self.operation_type}', {self.line_id})"

    def clean_data(self,data_block):

        keys = []
        for key in data_block:
            keys.append(key)

        self.points = data_block[keys[0]]

        self.operation_type = data_block[keys[1]]

        self.feature_id = data_block[keys[2]]

        self.other_features = data_block[keys[3]]

        self.visibility_score = data_block[keys[4]]

        self.line_id = data_block[keys[5]]
