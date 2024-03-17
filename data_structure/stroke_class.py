
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
        point0 = points[0]
        point1 = points[1]

        print("point0", point0, "point1", point1)
