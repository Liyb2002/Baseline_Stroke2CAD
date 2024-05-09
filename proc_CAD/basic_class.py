
class Face:
    def __init__(self, id, vertices, normal):
        print(f"An Face is created with ID: {id}")
        self.id = id
        self.vertices = vertices
        self.normal = normal


class Edge:
    def __init__(self, id, vertices):
        print(f"An edge is created with ID: {id}")
        self.id = id
        self.vertices = vertices
        self.round = False
    
    def fillet_edge(self):
        self.round = True

class Vertex:
    def __init__(self, id, vertices):
        print(f"A vertex is created with ID: {id}")
        self.id = id
        self.position = vertices

    