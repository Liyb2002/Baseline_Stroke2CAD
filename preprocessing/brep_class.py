


class Face:
    def __init__(self, face_data):
        print(f"An Face is created with ID: {face_data['id']}")
        self.id = face_data['id']
        self.edges = face_data['edges']
        self.param = face_data['param']
        self.loops_edge_ids = face_data['loops_edge_ids']



class Edge:
    def __init__(self, edge_data):
        print(f"An edge is created with ID: {edge_data['id']}")
        self.id = edge_data['id']
        self.param = edge_data['param']
        self.vertices = edge_data['vertices']


class Vertex:
    def __init__(self, vertex_data):
        print(f"A vertex is created with ID: {vertex_data['id']}")
        self.id = vertex_data['id']
        self.vector = vertex_data['param']['Vector']
        self.unit = vertex_data['param']['unit']


def build_face(data):
    faces = []
    for face_data in data:
        faces.append(Face(face_data))
    
    return faces


def build_edge(data):
    edges = []
    for edge_keys in data:
        edges.append(Edge(data[edge_keys]))
    
    return edges


def build_vertex(data):
    vertices = []
    for vertex_keys in data:
        vertices.append(Vertex(data[vertex_keys]))
    
    return vertices


    