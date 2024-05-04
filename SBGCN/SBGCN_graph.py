class HeteroGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, node_type, features=None):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, "features": features}

    def add_edge(self, src_node_id, dst_node_id, edge_type, features=None):
        if (src_node_id, dst_node_id) not in self.edges:
            self.edges[(src_node_id, dst_node_id)] = {"type": edge_type, "features": features}

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_edge(self, src_node_id, dst_node_id):
        return self.edges.get((src_node_id, dst_node_id))

    def get_neighbors(self, node_id):
        neighbors = []
        for src, dst in self.edges:
            if src == node_id:
                neighbors.append(dst)
        return neighbors

    def get_node_ids_by_type(self, node_type):
        node_ids = []
        for node_id, data in self.nodes.items():
            if data["type"] == node_type:
                node_ids.append(node_id)
        return node_ids

    def get_edge_ids_by_type(self, edge_type):
        edge_ids = []
        for edge_id, data in self.edges.items():
            if data["type"] == edge_type:
                edge_ids.append(edge_id)
        return edge_ids

    def count_nodes_by_type(self):
        node_counts = {}
        for node_id, data in self.nodes.items():
            node_type = data["type"]
            if node_type in node_counts:
                node_counts[node_type] += 1
            else:
                node_counts[node_type] = 1
        return node_counts

    def avoid_duplicate(self, node_type, features):
        for node_id, data in self.nodes.items():
            if data["type"] == node_type and data["features"] == features:
                return node_id
        return None
