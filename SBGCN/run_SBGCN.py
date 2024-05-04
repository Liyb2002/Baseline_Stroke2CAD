
import brep_read

def run():
    step_path = '../preprocessing/canvas/step_4.step'

    graph = brep_read.create_graph_from_step_file(step_path)

    node_counts = graph.count_nodes_by_type()
    for node_type, count in node_counts.items():
        print(f"Number of {node_type} nodes: {count}")

run()