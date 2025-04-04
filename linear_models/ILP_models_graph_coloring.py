from multiprocessing import Pool, Manager
from gurobipy import GRB
from collections import deque
import itertools
import gurobipy as gp
import networkx as nx
import time
import os
import csv


class AbstractMethod:
    def __init__(self, name, graph=None):
        self.name = name
        self.graph = graph

    def set_graph(self, graph):
        self.graph = graph

    def solve(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")

#######################################################################################################################################################
    
class AssignmentModel(AbstractMethod):
    def __init__(self, name):
        super().__init__(name)

    def solve(self):
        start_time = time.time()
        H = get_coloring_upper_bound(self.graph) + 1

        model = gp.Model(self.name)
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 60 * 60)

        zmax = model.addVar(vtype=GRB.CONTINUOUS, name="zmax")
        x = {}

        for i in self.graph.nodes:
            for j in range(1, H):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        model.setObjective(zmax, GRB.MINIMIZE)

        for i in self.graph.nodes:
            expr = 0
            for j in range(1, H):
                expr += x[i, j]

            model.addConstr(expr == 1, name=f"vertex_{i}")

        for [i, ngh] in self.graph.edges:
            for j in range(1, H):
                model.addConstr(x[i, j] + x[ngh, j] <= 1)

        for i in self.graph.nodes:
            for j in range(1, H):
                model.addConstr(j * x[i, j] <= zmax)
        
        model.update()
        model.optimize()
        end_time = time.time()
        result = []
        if model.status == GRB.OPTIMAL:
            for i in self.graph.nodes:
                for j in range(1, H):
                    if x[i, j].x > 0.5:
                        result.append(j)
                        break

        return (result, end_time - start_time)

#######################################################################################################################################################

class MixedPartialOrderModel(AbstractMethod):
    def __init__(self, name):
        super().__init__(name)

    def solve(self):
        start_time = time.time()
        H = get_coloring_upper_bound(self.graph) + 1
        model = gp.Model("AdvancedGraphColoring")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', 60 * 60)

        q = get_node_from_max_clique(self.graph)
        y = {}
        z = {}
        x = {}

        for v in self.graph.nodes:
            for i in range(1, H):
                y[i, v] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{v}")
                z[v, i] = model.addVar(vtype=GRB.BINARY, name=f"z_{v}_{i}")
                x[v, i] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{i}")
        
        model.setObjective(1 + gp.quicksum(y[i, q] for i in range(1, H)), GRB.MINIMIZE)
        
        for v in self.graph.nodes:
            model.addConstr(z[v, 1] == 0)
        
        for v in self.graph.nodes:
            model.addConstr(y[H - 1, v] == 0)
        
        for v in self.graph.nodes:
            for i in range(1, H - 1):
                model.addConstr(y[i, v] - y[i+1, v] >= 0)
        
        for v in self.graph.nodes:
            for i in range(1, H - 1):
                model.addConstr(y[i, v] + z[v, i+1] == 1)
        
        for v in self.graph.nodes:
            for i in range(1, H):
                model.addConstr(x[v, i] == 1 - (y[i, v] + z[v, i]))

        for [v, ngh] in self.graph.edges:
            for i in range(1, H):
                model.addConstr(x[v, i] + x[ngh, i] <= 1)

        for v in self.graph.nodes:
            for i in range(1, H):
                model.addConstr(y[i, q] - y[i, v] >= 0)
                
        model.update()  
        model.optimize()
        end_time = time.time()
        result = []
        if model.status == GRB.OPTIMAL:
            for i in self.graph.nodes:
                for j in range(1, H):
                    if x[i, j].x > 0.5:
                        result.append(j)
                        break
        
        return (result, end_time - start_time)

#######################################################################################################################################################

class BranchAndPriceNode:
    def __init__(self, same, different, columns):
        self.same = same.copy()
        self.different = different.copy()
        self.columns = columns.copy()
        self.lb = 0.0
        self.x = {}
        self.depth = 0

#######################################################################################################################################################

class BranchAndPriceModel(AbstractMethod):
    def __init__(self, name):
        super().__init__(name)
        self.best_upper = float('inf')
        self.best_solution = []
        self.nodes = deque()
        

    def create_initial_columns(self):
        coloring = nx.greedy_color(self.graph, strategy="largest_first")
        color_classes = {}
        index = 0
        for node, color in coloring.items():
            color_classes.setdefault(color, []).append(node)
            index += 1

        for node in self.graph.nodes():
            color_classes.setdefault(index, []).append(node)
            index += 1

        return list(color_classes.values())

    def create_maxim_independent_sets_for_node(self):
        max_indep_sets = []
        for node in self.graph.nodes():
            new_graph = self.graph.copy()
            new_graph.remove_node(node)
            for neighbor in self.graph.neighbors(node):
                new_graph.remove_node(neighbor)

            independent_set = nx.maximal_independent_set(new_graph)
            if independent_set:
                max_indep_sets.append(independent_set)
        return max_indep_sets
        

    def solve_rmp(self, node):
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 60 * 60)

        x = {idx: m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{idx}") for idx in range(len(node.columns))}
        m.setObjective(gp.quicksum(x[idx] for idx in x), GRB.MINIMIZE)
        
        vertex_constraints = {}
        for u in self.graph.nodes():
            lhs = gp.quicksum(x[idx] for idx, S in enumerate(node.columns) if u in S)
            vertex_constraints[u] = m.addConstr(lhs >= 1, name=f"cover_{u}")
        
        m.optimize()
        if m.status == GRB.OPTIMAL:
            node.lb = m.objVal
            node.x = {idx: var.X for idx, var in x.items()}
            duals = {u: vertex_constraints[u].Pi for u in self.graph.nodes()}
            return duals
        return None
    
    def solve_pricing(self, duals, node):
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', 60 * 60)

        z = {u: m.addVar(vtype=GRB.BINARY, name=f"z_{u}") for u in self.graph.nodes()}
        m.setObjective(gp.quicksum(duals[u] * z[u] for u in self.graph.nodes()) - 1, GRB.MAXIMIZE)
        
        for u, v in self.graph.edges():
            m.addConstr(z[u] + z[v] <= 1)
        
        for u, v in node.same:
            m.addConstr(z[u] == z[v])
        for u, v in node.different:
            m.addConstr(z[u] + z[v] <= 1)
        
        m.optimize()
        if m.objVal > 1 + 1e-6: 
            return [u for u in self.graph.nodes() if z[u].X > 0.5]
        return None
    
    def select_branching_vertices(self, node):
        for idx, val in node.x.items():
            if 1e-6 < val < 1 - 1e-6:
                S = node.columns[idx]
                for u in S:
                    for v in self.graph.nodes():
                        if v == u:
                            continue
                        for idx2 in node.x:
                            if idx2 == idx:
                                continue
                            if node.x[idx2] < 1e-6:
                                continue
                            if u in node.columns[idx2] and v in node.columns[idx2]:
                                return u, v
        return None, None

    def solve(self):
        start_time = time.time()
        self.best_upper = get_coloring_upper_bound(self.graph) + 1
        initial_columns = self.create_initial_columns()
        root_node = BranchAndPriceNode([], [], initial_columns)
        self.nodes.append(root_node)

        max_iterations = 100
        max_depth = 100
        current_iteration = 0
        while self.nodes:
            if current_iteration >= max_iterations or time.time() - start_time > 60 * 60:
                break

            node = self.nodes.popleft()
            if node.depth >= max_depth:
                continue

            duals = self.solve_rmp(node)
            if not duals:
                continue
            
            while True:
                new_col = self.solve_pricing(duals, node)
                if new_col:
                    node.columns.append(new_col)
                    duals = self.solve_rmp(node)
                else:
                    break
            
            is_integer = all(abs(val - round(val)) < 1e-6 for val in node.x.values())
            if is_integer:
                if len(node.columns) < self.best_upper:
                    self.best_upper = len(node.columns)
                    self.best_solution = [
                        S 
                        for idx, S in enumerate(node.columns) 
                        if round(node.x.get(idx, 0)) == 1
                    ]
                    current_iteration = 0
                continue
            
            u, v = self.select_branching_vertices(node)
            if not u or not v:
                continue
            
            same_child = BranchAndPriceNode(node.same + [(u, v)], node.different, node.columns)
            same_child.columns = [S for S in same_child.columns if (u in S and v in S) or (u not in S and v not in S)]
            same_child.depth = node.depth + 1
            self.nodes.append(same_child)
            
            different_child = BranchAndPriceNode(node.same, node.different + [(u, v)], node.columns)
            different_child.columns = [S for S in different_child.columns if not (u in S and v in S)]
            different_child.depth = node.depth + 1
            self.nodes.append(different_child)

            current_iteration += 1

        return ([len(self.best_solution)], time.time() - start_time)

#######################################################################################################################################################


def get_node_from_max_clique(graph):
    cliques = list(nx.clique.find_cliques(graph))
    max_clique = max(cliques, key=len)
    return max_clique[0] if max_clique else graph.nodes[0]


def get_coloring_upper_bound(graph):
    return max(nx.coloring.greedy_color(graph, strategy='independent_set').values()) + 1


def read_graph(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    G = nx.Graph()

    for line in lines:
        if line.startswith('e'):
            parts = line.split()
            u = int(parts[1]) - 1 
            v = int(parts[2]) - 1 
            G.add_edge(u, v)


    return G


def process_task(graph_file, model):
    try:
        graph = read_graph(graph_file)
        model.set_graph(graph)
        result = model.solve()
        
        with lock:
            with open(output_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if len(result[0]) > 0:
                    writer.writerow([os.path.basename(graph_file), model.name, max(result[0]), result[1]])
                else:
                    writer.writerow([os.path.basename(graph_file), model.name, 'tl', result[1]])
    except Exception as e:
        print(f"Error processing {graph_file}: {e}")


def init_child(lock_, output_filename_):
    global lock, output_filename
    lock = lock_
    output_filename = output_filename_


def batch_solve(models, folder):
    output_filename = os.path.join(folder, 'output_lp_models.csv')
    graph_files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.endswith('.col')
    ]

    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Name', 'Model', 'K*', 'Time'])

    manager = Manager()
    lock = manager.Lock()

    tasks = itertools.product(graph_files, models)

    with Pool(
        initializer=init_child,
        initargs=(lock, output_filename)
    ) as pool:
        pool.starmap(process_task, tasks)


def graph_visualisation(graph):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10)
    plt.show()


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.path.abspath(__file__ + '/..'))
    instances_dir = os.path.join(parent_dir, 'coloring_instances\mid')
    
    # TODO Branch and price needs to be modified OR changed to Column Generation only
    models = [
        AssignmentModel("GraphColoringAssignment"),
        MixedPartialOrderModel("GraphColoringMixedPartialOrder")
        # BranchAndPriceModel('GraphColoringBranchAndPrice')
    ]

    try:
        batch_solve(models, instances_dir)
    except Exception as e:
        print(e)