"""
用途: 图论模型演示 - 最短路 + 最小生成树 + 最大流 + 二分图匹配
输入: 内置案例数据
输出: 控制台结果 + output/graph_result.png
调用: python models/graph/graph_demo.py
"""
import sys
sys.path.insert(0, 'D:/虚拟C盘/数学建模培训')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ====== Case 1: Shortest Path ======
print("=" * 50)
print("Case 1: Dijkstra Shortest Path - Supply Distribution")
print("=" * 50)

G1 = nx.Graph()
edges1 = [
    ('Center','A',10),('Center','B',15),('Center','C',25),
    ('A','B',8),('A','D',20),('B','C',12),
    ('B','D',14),('C','D',18),('C','E',10),
    ('D','E',6),('D','F',15),('E','F',8)
]
G1.add_weighted_edges_from(edges1)
paths = nx.single_source_dijkstra_path(G1, 'Center', weight='weight')
dists = nx.single_source_dijkstra_path_length(G1, 'Center', weight='weight')

for node in sorted(dists.keys(), key=lambda n: dists[n]):
    if node != 'Center':
        print(f"  Center -> {node}: dist={dists[node]}, path={'->'.join(paths[node])}")

# ====== Case 2: MST ======
print("\n" + "=" * 50)
print("Case 2: Kruskal MST - Fiber Optic Layout")
print("=" * 50)

G2 = nx.Graph()
edges2 = [('A','B',5),('A','C',8),('A','D',12),('B','C',6),
          ('B','E',10),('C','D',7),('C','E',9),('C','F',11),
          ('D','F',8),('E','F',5)]
G2.add_weighted_edges_from(edges2)
mst = nx.minimum_spanning_tree(G2, weight='weight', algorithm='kruskal')
total_mst = sum(d['weight'] for _, _, d in mst.edges(data=True))
print(f"  MST total cost: {total_mst}")
for u, v, d in mst.edges(data=True):
    print(f"  {u} -- {v}: cost={d['weight']}")

# ====== Case 3: Max Flow ======
print("\n" + "=" * 50)
print("Case 3: Dinic Max Flow - Water Supply Network")
print("=" * 50)

G3 = nx.DiGraph()
edges3 = [
    ('Reservoir','A',20),('Reservoir','B',15),('A','B',8),
    ('A','C',15),('B','C',10),('B','D',12),
    ('C','City',25),('D','City',18),('A','D',5)
]
G3.add_weighted_edges_from(edges3, weight='capacity')
flow_val, flow_dict = nx.maximum_flow(G3, 'Reservoir', 'City', capacity='capacity')
print(f"  Max flow: {flow_val} m^3/h")
for u in flow_dict:
    for v, f in flow_dict[u].items():
        if f > 0:
            print(f"  {u} -> {v}: {f}/{G3[u][v]['capacity']} m^3/h")

# ====== Case 4: Bipartite Matching ======
print("\n" + "=" * 50)
print("Case 4: Bipartite Matching - Task Assignment")
print("=" * 50)

B = nx.Graph()
workers = ['W1','W2','W3','W4']
tasks = ['T1','T2','T3','T4']
B.add_nodes_from(workers, bipartite=0)
B.add_nodes_from(tasks, bipartite=1)
compatible = [('W1','T1'),('W1','T2'),('W2','T1'),
              ('W2','T3'),('W3','T2'),('W3','T4'),
              ('W4','T3'),('W4','T4')]
B.add_edges_from(compatible)
matching = nx.bipartite.maximum_matching(B)
print(f"  Matches: {len(matching)//2} pairs")
for w in workers:
    if w in matching:
        print(f"  {w} -> {matching[w]}")

# ====== Visualization ======
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# Shortest Path
ax = axes[0, 0]
pos1 = nx.spring_layout(G1, seed=42)
nx.draw(G1, pos1, ax=ax, with_labels=True, node_color='lightblue',
        edge_color='gray', node_size=600, font_size=8)
nx.draw_networkx_edge_labels(G1, pos1,
    edge_labels=nx.get_edge_attributes(G1, 'weight'), ax=ax, font_size=7)
ax.set_title('Case 1: Shortest Path Network')

# MST
ax = axes[0, 1]
pos2 = nx.spring_layout(G2, seed=42)
nx.draw(G2, pos2, ax=ax, with_labels=True, node_color='lightgray',
        edge_color='lightgray', node_size=500, style='dashed', font_size=8)
nx.draw(mst, pos2, ax=ax, with_labels=False, node_color='lightcoral',
        edge_color='darkred', node_size=500, width=2.5)
nx.draw_networkx_edge_labels(mst, pos2,
    edge_labels=nx.get_edge_attributes(mst, 'weight'), ax=ax, font_size=7)
ax.set_title(f'Case 2: MST (total cost={total_mst})')

# Max Flow
ax = axes[1, 0]
pos3 = nx.spring_layout(G3, seed=42)
nx.draw(G3, pos3, ax=ax, with_labels=True, node_color='lightyellow',
        edge_color='lightgray', node_size=700, font_size=8,
        arrows=True, arrowsize=15)
flow_labels = {(u,v): f"{f}/{G3[u][v]['capacity']}"
               for u in flow_dict for v, f in flow_dict[u].items() if f > 0}
nx.draw_networkx_edge_labels(G3, pos3, edge_labels=flow_labels, ax=ax, font_size=7)
ax.set_title(f'Case 3: Max Flow ({flow_val} m^3/h)')

# Bipartite Matching
ax = axes[1, 1]
pos_b = {}
for i, w in enumerate(workers):
    pos_b[w] = (0, 3 - i)
for i, t in enumerate(tasks):
    pos_b[t] = (1, 3 - i)
nx.draw(B, pos_b, ax=ax, with_labels=True,
        node_color=['lightblue' if n in workers else 'lightgreen' for n in B.nodes()],
        edge_color='gray', node_size=800, font_size=10)
matched_edges = [(w, matching[w]) for w in workers if w in matching]
nx.draw_networkx_edges(B, pos_b, edgelist=matched_edges, ax=ax,
                       edge_color='red', width=2.5)
ax.set_title(f'Case 4: Bipartite Matching ({len(matching)//2} pairs)')
ax.set_xlim(-0.5, 1.5)

plt.tight_layout()
plt.savefig('D:/虚拟C盘/数学建模培训/output/graph_result.png',
            dpi=300, bbox_inches='tight')
print("\n[OK] output/graph_result.png")
