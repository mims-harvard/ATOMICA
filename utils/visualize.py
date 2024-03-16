import re
from itertools import combinations
from math import sqrt
import plotly.graph_objs as go
import networkx as nx
import numpy as np

atomic_radii = dict(
    Ac=1.88,
    Ag=1.59,
    Al=1.35,
    Am=1.51,
    As=1.21,
    Au=1.50,
    B=0.83,
    Ba=1.34,
    Be=0.35,
    Bi=1.54,
    Br=1.21,
    C=0.68,
    Ca=0.99,
    Cd=1.69,
    Ce=1.83,
    Cl=0.99,
    Co=1.33,
    Cr=1.35,
    Cs=1.67,
    Cu=1.52,
    D=0.23,
    Dy=1.75,
    Er=1.73,
    Eu=1.99,
    F=0.64,
    Fe=1.34,
    Ga=1.22,
    Gd=1.79,
    Ge=1.17,
    H=0.23,
    Hf=1.57,
    Hg=1.70,
    Ho=1.74,
    I=1.40,
    In=1.63,
    Ir=1.32,
    K=1.33,
    La=1.87,
    Li=0.68,
    Lu=1.72,
    Mg=1.10,
    Mn=1.35,
    Mo=1.47,
    N=0.68,
    Na=0.97,
    Nb=1.48,
    Nd=1.81,
    Ni=1.50,
    Np=1.55,
    O=0.68,
    Os=1.37,
    P=1.05,
    Pa=1.61,
    Pb=1.54,
    Pd=1.50,
    Pm=1.80,
    Po=1.68,
    Pr=1.82,
    Pt=1.50,
    Pu=1.53,
    Ra=1.90,
    Rb=1.47,
    Re=1.35,
    Rh=1.45,
    Ru=1.40,
    S=1.02,
    Sb=1.46,
    Sc=1.44,
    Se=1.22,
    Si=1.20,
    Sm=1.80,
    Sn=1.46,
    Sr=1.12,
    Ta=1.43,
    Tb=1.76,
    Tc=1.35,
    Te=1.47,
    Th=1.79,
    Ti=1.47,
    Tl=1.55,
    Tm=1.72,
    U=1.58,
    V=1.33,
    W=1.37,
    Y=1.78,
    Yb=1.94,
    Zn=1.45,
    Zr=1.56,
)


class MolGraph:
    """Represents a molecular graph."""

    __slots__ = [
        "elements",
        "x",
        "y",
        "z",
        "adj_list",
        "atomic_radii",
        "bond_lengths",
        "adj_matrix",
    ]

    def __init__(self):
        self.elements = []
        self.x = []
        self.y = []
        self.z = []
        self.adj_list = {}
        self.atomic_radii = []
        self.bond_lengths = {}
        self.adj_matrix = None

    def read_xyz(self, file_path: str) -> None:
        """Reads an XYZ file, searches for elements and their cartesian coordinates
        and adds them to corresponding arrays."""
        pattern = re.compile(
            r"([A-Za-z]{1,3})\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)\s*(-?\d+(?:\.\d+)?)"
        )
        with open(file_path) as file:
            for element, x, y, z in pattern.findall(file.read()):
                self.elements.append(element)
                self.x.append(float(x))
                self.y.append(float(y))
                self.z.append(float(z))
        self.atomic_radii = [atomic_radii[element] for element in self.elements]
        self._generate_adjacency_list()
    
    def read_coords(self, atoms, coords):
        self.elements = atoms
        self.x = [c[0] for c in coords]
        self.y = [c[1] for c in coords]
        self.z = [c[2] for c in coords]
        self.atomic_radii = [atomic_radii[element] for element in self.elements]
        self._generate_adjacency_list()

    def _generate_adjacency_list(self):
        """Generates an adjacency list from atomic cartesian coordinates."""

        node_ids = range(len(self.elements))
        xyz = np.stack((self.x, self.y, self.z), axis=-1)
        distances = xyz[:, np.newaxis, :] - xyz
        distances = np.sqrt(np.einsum("ijk,ijk->ij", distances, distances))

        atomic_radii = np.array(self.atomic_radii)
        distance_bond = (atomic_radii[:, np.newaxis] + atomic_radii) * 1.3

        adj_matrix = np.logical_and(0.1 < distances, distance_bond > distances).astype(
            int
        )

        for i, j in zip(*np.nonzero(adj_matrix)):
            self.adj_list.setdefault(i, set()).add(j)
            self.adj_list.setdefault(j, set()).add(i)
            self.bond_lengths[frozenset([i, j])] = round(distance_bond[i, j], 5)

        self.adj_matrix = adj_matrix

    def edges(self):
        """Creates an iterator with all graph edges."""
        edges = set()
        for node, neighbours in self.adj_list.items():
            for neighbour in neighbours:
                edge = frozenset([node, neighbour])
                if edge in edges:
                    continue
                edges.add(edge)
                yield node, neighbour

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, position):
        return self.elements[position], (
            self.x[position],
            self.y[position],
            self.z[position],
        )
    


cpk_colors = dict(
    Ar="cyan",
    B="salmon",
    Ba="darkgreen",
    Be="darkgreen",
    Br="darkred",
    C="black",
    Ca="darkgreen",
    Cl="green",
    Cs="violet",
    F="green",
    Fe="darkorange",
    Fr="violet",
    H="white",
    He="cyan",
    I="darkviolet",
    K="violet",
    Kr="cyan",
    Li="violet",
    Mg="darkgreen",
    N="blue",
    Na="violet",
    Ne="cyan",
    O="red",
    P="orange",
    Ra="darkgreen",
    Rb="violet",
    S="yellow",
    Sr="darkgreen",
    Ti="gray",
    Xe="cyan",
)
cpk_color_rest = "pink"


def to_plotly_figure(graph: MolGraph) -> go.Figure:
    """Creates a Plotly figure."""

    def atom_trace():
        """Creates an atom trace for the plot."""
        colors = [cpk_colors.get(element, cpk_color_rest) for element in graph.elements]
        markers = dict(
            color=colors,
            line=dict(color="lightgray", width=2),
            size=7,
            symbol="circle",
            opacity=0.8,
        )
        trace = go.Scatter3d(
            x=graph.x,
            y=graph.y,
            z=graph.z,
            mode="markers",
            marker=markers,
            text=graph.elements,
        )
        return trace

    def bond_trace():
        """ "Creates a bond trace for the plot."""
        trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            hoverinfo="none",
            mode="lines",
            marker=dict(color="grey", size=7, opacity=1),
        )
        adjascent_atoms = (
            (atom, neighbour)
            for atom, neighbours in graph.adj_list.items()
            for neighbour in neighbours
        )
        for i, j in adjascent_atoms:
            trace["x"] += (graph.x[i], graph.x[j], None)
            trace["y"] += (graph.y[i], graph.y[j], None)
            trace["z"] += (graph.z[i], graph.z[j], None)
        return trace

    annotations_elements = [
        dict(text=element, x=x, y=y, z=z, showarrow=False, yshift=15)
        for element, (x, y, z) in graph
    ]
    annotations_indices = [
        dict(text=number, x=x, y=y, z=z, showarrow=False, yshift=15)
        for number, (_, (x, y, z)) in enumerate(graph)
    ]

    annotations_bonds = []
    for (i, j), length in graph.bond_lengths.items():
        x = (graph.x[i] + graph.x[j]) / 2
        y = (graph.y[i] + graph.y[j]) / 2
        z = (graph.z[i] + graph.z[j]) / 2
        annotations_bonds.append(
            dict(
                text=round(length, 2),
                x=x,
                y=y,
                z=z,
                showarrow=False,
                yshift=15,
                font=dict(color="steelblue"),
            )
        )

    updatemenus = list(
        [
            dict(
                buttons=list(
                    [
                        dict(
                            label=" Elements",
                            method="relayout",
                            args=[{"scene.annotations": annotations_elements}],
                        ),
                        dict(
                            label=" Elements & Bond Lengths",
                            method="relayout",
                            args=[
                                {
                                    "scene.annotations": annotations_elements
                                    + annotations_bonds
                                }
                            ],
                        ),
                        dict(
                            label="Indices",
                            method="relayout",
                            args=[{"scene.annotations": annotations_indices}],
                        ),
                        dict(
                            label="Indices & Bond Lengths",
                            method="relayout",
                            args=[
                                {
                                    "scene.annotations": annotations_indices
                                    + annotations_bonds
                                }
                            ],
                        ),
                        dict(
                            label="Bond Lengths",
                            method="relayout",
                            args=[{"scene.annotations": annotations_bonds}],
                        ),
                        dict(
                            label="Hide All",
                            method="relayout",
                            args=[{"scene.annotations": []}],
                        ),
                    ]
                ),
                direction="down",
                xanchor="left",
                yanchor="top",
            ),
        ]
    )

    data = [atom_trace(), bond_trace()]
    axis_params = dict(
        showgrid=False,
        showbackground=False,
        showticklabels=False,
        zeroline=False,
        titlefont=dict(color="white"),
    )
    layout = dict(
        scene=dict(
            xaxis=axis_params,
            yaxis=axis_params,
            zaxis=axis_params,
            annotations=annotations_elements,
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        showlegend=False,
        updatemenus=updatemenus,
    )
    figure = go.Figure(data=data, layout=layout)

    return figure


def to_networkx_graph(graph: MolGraph) -> nx.Graph:
    """Creates a NetworkX graph.
    Atomic elements and coordinates are added to the graph as node attributes 'element' and 'xyz" respectively.
    Bond lengths are added to the graph as edge attribute 'length''"""
    G = nx.Graph(graph.adj_list)
    node_attrs = {
        num: {"element": element, "xyz": xyz}
        for num, (element, xyz) in enumerate(graph)
    }
    nx.set_node_attributes(G, node_attrs)
    edge_attrs = {
        edge: {"length": length} for edge, length in graph.bond_lengths.items()
    }
    nx.set_edge_attributes(G, edge_attrs)
    return G