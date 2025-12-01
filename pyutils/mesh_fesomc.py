
"""
FESOM-C mesh class for loading and processing unstructured mesh data.
Used with FESOM-C 'drift' model and new 'rise' model.

Created on Mon Dec  1 12:00:00 2025

@author: Ivan Kuznetsov aka kuivi
"""

import numpy as np
import numba
from numba import njit, prange
import os
# Force CPU backend due to jax-metal compatibility issues with JAX 0.8.1
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import jit, vmap

# Enable 64-bit precision for JAX (required for scientific meshes)
jax.config.update("jax_enable_x64", True)

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import pickle
import logging

# Configure logging to display warnings and information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Numba-accelerated kernels for topology operations
# =============================================================================

@njit(cache=True)
def _numba_count_node_elements(nod2d: int, elem2d: int, elem2d_nodes: np.ndarray) -> np.ndarray:
    """Count the number of elements each node belongs to."""
    ne_num = np.zeros(nod2d, dtype=np.int32)
    for n in range(elem2d):
        elnodes = elem2d_nodes[n, :]
        if elnodes[0] == elnodes[3]:
            # Triangle: nodes 0,1,2
            for i in range(3):
                ne_num[elnodes[i]] += 1
        else:
            # Quadrilateral: nodes 0,1,2,3
            for i in range(4):
                ne_num[elnodes[i]] += 1
    return ne_num


@njit(cache=True)
def _numba_build_ne_pos(nod2d: int, elem2d: int, elem2d_nodes: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build node-to-element adjacency array."""
    ne_num = np.zeros(nod2d, dtype=np.int32)
    ne_pos = np.zeros((nod2d, k), dtype=np.int32)

    for n in range(elem2d):
        elnodes = elem2d_nodes[n, :]
        q1 = 3 if elnodes[0] == elnodes[3] else 4
        for q in range(q1):
            node = elnodes[q]
            ne_pos[node, ne_num[node]] = n
            ne_num[node] += 1

    return ne_num, ne_pos


@njit(cache=True)
def _numba_count_neighbor_nodes(nod2d: int, elem2d: int, elem2d_nodes: np.ndarray,
                                 ne_num: np.ndarray, ne_pos: np.ndarray) -> np.ndarray:
    """Count the number of neighbor nodes for each node."""
    nn_num = np.zeros(nod2d, dtype=np.int32)
    aux1 = np.zeros(nod2d, dtype=np.int32)
    ed = np.zeros(2, dtype=np.int32)

    for n in range(nod2d):
        counter = 0
        for k in range(ne_num[n]):
            elem = ne_pos[n, k]
            elnodes = elem2d_nodes[elem, :]
            if elnodes[0] == elnodes[3]:
                # Triangle
                for q in range(3):
                    if elnodes[q] == n:
                        continue
                    if aux1[elnodes[q]] != 1:
                        counter += 1
                        aux1[elnodes[q]] = 1
            else:
                # Quadrilateral
                if elnodes[0] == n:
                    ed[0] = elnodes[1]
                    ed[1] = elnodes[3]
                elif elnodes[1] == n:
                    ed[0] = elnodes[0]
                    ed[1] = elnodes[2]
                elif elnodes[2] == n:
                    ed[0] = elnodes[1]
                    ed[1] = elnodes[3]
                else:  # elnodes[3] == n
                    ed[0] = elnodes[0]
                    ed[1] = elnodes[2]
                for q in range(2):
                    if aux1[ed[q]] != 1:
                        counter += 1
                        aux1[ed[q]] = 1

        nn_num[n] = counter
        # Reset aux1 for nodes we marked
        for k in range(ne_num[n]):
            elem = ne_pos[n, k]
            elnodes = elem2d_nodes[elem, :]
            if elnodes[0] == elnodes[3]:
                for q in range(3):
                    aux1[elnodes[q]] = 0
            else:
                for q in range(4):
                    aux1[elnodes[q]] = 0

    return nn_num


@njit(cache=True)
def _numba_build_nn_pos(nod2d: int, elem2d: int, elem2d_nodes: np.ndarray,
                        ne_num: np.ndarray, ne_pos: np.ndarray, maxnn: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build node-to-node neighbor array."""
    nn_num = np.zeros(nod2d, dtype=np.int32)
    nn_pos = np.zeros((nod2d, maxnn), dtype=np.int32)
    aux1 = np.zeros(nod2d, dtype=np.int32)
    temp = np.zeros(100, dtype=np.int32)
    ed = np.zeros(2, dtype=np.int32)

    for n in range(nod2d):
        counter = 0
        for k in range(ne_num[n]):
            elem = ne_pos[n, k]
            elnodes = elem2d_nodes[elem, :]
            if elnodes[0] == elnodes[3]:
                # Triangle
                for q in range(3):
                    if elnodes[q] == n:
                        continue
                    if aux1[elnodes[q]] != 1:
                        if counter < 100:
                            temp[counter] = elnodes[q]
                        counter += 1
                        aux1[elnodes[q]] = 1
            else:
                # Quadrilateral
                if elnodes[0] == n:
                    ed[0] = elnodes[1]
                    ed[1] = elnodes[3]
                elif elnodes[1] == n:
                    ed[0] = elnodes[0]
                    ed[1] = elnodes[2]
                elif elnodes[2] == n:
                    ed[0] = elnodes[1]
                    ed[1] = elnodes[3]
                else:
                    ed[0] = elnodes[0]
                    ed[1] = elnodes[2]
                for q in range(2):
                    if aux1[ed[q]] != 1:
                        if counter < 100:
                            temp[counter] = ed[q]
                        counter += 1
                        aux1[ed[q]] = 1

        nn_num[n] = counter + 1
        # Reset aux1
        for i in range(min(counter, 100)):
            aux1[temp[i]] = 0
        # Fill nn_pos
        if counter > 0:
            for i in range(min(counter, maxnn - 1)):
                nn_pos[n, i + 1] = temp[i]
        nn_pos[n, 0] = n  # The node itself

    return nn_num, nn_pos


@njit(cache=True)
def _numba_count_edges(nod2d: int, nn_num: np.ndarray, nn_pos: np.ndarray) -> int:
    """Count total number of edges."""
    counter = 0
    for n in range(nod2d):
        for q in range(1, nn_num[n]):
            node = nn_pos[n, q]
            if node > n:
                counter += 1
    return counter


@njit(cache=True)
def _numba_find_edges(nod2d: int, elem2d: int, elem2d_nodes: np.ndarray,
                      ne_num: np.ndarray, ne_pos: np.ndarray,
                      nn_num: np.ndarray, nn_pos: np.ndarray,
                      edge2d: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Find edges and the elements on each side."""
    edge_nodes = np.zeros((edge2d, 2), dtype=np.int32)
    edge_tri = np.zeros((edge2d, 2), dtype=np.int32)
    counter_in = 0

    # First pass: internal edges (edges with 2 elements)
    for n in range(nod2d):
        for q in range(1, nn_num[n]):
            node = nn_pos[n, q]
            if node < n:
                continue

            # Find elements containing both n and node
            flag = 0
            elems = np.zeros(2, dtype=np.int32)
            for k in range(ne_num[n]):
                elem = ne_pos[n, k]
                elnodes = elem2d_nodes[elem, :]
                q2 = 3 if elnodes[0] == elnodes[3] else 4
                for q1 in range(q2):
                    if elnodes[q1] == node:
                        if flag < 2:
                            elems[flag] = elem
                        flag += 1
                        break
                if flag >= 2:
                    break

            if flag == 2:
                edge_nodes[counter_in, 0] = n
                edge_nodes[counter_in, 1] = node
                edge_tri[counter_in, 0] = elems[0]
                edge_tri[counter_in, 1] = elems[1]
                counter_in += 1

    edge2d_in = counter_in

    # Second pass: boundary edges (edges with 1 element)
    for n in range(nod2d):
        for q in range(1, nn_num[n]):
            node = nn_pos[n, q]
            if node < n:
                continue

            flag = 0
            elems = np.zeros(2, dtype=np.int32)
            for k in range(ne_num[n]):
                elem = ne_pos[n, k]
                elnodes = elem2d_nodes[elem, :]
                q2 = 3 if elnodes[0] == elnodes[3] else 4
                for q1 in range(q2):
                    if elnodes[q1] == node:
                        if flag < 2:
                            elems[flag] = elem
                        flag += 1
                        break

            if flag == 1:
                edge_nodes[counter_in, 0] = n
                edge_nodes[counter_in, 1] = node
                edge_tri[counter_in, 0] = elems[0]
                edge_tri[counter_in, 1] = -999
                counter_in += 1

    return edge_nodes, edge_tri, edge2d_in


@njit(cache=True)
def _numba_build_elem_edges(edge2d: int, elem2d: int, edge_tri: np.ndarray,
                            edge_nodes: np.ndarray, elem2d_nodes: np.ndarray) -> np.ndarray:
    """Build element-to-edge mapping."""
    elem_edges = np.zeros((elem2d, 4), dtype=np.int32)
    aux1_elem = np.zeros(elem2d, dtype=np.int32)

    # First pass: populate elem_edges
    for n in range(edge2d):
        for k in range(2):
            q = edge_tri[n, k]
            if q >= 0:
                elem_edges[q, aux1_elem[q]] = n
                aux1_elem[q] += 1

    # Second pass: order edges to match node order
    for elem in range(elem2d):
        elnodes = elem2d_nodes[elem, :]
        q1 = 3 if elnodes[0] == elnodes[3] else 4
        eledges = elem_edges[elem, :].copy()

        for q in range(q1 - 1):
            for k in range(q1):
                edge = eledges[k]
                if ((edge_nodes[edge, 0] == elnodes[q] and edge_nodes[edge, 1] == elnodes[q + 1]) or
                    (edge_nodes[edge, 0] == elnodes[q + 1] and edge_nodes[edge, 1] == elnodes[q])):
                    elem_edges[elem, q] = edge
                    break

        for k in range(q1):
            edge = eledges[k]
            if ((edge_nodes[edge, 0] == elnodes[q1 - 1] and edge_nodes[edge, 1] == elnodes[0]) or
                (edge_nodes[edge, 0] == elnodes[0] and edge_nodes[edge, 1] == elnodes[q1 - 1])):
                elem_edges[elem, q1 - 1] = edge
                break

        if q1 == 3:
            elem_edges[elem, 3] = elem_edges[elem, 0]

    return elem_edges


@njit(cache=True)
def _numba_find_elem_neighbors(elem2d: int, elem_edges: np.ndarray, edge_tri: np.ndarray,
                               elem2d_nodes: np.ndarray) -> np.ndarray:
    """Find neighbors for each element."""
    elem_neighbors = np.zeros((elem2d, 4), dtype=np.int32)

    for elem in range(elem2d):
        elnodes = elem_edges[elem, :]
        q = 3 if elnodes[0] == elnodes[3] else 4

        for j in range(q):
            eedge = elem_edges[elem, j]
            elem1 = edge_tri[eedge, 0]
            if elem1 == elem:
                elem1 = edge_tri[eedge, 1]
            elem_neighbors[elem, j] = elem1

        if q == 3:
            elem_neighbors[elem, 3] = elem_neighbors[elem, 0]

    return elem_neighbors


@njit(cache=True)
def _numba_build_nod_in_elem2d(nod2d: int, elem2d: int, elem2d_nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build node-to-element mapping for each node."""
    # First pass: count
    nod_in_elem2d_num = np.zeros(nod2d, dtype=np.int32)
    for el in range(elem2d):
        elnodes = elem2d_nodes[el, :]
        q = 3 if elnodes[0] == elnodes[3] else 4
        for j in range(q):
            nod_in_elem2d_num[elnodes[j]] += 1

    max_elements_per_node = np.max(nod_in_elem2d_num)
    nod_in_elem2d = np.zeros((nod2d, max_elements_per_node), dtype=np.int32)

    # Reset for second pass
    nod_in_elem2d_num[:] = 0

    # Second pass: populate
    for el in range(elem2d):
        elnodes = elem2d_nodes[el, :]
        q = 3 if elnodes[0] == elnodes[3] else 4
        for j in range(q):
            node = elnodes[j]
            nod_in_elem2d[node, nod_in_elem2d_num[node]] = el
            nod_in_elem2d_num[node] += 1

    return nod_in_elem2d_num, nod_in_elem2d


# =============================================================================
# JAX-accelerated kernels for geometry operations
# =============================================================================

def _jax_elem_center_tri(elnodes: jnp.ndarray, coord_nod2d: jnp.ndarray, cyclic_length: float) -> Tuple[float, float, float]:
    """Calculate center and area for a single triangle using JAX."""
    # elnodes: (3,) or (4,) with elnodes[3] == elnodes[0] for triangles
    ax = coord_nod2d[elnodes[:3], :]  # (3, 2)
    amin = jnp.min(ax[:, 0])
    ax_x = jnp.where(ax[:, 0] - amin > cyclic_length / 2.0, ax[:, 0] - cyclic_length, ax[:, 0])
    ax_x = jnp.where(ax_x - amin < -cyclic_length / 2.0, ax_x + cyclic_length, ax_x)
    x = jnp.mean(ax_x)
    y = jnp.mean(ax[:, 1])
    s = 0.5 * jnp.abs((ax_x[1] - ax_x[0]) * (ax[2, 1] - ax[0, 1]) - (ax[1, 1] - ax[0, 1]) * (ax_x[2] - ax_x[0]))
    return x, y, s


def _jax_elem_center_quad(elnodes: jnp.ndarray, coord_nod2d: jnp.ndarray, cyclic_length: float) -> Tuple[float, float, float]:
    """Calculate center and area for a single quadrilateral using JAX."""
    ax = coord_nod2d[elnodes, :]  # (4, 2)
    amin = jnp.min(ax[:, 0])
    ax_x = jnp.where(ax[:, 0] - amin > cyclic_length / 2.0, ax[:, 0] - cyclic_length, ax[:, 0])
    ax_x = jnp.where(ax_x - amin < -cyclic_length / 2.0, ax_x + cyclic_length, ax_x)

    x1 = jnp.sum(ax_x[:3])
    y1 = jnp.sum(ax[:3, 1])
    s1 = jnp.abs((ax_x[2] - ax_x[3]) * (ax[0, 1] - ax[3, 1]) - (ax[2, 1] - ax[3, 1]) * (ax_x[0] - ax_x[3]))
    s2 = jnp.abs((ax_x[2] - ax_x[1]) * (ax[0, 1] - ax[1, 1]) - (ax[2, 1] - ax[1, 1]) * (ax_x[0] - ax_x[1]))
    s = 0.5 * (s1 + s2)

    # Avoid division by zero
    denom = 3.0 * (s1 + s2)
    denom = jnp.where(denom == 0, 1.0, denom)

    x = (jnp.sum(ax_x[jnp.array([0, 2, 3])]) * s1 + x1 * s2) / denom
    y = (jnp.sum(ax[jnp.array([0, 2, 3]), 1]) * s1 + y1 * s2) / denom
    return x, y, s


def _jax_elem_center_single(elem: int, elem2d_nodes: jnp.ndarray, coord_nod2d: jnp.ndarray, cyclic_length: float) -> Tuple[float, float, float]:
    """Calculate center and area for a single element using JAX with conditional."""
    elnodes = elem2d_nodes[elem, :]
    is_tri = elnodes[0] == elnodes[3]

    # Compute both cases
    x_tri, y_tri, s_tri = _jax_elem_center_tri(elnodes, coord_nod2d, cyclic_length)
    x_quad, y_quad, s_quad = _jax_elem_center_quad(elnodes, coord_nod2d, cyclic_length)

    # Select based on element type
    x = jnp.where(is_tri, x_tri, x_quad)
    y = jnp.where(is_tri, y_tri, y_quad)
    s = jnp.where(is_tri, s_tri, s_quad)

    return x, y, s


@jit
def _jax_compute_all_elem_centers(elem2d_nodes: jnp.ndarray, coord_nod2d: jnp.ndarray, cyclic_length: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute centers and areas for all elements using JAX vmap."""
    elem2d = elem2d_nodes.shape[0]
    elem_indices = jnp.arange(elem2d)

    # Vectorize over all elements
    compute_fn = lambda elem: _jax_elem_center_single(elem, elem2d_nodes, coord_nod2d, cyclic_length)
    x_all, y_all, s_all = vmap(compute_fn)(elem_indices)

    return x_all, y_all, s_all


@jit
def _jax_cdiff(a: jnp.ndarray, b: jnp.ndarray, cyclic_length: float) -> jnp.ndarray:
    """Compute cyclic difference using JAX."""
    diff = a - b
    diff_x = diff[0]
    diff_x = jnp.where(diff_x > cyclic_length / 2.0, diff_x - cyclic_length, diff_x)
    diff_x = jnp.where(diff_x < -cyclic_length / 2.0, diff_x + cyclic_length, diff_x)
    return jnp.array([diff_x, diff[1]])


def _jax_edge_center(n1: int, n2: int, coord_nod2d: jnp.ndarray, cyclic_length: float) -> Tuple[float, float]:
    """Calculate edge center using JAX."""
    a = coord_nod2d[n1, :]
    b = coord_nod2d[n2, :]
    a_x = jnp.where(a[0] - b[0] > cyclic_length / 2.0, a[0] - cyclic_length, a[0])
    b_x = jnp.where(a_x - b[0] < -cyclic_length / 2.0, b[0] + cyclic_length, b[0])
    x = 0.5 * (a_x + b_x)
    y = 0.5 * (a[1] + b[1])
    return x, y


@dataclass
class RiseMesh:
    # Core fields (always required)
    nod2d: int = field(init=False)
    coord_nod2d: np.ndarray = field(init=False)
    elem2d: int = field(init=False)
    elem2d_nodes: np.ndarray = field(init=False)

    # Optional fields (can be None for partial meshes)
    nodes_filename: Optional[str] = None
    nodes_obindex: Optional[np.ndarray] = None
    nod2d_ob: Optional[int] = None
    nodes_ob: Optional[np.ndarray] = None
    nodes_in: Optional[np.ndarray] = None
    elem_filename: Optional[str] = None
    elem_tri: Optional[np.ndarray] = None
    numtrian: Optional[int] = None
    numquads: Optional[int] = None
    depth: Optional[np.ndarray] = None
    ne_num: Optional[np.ndarray] = None
    ne_pos: Optional[np.ndarray] = None
    nn_num: Optional[np.ndarray] = None
    nn_pos: Optional[np.ndarray] = None
    edge2d: Optional[int] = None
    edge_nodes: Optional[np.ndarray] = None
    edge_tri: Optional[np.ndarray] = None
    edge2d_in: Optional[int] = None
    elem_edges: Optional[np.ndarray] = None
    elem_neighbors: Optional[np.ndarray] = None
    nod_in_elem2d_num: Optional[np.ndarray] = None
    nod_in_elem2d: Optional[np.ndarray] = None
    elem_area: Optional[np.ndarray] = None
    area: Optional[np.ndarray] = None
    metric_factor: Optional[np.ndarray] = None
    elem_cos: Optional[np.ndarray] = None
    coriolis: Optional[np.ndarray] = None
    w_cv: Optional[np.ndarray] = None
    edge_dxdy: Optional[np.ndarray] = None
    edge_cross_dxdy: Optional[np.ndarray] = None
    gradient_sca: Optional[np.ndarray] = None
    gradient_vec: Optional[np.ndarray] = None

    def __post_init__(self):
        """
        Validate that core fields are set if this is intended to be a complete mesh.

        Note: This only validates after initialization. Use classmethods to create instances:
            - RiseMesh.read_mesh(config) - Full mesh from files
            - RiseMesh.load(filepath) - Load from pickle
        """
        # No validation here - allow partial meshes
        # Core fields (nod2d, coord_nod2d, elem2d, elem2d_nodes) will be set by classmethods
        pass

    def save(self, output_dir: str = "./") -> str:
        """
        Save the mesh instance to a pickle file.

        Args:
            output_dir: Directory where to save the mesh file (default: "./")

        Returns:
            str: Full path to the saved file

        Raises:
            AttributeError: If core fields (nod2d, elem2d) are not set

        Example:
            >>> mesh = RiseMesh.read_mesh(config)
            >>> filepath = mesh.save(output_dir="./cache")
            >>> print(f"Saved to {filepath}")
        """
        # Core fields are required for saving (have field(init=False), so must be set manually)
        # If not set, accessing them will raise AttributeError
        try:
            filename = f"fesomc_mesh_N{self.nod2d}_E{self.elem2d}.pkl"
        except AttributeError as e:
            raise AttributeError(
                "Cannot save mesh: core fields (nod2d, elem2d) must be set. "
                "Use RiseMesh.read_mesh(config) to create a complete mesh."
            ) from e

        filepath = Path(output_dir) / filename

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Mesh saved to: {filepath.absolute()}")
        return str(filepath.absolute())

    @classmethod
    def load(cls, filepath: str) -> 'RiseMesh':
        """
        Load a mesh instance from a pickle file.

        Args:
            filepath: Path to the pickle file

        Returns:
            RiseMesh: The loaded mesh instance

        Raises:
            FileNotFoundError: If the file doesn't exist

        Example:
            >>> mesh = RiseMesh.load("./fesomc_mesh_N1000_E2000.pkl")
        """
        file_path = Path(filepath)

        if not file_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {file_path.absolute()}")

        with open(file_path, 'rb') as f:
            mesh_instance = pickle.load(f)

        print(f"Mesh loaded from: {file_path.absolute()}")
        return mesh_instance

    @staticmethod
    def cdiff(x2: np.ndarray, x1: np.ndarray, cyclic_length: float) -> np.ndarray:
        """
        Compute the cyclic difference between two points.

        Args:
            x2 (np.ndarray): Second point coordinates.
            x1 (np.ndarray): First point coordinates.
            cyclic_length (float): Length of the cyclic boundary.

        Returns:
            np.ndarray: The difference vector considering cyclic conditions.
        """
        xdiff = x2 - x1
        half_cyclic = cyclic_length / 2.0
        xdiff[0] = (xdiff[0] + half_cyclic) % cyclic_length - half_cyclic
        return xdiff

    @staticmethod
    def elem_center(elem: int, elem2d_nodes: np.ndarray, coord_nod2d: np.ndarray, cyclic_length: float) -> Tuple[float, float, float]:
        """
        Calculate the center and area of an element.

        Args:
            elem (int): Element index (0-based).
            elem2d_nodes (np.ndarray): Nodes of each element. Shape: (elem2d, 4)
            coord_nod2d (np.ndarray): Coordinates of nodes. Shape: (nod2d, 2)
            cyclic_length (float): Length of the cyclic boundary.

        Returns:
            Tuple[float, float, float]: Center coordinates (x, y) and area.
        """
        # Phase 3 refactor: Access row elem for element nodes
        elnodes = elem2d_nodes[elem, :]
        if elnodes[0] == elnodes[3]:  # Triangle case (nodes are 0-based)
            # Phase 3.1 refactor: Keep as (3, 2) - no transpose needed
            # ax[:, 0] = X coordinates, ax[:, 1] = Y coordinates
            ax = coord_nod2d[elnodes[:3], :].copy()
            amin = np.min(ax[:, 0])
            ax[:, 0] = np.where(ax[:, 0] - amin > cyclic_length / 2.0, ax[:, 0] - cyclic_length, ax[:, 0])
            ax[:, 0] = np.where(ax[:, 0] - amin < -cyclic_length / 2.0, ax[:, 0] + cyclic_length, ax[:, 0])
            x = np.mean(ax[:, 0])
            y = np.mean(ax[:, 1])
            # Cross product: (X2-X1)*(Y3-Y1) - (Y2-Y1)*(X3-X1)
            s = 0.5 * abs((ax[1, 0] - ax[0, 0]) * (ax[2, 1] - ax[0, 1]) - (ax[1, 1] - ax[0, 1]) * (ax[2, 0] - ax[0, 0]))
        else:  # Quadrilateral case
            # Phase 3.1 refactor: Keep as (4, 2) - no transpose needed
            ax = coord_nod2d[elnodes, :].copy()
            amin = np.min(ax[:, 0])
            ax[:, 0] = np.where(ax[:, 0] - amin > cyclic_length / 2.0, ax[:, 0] - cyclic_length, ax[:, 0])
            ax[:, 0] = np.where(ax[:, 0] - amin < -cyclic_length / 2.0, ax[:, 0] + cyclic_length, ax[:, 0])
            x1 = np.sum(ax[:3, 0])
            y1 = np.sum(ax[:3, 1])
            s1 = abs((ax[2, 0] - ax[3, 0]) * (ax[0, 1] - ax[3, 1]) - (ax[2, 1] - ax[3, 1]) * (ax[0, 0] - ax[3, 0]))
            s2 = abs((ax[2, 0] - ax[1, 0]) * (ax[0, 1] - ax[1, 1]) - (ax[2, 1] - ax[1, 1]) * (ax[0, 0] - ax[1, 0]))
            s = 0.5 * (s1 + s2)
            x = (np.sum(ax[[0, 2, 3], 0]) * s1 + x1 * s2) / (3.0 * (s1 + s2))
            y = (np.sum(ax[[0, 2, 3], 1]) * s1 + y1 * s2) / (3.0 * (s1 + s2))
        return x, y, s

    @staticmethod
    def edge_center(n1: int, n2: int, coord_nod2d: np.ndarray, cyclic_length: float) -> Tuple[float, float]:
        """
        Calculate the center of an edge.

        Args:
            n1 (int): First node index (0-based).
            n2 (int): Second node index (0-based).
            coord_nod2d (np.ndarray): Coordinates of nodes. Shape: (nod2d, 2)
            cyclic_length (float): Length of the cyclic boundary.

        Returns:
            Tuple[float, float]: Center coordinates (x, y).
        """
        # Phase 3 refactor: Access coord_nod2d[node, :] for row-major layout
        a = coord_nod2d[n1, :].copy()
        b = coord_nod2d[n2, :].copy()
        if a[0] - b[0] > cyclic_length / 2.0:
            a[0] -= cyclic_length
        if a[0] - b[0] < -cyclic_length / 2.0:
            b[0] += cyclic_length
        x = 0.5 * (a[0] + b[0])
        y = 0.5 * (a[1] + b[1])
        return x, y

    @staticmethod
    def read_nodes_from_file(filename: str, config: Dict[str, Any]) -> Tuple[int, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Read nodes from a file.

        Args:
            filename (str): Path to the nodes file.
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Tuple containing:
                - nod2d (int): Total number of nodes.
                - coord_nod2d (np.ndarray): Coordinates of nodes. Shape: (nod2d, 2)
                - nodes_obindex (np.ndarray): Open boundary indices.
                - nod2d_ob (int): Number of open boundary nodes.
                - nodes_ob (np.ndarray): Open boundary nodes.
                - nodes_in (np.ndarray): Internal nodes.
        """
        r_earth = 6400000.0
        rad = np.pi / 180.0

        with open(filename, 'r') as file:
            nod2d = int(file.readline())
            # Phase 1 refactor: Initialize as (nod2d, 2) for row-major layout
            coord_nod2d = np.empty((nod2d, 2), dtype=float)
            nodes_obindex = np.empty(nod2d, dtype=int)
            nod2d_ob = 0

            for i in range(nod2d):
                line = file.readline().strip()
                parts = line.split()
                # Adjusting indices: Julia is 1-based, Python is 0-based
                index = int(parts[0]) - 1
                value1 = float(parts[1])
                value2 = float(parts[2])
                obindex = int(parts[3])
                # Phase 1 refactor: Assign as row [lon, lat]
                coord_nod2d[i, :] = [value1, value2]
                nodes_obindex[i] = obindex
                if obindex == 2:
                    nod2d_ob += 1

            nodes_ob = np.array([i for i, ob in enumerate(nodes_obindex) if ob == 2], dtype=int)
            nodes_in = np.setdiff1d(np.arange(nod2d), nodes_ob)

            if config["mesh"]["is_cartesian"]:
                coord_nod2d /= r_earth
            else:
                coord_nod2d *= rad

            return nod2d, coord_nod2d, nodes_obindex, nod2d_ob, nodes_ob, nodes_in

    @staticmethod
    def read_elements_from_file(filename: str) -> Tuple[int, np.ndarray, np.ndarray, int, int]:
        """
        Read elements from a file.

        Args:
            filename (str): Path to the elements file.

        Returns:
            Tuple containing:
                - elem2d (int): Number of elements.
                - elem2d_nodes (np.ndarray): Nodes of each element. Shape: (elem2d, 4)
                - elem_tri (np.ndarray): Triangle flags for elements.
                - numtrian (int): Number of triangles.
                - numquads (int): Number of quadrilaterals.
        """
        with open(filename, 'r') as file:
            elem2d = int(file.readline())
            # Phase 1 refactor: Initialize as (elem2d, 4) for row-major layout
            elem2d_nodes_unsort = np.empty((elem2d, 4), dtype=int)
            elem_tri = np.zeros(elem2d, dtype=bool)

            for i in range(elem2d):
                line = file.readline().strip()
                parts = line.split()
                value1 = int(parts[0]) - 1
                value2 = int(parts[1]) - 1
                value3 = int(parts[2]) - 1
                value4 = int(parts[3]) - 1
                # Phase 1 refactor: Assign as row [n0, n1, n2, n3]
                elem2d_nodes_unsort[i, :] = [value1, value2, value3, value4]
                elem_tri[i] = False

            # Sort elements: triangles first
            numtrian = 0
            numquads = 0
            elem2d_nodes = np.empty_like(elem2d_nodes_unsort)
            for i in range(elem2d):
                # Phase 1 refactor: Access row i
                if elem2d_nodes_unsort[i, 0] == elem2d_nodes_unsort[i, 3]:
                    elem_tri[i] = True
                    elem2d_nodes[numtrian, :] = elem2d_nodes_unsort[i, :]
                    numtrian += 1

            for i in range(elem2d):
                # Phase 1 refactor: Access row i
                if elem2d_nodes_unsort[i, 0] != elem2d_nodes_unsort[i, 3]:
                    elem_tri[i] = False
                    elem2d_nodes[numtrian + numquads, :] = elem2d_nodes_unsort[i, :]
                    numquads += 1

            return elem2d, elem2d_nodes, elem_tri, numtrian, numquads

    @staticmethod
    def test_elements_in_mesh(cyclic_length: float, elem2d: int,
                              elem2d_nodes: np.ndarray, elem_tri: np.ndarray,
                              coord_nod2d: np.ndarray) -> np.ndarray:
        """
        Test and rotate elements in the mesh if necessary.

        Args:
            cyclic_length (float): Length of the cyclic boundary.
            elem2d (int): Number of elements.
            elem2d_nodes (np.ndarray): Nodes of each element. Shape: (elem2d, 4)
            elem_tri (np.ndarray): Triangle flags for elements.
            coord_nod2d (np.ndarray): Coordinates of nodes. Shape: (nod2d, 2)

        Returns:
            np.ndarray: Updated element nodes.
        """
        rotatednodes = []
        numrotation = 0

        for i in range(elem2d):
            # Phase 3 refactor: Access row i for element nodes
            elnodes = elem2d_nodes[i, :].copy()
            # Phase 3 refactor: Access coord_nod2d[node, :] for row-major layout
            a = coord_nod2d[elnodes[0], :]
            b = coord_nod2d[elnodes[1], :] - a
            c = coord_nod2d[elnodes[2], :] - a
            d = coord_nod2d[elnodes[3], :] - a

            # Adjust for cyclic boundary
            for vec in [b, c, d]:
                if vec[0] > cyclic_length / 2.0:
                    vec[0] -= cyclic_length
                elif vec[0] < -cyclic_length / 2.0:
                    vec[0] += cyclic_length

            r = b[0] * c[1] - b[1] * c[0]

            if elem_tri[i]:
                # Triangle case
                if r > 0:
                    numrotation += 1
                    rotatednodes.append(i)
                    elnodes[1], elnodes[2] = elnodes[2], elnodes[1]
                    # Phase 3 refactor: Store at row i
                    elem2d_nodes[i, :] = elnodes
            else:
                # Quadrilateral case
                r1 = b[0] * d[1] - b[1] * d[0]
                if r1 * r < 0:
                    raise ValueError(f"Node list problem, node={i}")
                if r > 0:
                    numrotation += 1
                    rotatednodes.append(i)
                    elnodes[1], elnodes[3] = elnodes[3], elnodes[1]
                    # Phase 3 refactor: Store at row i
                    elem2d_nodes[i, :] = elnodes

        if numrotation > 0:
            logger.warning(f"Nodes needed to be rotated: total {numrotation} rotations.")
            logger.warning(f"List of rotated elements: {rotatednodes}")

        return elem2d_nodes

    @staticmethod
    def read_depth_from_file(filename: str, nod2d: int) -> np.ndarray:
        """
        Read depth data from a file.

        Args:
            filename (str): Path to the depth file.
            nod2d (int): Number of nodes.

        Returns:
            np.ndarray: Depth values per node.
        """
        depth = np.empty(nod2d, dtype=float)
        with open(filename, 'r') as file:
            for i in range(nod2d):
                line = file.readline().strip()
                depth[i] = float(line)
        return depth

    @staticmethod
    def find_edges(
        nod2d: int,
        elem2d: int,
        elem2d_nodes: np.ndarray,
        coord_nod2d: np.ndarray,
        cyclic_length: float
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        int, np.ndarray, np.ndarray, int, np.ndarray
    ]:
        """
        Find edges and related information in the mesh using Numba-accelerated kernels.

        Parameters:
        - nod2d: Number of nodes in 2D.
        - elem2d: Number of elements in 2D.
        - elem2d_nodes: 2D NumPy array of shape (elem2d, 4) containing node indices for each element.
                        For triangles, the fourth node should be equal to the first node.
        - coord_nod2d: 2D NumPy array of shape (nod2d, 2) containing coordinates for each node.
        - cyclic_length: The length used for cyclic boundary conditions.

        Returns:
        - ne_num: 1D NumPy array of shape (nod2d,) containing the number of neighboring elements per node.
        - ne_pos: 2D NumPy array of shape (nod2d, k) containing positions of neighboring elements.
        - nn_num: 1D NumPy array of shape (nod2d,) containing the number of neighboring nodes per node.
        - nn_pos: 2D NumPy array of shape (nod2d, maxnn) containing positions of neighboring nodes.
        - edge2d: Total number of edges.
        - edge_nodes: 2D NumPy array of shape (edge2d, 2) containing node pairs for each edge.
        - edge_tri: 2D NumPy array of shape (edge2d, 2) containing element pairs for each edge.
        - edge2d_in: Number of internal edges.
        - elem_edges: 2D NumPy array of shape (elem2d, 4) mapping elements to their edges.
        """
        # Ensure int32 for Numba compatibility
        elem2d_nodes_i32 = elem2d_nodes.astype(np.int32)

        ##################### part A ##########################
        # (a) Build node-to-element adjacency using Numba kernels

        # First pass: count elements per node to determine k
        ne_num_temp = _numba_count_node_elements(nod2d, elem2d, elem2d_nodes_i32)
        k = int(np.max(ne_num_temp))

        # Build ne_pos with Numba
        ne_num, ne_pos = _numba_build_ne_pos(nod2d, elem2d, elem2d_nodes_i32, k)

        # Count neighbor nodes to determine maxnn
        nn_num_temp = _numba_count_neighbor_nodes(nod2d, elem2d, elem2d_nodes_i32, ne_num, ne_pos)
        maxnn = int(np.max(nn_num_temp)) + 1

        # Build nn_pos with Numba
        nn_num, nn_pos = _numba_build_nn_pos(nod2d, elem2d, elem2d_nodes_i32, ne_num, ne_pos, maxnn)

        ##################### part B ##########################
        # (b) Find edges and elements containing them using Numba

        # Count edges
        edge2d = _numba_count_edges(nod2d, nn_num, nn_pos)

        # Find edges (internal and boundary)
        edge_nodes, edge_tri, edge2d_in = _numba_find_edges(
            nod2d, elem2d, elem2d_nodes_i32, ne_num, ne_pos, nn_num, nn_pos, edge2d
        )

        ##################### part D ##########################
        # (d) Orient edges so that the first element is to the left.
        # This part uses elem_center and geometry, so we keep it in Python.

        for n in range(edge2d):
            ed = edge_nodes[n, :]
            elem = edge_tri[n, 0]
            x, y, s = RiseMesh.elem_center(elem, elem2d_nodes, coord_nod2d, cyclic_length)
            xc_array = np.array([x - coord_nod2d[ed[0], 0], y - coord_nod2d[ed[0], 1]])
            xe = coord_nod2d[ed[1], :] - coord_nod2d[ed[0], :]

            if xe[1] > cyclic_length / 2:
                xe[1] -= cyclic_length
            if xe[1] < -cyclic_length / 2:
                xe[1] += cyclic_length
            if xc_array[0] > cyclic_length / 2:
                xc_array[0] -= cyclic_length
            if xc_array[0] < -cyclic_length / 2:
                xc_array[0] += cyclic_length

            # Calculate cross product: xc x xe
            cross = xc_array[0] * xe[1] - xc_array[1] * xe[0]
            if cross > 0.0:
                # Swap elements
                elem = edge_tri[n, 0]
                elem1 = edge_tri[n, 1]
                if elem1 > 0:
                    edge_tri[n, 0] = elem1
                    edge_tri[n, 1] = elem
                else:
                    # Swap the order of nodes in edge_nodes
                    temp_node = edge_nodes[n, 1]
                    edge_nodes[n, 1] = edge_nodes[n, 0]
                    edge_nodes[n, 0] = temp_node

        ##################### part E ##########################
        # (e) Build element-to-edge mapping using Numba

        elem_edges = _numba_build_elem_edges(edge2d, elem2d, edge_tri, edge_nodes, elem2d_nodes_i32)

        return ne_num, ne_pos, nn_num, nn_pos, edge2d, edge_nodes, edge_tri, edge2d_in, elem_edges     
 
    @staticmethod
    def find_elem_neighbors(
        nod2d: int,
        elem2d: int,
        elem_edges: np.ndarray,
        edge_tri: np.ndarray,
        elem2d_nodes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find neighbors for each element using Numba-accelerated kernels.

        Args:
            nod2d (int): Number of nodes.
            elem2d (int): Number of elements.
            elem_edges (np.ndarray): Element edges. Shape (elem2d, 4).
            edge_tri (np.ndarray): Edge triangles. Shape (edge2d, 2).
            elem2d_nodes (np.ndarray): Nodes of each element. Shape (elem2d, 4).

        Returns:
            Tuple containing:
                - elem_neighbors (np.ndarray): Neighbors for each element. Shape (elem2d, 4).
                - nod_in_elem2d_num (np.ndarray): Number of elements per node. Shape (nod2d,).
                - nod_in_elem2d (np.ndarray): Elements per node. Shape (nod2d, max_elements_per_node).
        """
        # Ensure int32 for Numba compatibility
        elem_edges_i32 = elem_edges.astype(np.int32)
        edge_tri_i32 = edge_tri.astype(np.int32)
        elem2d_nodes_i32 = elem2d_nodes.astype(np.int32)

        # Find element neighbors using Numba
        elem_neighbors = _numba_find_elem_neighbors(elem2d, elem_edges_i32, edge_tri_i32, elem2d_nodes_i32)

        # Build node-to-element mapping using Numba
        nod_in_elem2d_num, nod_in_elem2d = _numba_build_nod_in_elem2d(nod2d, elem2d, elem2d_nodes_i32)

        return elem_neighbors, nod_in_elem2d_num, nod_in_elem2d

    @staticmethod
    def mesh_arrays1(nod2d: int, elem2d: int, elem2d_nodes: np.ndarray, coord_nod2d: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize mesh-related arrays using JAX-accelerated geometry calculations.

        Args:
            nod2d (int): Number of nodes.
            elem2d (int): Number of elements.
            elem2d_nodes (np.ndarray): Nodes of each element. Shape: (elem2d, 4)
            coord_nod2d (np.ndarray): Coordinates of nodes. Shape: (nod2d, 2)
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            Tuple containing:
                - elem_area (np.ndarray): Area of each element.
                - area (np.ndarray): Area metric per node.
                - metric_factor (np.ndarray): Metric factors.
                - elem_cos (np.ndarray): Cosine values per element.
                - coriolis (np.ndarray): Coriolis parameters.
                - w_cv (np.ndarray): Weight coefficients. Shape: (elem2d, 4)
        """
        omega = 2.0 * np.pi / (3600.0 * 24.0)  # Coriolis parameter
        r_earth = 6400000.0
        cyclic_length = config["mesh"]["cyclic_length"]

        # Convert to JAX arrays for vectorized computation
        elem2d_nodes_jax = jnp.array(elem2d_nodes)
        coord_nod2d_jax = jnp.array(coord_nod2d)

        # Compute all element centers and areas in one vectorized call
        x_all, y_all, elem_area_jax = _jax_compute_all_elem_centers(elem2d_nodes_jax, coord_nod2d_jax, cyclic_length)

        # Vectorized computation of coriolis, elem_cos, metric_factor
        coriolis_jax = 2.0 * omega * jnp.sin(y_all)
        elem_cos_jax = jnp.cos(y_all)
        metric_factor_jax = jnp.tan(y_all) / r_earth

        # Convert back to NumPy
        elem_area = np.array(elem_area_jax)
        elem_cos = np.array(elem_cos_jax)
        metric_factor = np.array(metric_factor_jax)
        coriolis = np.array(coriolis_jax)

        if config["mesh"]["is_cartesian"]:
            elem_cos[:] = 1.0
            metric_factor[:] = 0.0

        if config["setup"]["is_coriolis"]:
            if config["mesh"]["is_cartesian"]:
                coriolis[:] = 2.0 * omega * np.sin(config["setup"]["lat_cartesian"] * np.pi / 180.0)
        else:
            coriolis[:] = 0.0

        elem_area *= elem_cos

        # Area and w_cv computation (still uses loops due to node accumulation)
        area = np.zeros(nod2d, dtype=float)
        w_cv = np.zeros((elem2d, 4), dtype=float)

        # Get element center coordinates as NumPy for the second loop
        x_all_np = np.array(x_all)
        y_all_np = np.array(y_all)

        for elem in range(elem2d):
            elnodes = elem2d_nodes[elem, :]
            if elnodes[0] == elnodes[3]:  # Triangle
                for j in range(1, 4):
                    area[elnodes[j]] += elem_area[elem] / 3.0
                    w_cv[elem, j - 1] = 1.0 / 3.0
                continue

            # Quadrilateral case
            rc_x, rc_y = x_all_np[elem], y_all_np[elem]
            elnodes_extended = np.zeros(6, dtype=int)
            elnodes_extended[0] = elnodes[3]
            elnodes_extended[1:5] = elnodes
            elnodes_extended[5] = elnodes[0]

            for j in range(1, 5):
                rdiff = RiseMesh.cdiff(coord_nod2d[elnodes_extended[j - 1], :], coord_nod2d[elnodes_extended[j], :], cyclic_length)
                rl = RiseMesh.cdiff(coord_nod2d[elnodes_extended[j + 1], :], coord_nod2d[elnodes_extended[j], :], cyclic_length)
                r0 = RiseMesh.cdiff(np.array([rc_x, rc_y]), coord_nod2d[elnodes_extended[j], :], cyclic_length)
                area_contribution = 0.25 * elem_cos[elem] * (abs(rl[0] * r0[1] - rl[1] * r0[0]) + abs(rdiff[0] * r0[1] - rdiff[1] * r0[0]))
                area[elnodes_extended[j]] += area_contribution
                w_cv[elem, j - 1] = area_contribution / elem_area[elem]

        # Update to proper dimension
        elem_area *= r_earth**2
        area *= r_earth**2

        return elem_area, area, metric_factor, elem_cos, coriolis, w_cv

    @staticmethod
    def mesh_arrays2(nod2d: int, elem2d: int, edge2d: int, coord_nod2d: np.ndarray,
                     elem2d_nodes: np.ndarray, elem_edges: np.ndarray, edge_tri: np.ndarray,
                     edge_nodes: np.ndarray, elem_cos: np.ndarray, elem_area: np.ndarray,
                     elem_neighbors: np.ndarray, cyclic_length: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize additional mesh-related arrays using JAX-accelerated computations.

        Args:
            nod2d (int): Number of nodes.
            elem2d (int): Number of elements.
            edge2d (int): Number of edges.
            coord_nod2d (np.ndarray): Coordinates of nodes. Shape: (nod2d, 2)
            elem2d_nodes (np.ndarray): Nodes of each element. Shape: (elem2d, 4)
            elem_edges (np.ndarray): Element edges. Shape: (elem2d, 4)
            edge_tri (np.ndarray): Edge triangles. Shape: (edge2d, 2)
            edge_nodes (np.ndarray): Edge nodes. Shape: (edge2d, 2)
            elem_cos (np.ndarray): Cosine values per element.
            elem_area (np.ndarray): Area of each element.
            elem_neighbors (np.ndarray): Element neighbors. Shape: (elem2d, 4)
            cyclic_length (float): Length of the cyclic boundary.

        Returns:
            Tuple containing:
                - edge_dxdy (np.ndarray): Edge derivatives. Shape: (edge2d, 2)
                - edge_cross_dxdy (np.ndarray): Cross derivatives for edges. Shape: (edge2d, 4)
                - gradient_sca (np.ndarray): Scalar gradients. Shape: (elem2d, 8)
                - gradient_vec (np.ndarray): Vector gradients. Shape: (elem2d, 8)
        """
        r_earth = 6400000.0

        # Convert to JAX arrays for vectorized computation
        coord_nod2d_jax = jnp.array(coord_nod2d)
        edge_nodes_jax = jnp.array(edge_nodes)
        elem2d_nodes_jax = jnp.array(elem2d_nodes)

        # Vectorized edge_dxdy computation using JAX
        @jit
        def compute_edge_dxdy(edge_nodes_arr, coord_arr, cyc_len):
            def single_edge(ed):
                a = coord_arr[ed[1], :] - coord_arr[ed[0], :]
                a_x = a[0]
                a_x = jnp.where(a_x > cyc_len / 2.0, a_x - cyc_len, a_x)
                a_x = jnp.where(a_x < -cyc_len / 2.0, a_x + cyc_len, a_x)
                return jnp.array([a_x, a[1]])
            return vmap(single_edge)(edge_nodes_arr)

        edge_dxdy_jax = compute_edge_dxdy(edge_nodes_jax, coord_nod2d_jax, cyclic_length)
        edge_dxdy = np.array(edge_dxdy_jax)

        # Cross-distances for edges (still uses loop due to conditional elem_center calls)
        edge_cross_dxdy = np.zeros((edge2d, 4), dtype=float)

        # Pre-compute all element centers for faster lookup
        x_all, y_all, _ = _jax_compute_all_elem_centers(elem2d_nodes_jax, coord_nod2d_jax, cyclic_length)
        x_all_np = np.array(x_all)
        y_all_np = np.array(y_all)

        for n in range(edge2d):
            ed = edge_nodes[n, :]
            el = edge_tri[n, :].copy()

            b = np.zeros(2, dtype=float)
            if el[0] >= 0:
                b[0], b[1] = x_all_np[int(el[0])], y_all_np[int(el[0])]
                a = np.array(RiseMesh.edge_center(ed[0], ed[1], coord_nod2d, cyclic_length))
                b -= a
                b = RiseMesh.cdiff(b, np.zeros(2), cyclic_length)
                b[0] *= elem_cos[int(el[0])]
                b *= r_earth
                edge_cross_dxdy[n, 0:2] = b

                if el[1] >= 0:
                    b[0], b[1] = x_all_np[int(el[1])], y_all_np[int(el[1])]
                    b -= a
                    b = RiseMesh.cdiff(b, np.zeros(2), cyclic_length)
                    b[0] *= elem_cos[int(el[1])]
                    b *= r_earth
                    edge_cross_dxdy[n, 2:4] = b
                else:
                    edge_cross_dxdy[n, 2:4] = 0.0
            else:
                edge_cross_dxdy[n, :] = 0.0

        # Compute gradients (still uses loop for mixed tri/quad)
        gradient_sca = np.zeros((elem2d, 8), dtype=float)
        gradient_vec = np.zeros((elem2d, 8), dtype=float)

        for elem in range(elem2d):
            elnodes = elem2d_nodes[elem, :]
            if elnodes[0] == elnodes[3]:  # Triangle
                deltaX31 = coord_nod2d[elnodes[2], 0] - coord_nod2d[elnodes[0], 0]
                deltaX31 = (deltaX31 + cyclic_length / 2.0) % cyclic_length - cyclic_length / 2.0
                deltaX31 *= elem_cos[elem]

                deltaX21 = coord_nod2d[elnodes[1], 0] - coord_nod2d[elnodes[0], 0]
                deltaX21 = (deltaX21 + cyclic_length / 2.0) % cyclic_length - cyclic_length / 2.0
                deltaX21 *= elem_cos[elem]

                deltaY31 = coord_nod2d[elnodes[2], 1] - coord_nod2d[elnodes[0], 1]
                deltaY21 = coord_nod2d[elnodes[1], 1] - coord_nod2d[elnodes[0], 1]

                dfactor = -0.5 * r_earth / elem_area[elem]
                gradient_sca[elem, 0] = (-deltaY31 + deltaY21) * dfactor
                gradient_sca[elem, 1] = deltaY31 * dfactor
                gradient_sca[elem, 2] = -deltaY21 * dfactor
                gradient_sca[elem, 3] = 0.0
                gradient_sca[elem, 4] = (deltaX31 - deltaX21) * dfactor
                gradient_sca[elem, 5] = -deltaX31 * dfactor
                gradient_sca[elem, 6] = deltaX21 * dfactor
                gradient_sca[elem, 7] = 0.0
            else:  # Quadrilateral
                enodes = np.zeros(6, dtype=int)
                enodes[1:5] = elnodes
                enodes[0] = elnodes[3]
                enodes[5] = elnodes[0]
                dfactor = 0.5 * r_earth / elem_area[elem]
                gradient_sca[elem, :] = 0.0
                for j in range(1, 5):
                    x1 = coord_nod2d[enodes[j], :]
                    x2 = coord_nod2d[enodes[j - 1], :]
                    xd = RiseMesh.cdiff(x1, x2, cyclic_length)
                    x3 = coord_nod2d[enodes[j + 1], :]
                    xd += RiseMesh.cdiff(x3, x1, cyclic_length)
                    xd[0] *= elem_cos[elem]
                    gradient_sca[elem, j - 1] = -xd[1] * dfactor
                    gradient_sca[elem, j - 1 + 4] = xd[0] * dfactor

        # Derivatives of vector quantities (Least squares interpolation)
        for elem in range(elem2d):
            a_x, a_y = x_all_np[elem], y_all_np[elem]
            q = 3 if elem2d_nodes[elem, 0] == elem2d_nodes[elem, 3] else 4

            x = np.zeros(4, dtype=float)
            y = np.zeros(4, dtype=float)

            for j in range(q):
                el = elem_neighbors[elem, j]
                if el >= 0:
                    b_x, b_y = x_all_np[el], y_all_np[el]
                    x[j] = b_x - a_x
                    x[j] = (x[j] + cyclic_length / 2.0) % cyclic_length - cyclic_length / 2.0
                    y[j] = b_y - a_y
                else:
                    # Virtual element center
                    ed = elem_edges[elem, j]
                    b_x, b_y = RiseMesh.edge_center(edge_nodes[ed, 0], edge_nodes[ed, 1], coord_nod2d, cyclic_length)
                    x[j] = (b_x - a_x)
                    x[j] = (x[j] + cyclic_length / 2.0) % cyclic_length - cyclic_length / 2.0
                    x[j] *= 2.0
                    y[j] = 2.0 * (b_y - a_y)

            if q == 3:
                x[3] = 0.0
                y[3] = 0.0

            x *= elem_cos[elem] * r_earth
            y *= r_earth
            cxx = np.sum(x**2)
            cxy = np.sum(x * y)
            cyy = np.sum(y**2)
            d = cxy**2 - cxx * cyy
            if d == 0:
                gradient_vec[elem, :] = 0.0
            else:
                gradient_vec[elem, 0:4] = (cxy * y - cyy * x) / d
                gradient_vec[elem, 4:8] = (cxy * x - cxx * y) / d

        return edge_dxdy, edge_cross_dxdy, gradient_sca, gradient_vec




    def compare_dumped_array(var_name, py_array, rtol=1e-5, atol=1e-8, add_one=0, mask_zero=False):
        import os
        import numpy as np        
        """
        Reads a dumped array from a Fortran-generated ASCII file and compares it to a given Python array.
        
        Parameters
        ----------
        var_name : str
            The name of the variable; the dumped file is assumed to be 'debug_dump_<var_name>.dat'
            located in the directory '/Users/ikuznets/work/projects/rise/fesom-c/run/'.
        py_array : np.ndarray
            The Python array to compare.
        rtol : float, optional
            Relative tolerance for floating point comparison (default is 1e-5).
        atol : float, optional
            Absolute tolerance for floating point comparison (default is 1e-8).
        add_one : int or bool, optional
            If nonzero/True, add 1 to all values in the Python array before comparing,
            except for those that are exactly -999 (they remain unchanged). (default is 0).
        mask_zero : bool, optional
            If True, positions where the Fortran file array is 0 will be ignored in the comparison 
            (default is False).
        
        Returns
        -------
        bool
            True if arrays have the same shape and content (after any modifications), False otherwise.
        """
        # Construct the file name based on the variable name.
        file_name = f"debug_dump_{var_name}.dat"
        file_name = os.path.join(os.path.dirname('/Users/ikuznets/work/projects/rise/fesom-c/run/'), file_name)
        
        try:
            # Use integer type if the Python array is of an integer type, otherwise use float.
            if np.issubdtype(py_array.dtype, np.integer):
                loaded_array = np.loadtxt(file_name, dtype=int)
            else:
                loaded_array = np.loadtxt(file_name, dtype=float)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            return False

        # Sometimes np.loadtxt returns a 1D array when Fortran produced a 2D array.
        if loaded_array.ndim == 1 and py_array.ndim == 2:
            # If Python array has one column and the number of rows match, reshape.
            if py_array.shape[1] == 1 and loaded_array.shape[0] == py_array.shape[0]:
                loaded_array = loaded_array.reshape(py_array.shape)
        
        # Compare shapes.
        if loaded_array.shape != py_array.shape:
            print(f"Shape mismatch: dumped array shape {loaded_array.shape} != python array shape {py_array.shape}")
            return False

        # If add_one is true, add 1 to the Python array except for values equal to -999.
        if add_one:
            py_array = np.where(py_array == -999, py_array, py_array + 1)

        # If mask_zero is true, compare only positions where the dumped array is nonzero.
        if mask_zero:
            mask = loaded_array != 0
            if not np.any(mask):
                print("No non-zero elements in dumped array, nothing to compare.")
                return True
            # Compare only the values where loaded_array is nonzero.
            if np.issubdtype(py_array.dtype, np.floating):
                comparison = np.allclose(loaded_array[mask], py_array[mask], rtol=rtol, atol=atol)
            else:
                comparison = np.array_equal(loaded_array[mask], py_array[mask])
        else:
            # Full comparison.
            if np.issubdtype(py_array.dtype, np.floating):
                comparison = np.allclose(loaded_array, py_array, rtol=rtol, atol=atol)
            else:
                comparison = np.array_equal(loaded_array, py_array)
        
        if comparison:
            print("Arrays match!")
        else:
            print("Arrays do not match!")
        
        return comparison

    @classmethod
    def read_mesh_from_netcdf(cls, netcdf_file: str, compute_derived: bool = False, config: Dict[str, Any] = None) -> 'RiseMesh':
        """
        Read the mesh from a NetCDF file.

        Args:
            netcdf_file: Path to NetCDF file containing mesh data
            compute_derived:  If True, compute derived quantities if not in NetCDF file(nv, nod_in_elem2d, nod_in_elem2d_num, ...)

        Returns:
            RiseMesh: Fully populated mesh instance

        Example:
            >>> mesh = RiseMesh.read_mesh_from_netcdf("./mesh/fesom_mesh.nc")
        """
        import xarray as xr

        # Extract configuration
        if netcdf_file is None:
            raise ValueError("netcdf_file is required for NetCDF mesh reading")
        if config is None:
            config = {'mesh': {},'setup': {}}
            config["mesh"]["cyclic_length"]=360.
            config["mesh"]["is_cartesian"]=False
            config["setup"]["is_coriolis"]=False
            config["mesh"]["meshpath"]=netcdf_file

        print(f"Reading mesh from NetCDF: {netcdf_file}")

        # Open NetCDF file
        ds = xr.open_dataset(netcdf_file)

        # Create mesh instance
        mesh_instance = cls.__new__(cls)

        # ========================================
        # CORE FIELDS (required)
        # ========================================

        # Number of nodes and elements
        mesh_instance.nod2d = ds.sizes['node']
        mesh_instance.elem2d = ds.sizes['nele']

        # Phase 1 refactor: Node coordinates as (nod2d, 2) for row-major layout
        # Each row is [lon, lat]
        mesh_instance.coord_nod2d = np.column_stack([
            ds['lon'].values,
            ds['lat'].values
        ])

        # Phase 1 refactor: Element connectivity as (elem2d, 4) for row-major layout
        # NetCDF 'nv' is typically (maxnod, nele) in Fortran order, transpose to (nele, maxnod)
        # Convert from 1-based to 0-based indexing
        nv_data = ds['nv'].values  # Shape: (maxnod, nele) or (nele, maxnod)

        # Check if transpose is needed
        if nv_data.shape[0] == 4 and nv_data.shape[1] != 4:
            # Fortran-style (4, nele) -> transpose to (nele, 4)
            mesh_instance.elem2d_nodes = nv_data.T - 1
        else:
            # Already in correct shape (nele, 4)
            mesh_instance.elem2d_nodes = nv_data - 1

        # ========================================
        # OPTIONAL FIELDS (from NetCDF if present)
        # ========================================

        # Filenames (set to NetCDF file)
        mesh_instance.nodes_filename = netcdf_file
        mesh_instance.elem_filename = netcdf_file

        # Depth
        if 'depth' in ds:
            mesh_instance.depth = ds['depth'].values

        # Node area (Scv)
        if 'area' in ds:
            mesh_instance.area = ds['area'].values

        # Element area
        if 'elem_area' in ds:
            mesh_instance.elem_area = ds['elem_area'].values

        # Phase 1 refactor: w_cv (control volume weights) as (elem2d, 4)
        if 'w_cv' in ds:
            w_cv_data = ds['w_cv'].values
            # Ensure shape is (elem2d, 4)
            if w_cv_data.shape[0] == 4 and w_cv_data.shape[1] != 4:
                mesh_instance.w_cv = w_cv_data.T
            else:
                mesh_instance.w_cv = w_cv_data

        # Phase 1 refactor: Node in element connectivity as (nod2d, max_elems)
        if 'nod_in_elem2d' in ds:
            # Convert from 1-based to 0-based indexing
            nod_in_elem_data = ds['nod_in_elem2d'].values - 1
            # Ensure shape is (nod2d, max_elems)
            if nod_in_elem_data.shape[1] == mesh_instance.nod2d:
                mesh_instance.nod_in_elem2d = nod_in_elem_data.T
            else:
                mesh_instance.nod_in_elem2d = nod_in_elem_data

        if 'nod_in_elem2d_num' in ds:
            mesh_instance.nod_in_elem2d_num = ds['nod_in_elem2d_num'].values

        # ========================================
        # DERIVED/COMPUTED FIELDS
        # ========================================

        # Determine element types (triangle vs quad)
        # Triangle: last column equals first column (degenerate quad)
        # Quad: last column != first column
        first_col = mesh_instance.elem2d_nodes[:, 0]
        last_col = mesh_instance.elem2d_nodes[:, -1]
        mesh_instance.elem_tri = (last_col == first_col).astype(np.int32)
        mesh_instance.numtrian = np.sum(mesh_instance.elem_tri)
        mesh_instance.numquads = mesh_instance.elem2d - mesh_instance.numtrian

        # TODO: Questions for user - these fields are not in NetCDF, should we compute them?
        # - nodes_obindex, nod2d_ob, nodes_ob, nodes_in (open boundary info)
        # - ne_num, ne_pos, nn_num, nn_pos (node-edge connectivity)
        # - edge2d, edge_nodes, edge_tri, edge2d_in, elem_edges (edge info)
        # - elem_neighbors (element neighbor connectivity)
        # - metric_factor, elem_cos (geometric factors)
        # - coriolis (Coriolis parameter)
        # - edge_dxdy, edge_cross_dxdy (edge geometry)
        # - gradient_sca, gradient_vec (gradient operators)

        if compute_derived:
            print("Computing derived quantities not present in NetCDF...")
            cyclic_length = config["mesh"]["cyclic_length"]

            # Find edges (if not in NetCDF)
            if mesh_instance.edge2d is None:
                print("  Computing edges...")
                ne_num, ne_pos, nn_num, nn_pos, edge2d, edge_nodes, edge_tri, edge2d_in, elem_edges = cls.find_edges(
                    mesh_instance.nod2d,
                    mesh_instance.elem2d,
                    mesh_instance.elem2d_nodes,
                    mesh_instance.coord_nod2d,
                    cyclic_length
                )
                mesh_instance.ne_num = ne_num
                mesh_instance.ne_pos = ne_pos
                mesh_instance.nn_num = nn_num
                mesh_instance.nn_pos = nn_pos
                mesh_instance.edge2d = edge2d
                mesh_instance.edge_nodes = edge_nodes
                mesh_instance.edge_tri = edge_tri
                mesh_instance.edge2d_in = edge2d_in
                mesh_instance.elem_edges = elem_edges

            # Find element neighbors (if not in NetCDF)
            if mesh_instance.elem_neighbors is None:
                print("  Computing element neighbors...")
                elem_neighbors, nod_in_elem2d_num, nod_in_elem2d = cls.find_elem_neighbors(
                    mesh_instance.nod2d,
                    mesh_instance.elem2d,
                    mesh_instance.elem_edges,
                    mesh_instance.edge_tri,
                    mesh_instance.elem2d_nodes
                )
                mesh_instance.elem_neighbors = elem_neighbors
                # Only set if not already loaded from NetCDF
                if mesh_instance.nod_in_elem2d_num is None:
                    mesh_instance.nod_in_elem2d_num = nod_in_elem2d_num
                if mesh_instance.nod_in_elem2d is None:
                    mesh_instance.nod_in_elem2d = nod_in_elem2d

            # Compute mesh arrays (if not in NetCDF)
            if mesh_instance.elem_area is None or mesh_instance.coriolis is None:
                print("  Computing mesh geometric arrays...")
                elem_area, area, metric_factor, elem_cos, coriolis, w_cv = cls.mesh_arrays1(
                    mesh_instance.nod2d,
                    mesh_instance.elem2d,
                    mesh_instance.elem2d_nodes,
                    mesh_instance.coord_nod2d,
                    config
                )
                if mesh_instance.elem_area is None:
                    mesh_instance.elem_area = elem_area
                if mesh_instance.area is None:
                    mesh_instance.area = area
                mesh_instance.metric_factor = metric_factor
                mesh_instance.elem_cos = elem_cos
                mesh_instance.coriolis = coriolis
                if mesh_instance.w_cv is None:
                    mesh_instance.w_cv = w_cv

            # Compute gradient operators (if not in NetCDF)
            if mesh_instance.gradient_sca is None or mesh_instance.gradient_vec is None:
                print("  Computing gradient operators...")
                edge_dxdy, edge_cross_dxdy, gradient_sca, gradient_vec = cls.mesh_arrays2(
                    mesh_instance.nod2d,
                    mesh_instance.elem2d,
                    mesh_instance.edge2d,
                    mesh_instance.coord_nod2d,
                    mesh_instance.elem2d_nodes,
                    mesh_instance.elem_edges,
                    mesh_instance.edge_tri,
                    mesh_instance.edge_nodes,
                    mesh_instance.elem_cos,
                    mesh_instance.elem_area,
                    mesh_instance.elem_neighbors,
                    cyclic_length
                )
                mesh_instance.edge_dxdy = edge_dxdy
                mesh_instance.edge_cross_dxdy = edge_cross_dxdy
                mesh_instance.gradient_sca = gradient_sca
                mesh_instance.gradient_vec = gradient_vec

        ds.close()

        logger.info(f"Mesh loaded from NetCDF: {mesh_instance.nod2d} nodes, {mesh_instance.elem2d} elements")
        logger.info(f"  {mesh_instance.numtrian} triangles, {mesh_instance.numquads} quadrilaterals")
        if compute_derived:
            logger.info("  All derived quantities computed")
        else:
            logger.info("  Only data from NetCDF loaded (use compute_derived=True for full mesh)")

        return mesh_instance

    @classmethod
    def read_mesh(cls, config: Dict[str, Any]) -> 'RiseMesh':
        """
        Read the mesh from ASCII files based on the configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.

        Returns:
            RiseMesh: An instance of the RiseMesh class populated with data.
        """
        cyclic_length = config["mesh"]["cyclic_length"]
        meshpath = config["mesh"]["meshpath"]
        print(" Reading nods")
        # Read nodes from file
        nodes_filename = f"{meshpath}nod2d.out"
        nod2d, coord_nod2d, nodes_obindex, nod2d_ob, nodes_ob, nodes_in = cls.read_nodes_from_file(nodes_filename, config)

        print(" Reading elements")
        # Read elements from file
        elem_filename = f"{meshpath}elem2d.out"
        elem2d, elem2d_nodes, elem_tri, numtrian, numquads = cls.read_elements_from_file(elem_filename)

        print(" Checking elements for clockwise rotation")
        # Check elements for clockwise rotation
        elem2d_nodes = cls.test_elements_in_mesh(cyclic_length, elem2d, elem2d_nodes, elem_tri, coord_nod2d)

        print(" Find edges")
        # Find edges
        ne_num, ne_pos, nn_num, nn_pos, edge2d, edge_nodes, edge_tri, edge2d_in, elem_edges = cls.find_edges(
            nod2d, elem2d, elem2d_nodes, coord_nod2d, cyclic_length
        )
        
        print(" Find element neighbors")
        # Find element neighbors
        elem_neighbors, nod_in_elem2d_num, nod_in_elem2d = cls.find_elem_neighbors(
            nod2d, elem2d, elem_edges, edge_tri, elem2d_nodes
        )
        print(" Initialize mesh arrays")
        # Initialize mesh arrays
        elem_area, area, metric_factor, elem_cos, coriolis, w_cv = cls.mesh_arrays1(
            nod2d, elem2d, elem2d_nodes, coord_nod2d, config
        )
        print(" Initialize mesh arrays 2")
        #cls.compare_dumped_array('elem_area',elem_area) #
        #cls.compare_dumped_array('area',area) #
        #cls.compare_dumped_array('metric_factor',metric_factor)
        #cls.compare_dumped_array('elem_cos',elem_cos) #
        #cls.compare_dumped_array('coriolis',coriolis)
        #cls.compare_dumped_array('w_cv',w_cv) 
        print(" Initialize mesh arrays 3")
        edge_dxdy, edge_cross_dxdy, gradient_sca, gradient_vec = cls.mesh_arrays2(
            nod2d, elem2d, edge2d, coord_nod2d, elem2d_nodes, elem_edges, edge_tri, edge_nodes,
            elem_cos, elem_area, elem_neighbors, cyclic_length
        )
        print(" Initialize mesh arrays 4")
        #cls.compare_dumped_array('edge_dxdy',edge_dxdy)
        #cls.compare_dumped_array('edge_cross_dxdy',edge_cross_dxdy)#
        #cls.compare_dumped_array('gradient_sca',gradient_sca)
        #cls.compare_dumped_array('gradient_vec',gradient_vec)
        
        print(" Read depth")
        # Read depth
        depth_filename = f"{meshpath}depth.out"
        depth = cls.read_depth_from_file(depth_filename, nod2d)

        # Logging mesh information
        logger.info(f"Mesh has been read: {nod2d} nodes and {elem2d} elements.")
        logger.info("Mesh has been read from files:")
        logger.info(f"  {nodes_filename}")
        logger.info(f"  {elem_filename}")
        logger.info(f"  {depth_filename}")
        logger.info(f"  The mesh includes {numtrian} triangles")
        logger.info(f"  The mesh includes {numquads} quadrilaterals")
        logger.info(f"  Number of open boundary nodes: {nod2d_ob}")
        logger.info("The list of elements starts from triangles.")
        logger.info(f"Coriolis max, min: {np.max(coriolis)}, {np.min(coriolis)}")
        logger.info(f"max_min_area_weight: {np.max(w_cv)}, {np.min(w_cv)}")
        logger.info("Mesh statistics:")
        logger.info(f"  maxArea: {np.max(elem_area)}, MinArea: {np.min(elem_area)}")
        logger.info(f"  maxScArea: {np.max(area)}, MinScArea: {np.min(area)}")
        logger.info(f"  Edges: {edge2d}, internal: {edge2d_in}")
        logger.info(f"Depth maximum: {np.max(depth)}, minimum: {np.min(depth)}.")

        # Create a Mesh instance with all fields set
        mesh_instance = cls.__new__(cls)  # Bypass __init__
        # Assign all attributes manually
        mesh_instance.nodes_filename = nodes_filename
        mesh_instance.nod2d = nod2d
        mesh_instance.coord_nod2d = coord_nod2d
        mesh_instance.nodes_obindex = nodes_obindex
        mesh_instance.nod2d_ob = nod2d_ob
        mesh_instance.nodes_ob = nodes_ob
        mesh_instance.nodes_in = nodes_in
        mesh_instance.elem_filename = elem_filename
        mesh_instance.elem2d = elem2d
        mesh_instance.elem2d_nodes = elem2d_nodes
        mesh_instance.elem_tri = elem_tri
        mesh_instance.numtrian = numtrian
        mesh_instance.numquads = numquads
        mesh_instance.depth = depth
        mesh_instance.ne_num = ne_num
        mesh_instance.ne_pos = ne_pos
        mesh_instance.nn_num = nn_num
        mesh_instance.nn_pos = nn_pos
        mesh_instance.edge2d = edge2d
        mesh_instance.edge_nodes = edge_nodes
        mesh_instance.edge_tri = edge_tri
        mesh_instance.edge2d_in = edge2d_in
        mesh_instance.elem_edges = elem_edges
        mesh_instance.elem_neighbors = elem_neighbors
        mesh_instance.nod_in_elem2d_num = nod_in_elem2d_num
        mesh_instance.nod_in_elem2d = nod_in_elem2d
        mesh_instance.elem_area = elem_area
        mesh_instance.area = area
        mesh_instance.metric_factor = metric_factor
        mesh_instance.elem_cos = elem_cos
        mesh_instance.coriolis = coriolis
        mesh_instance.w_cv = w_cv
        mesh_instance.edge_dxdy = edge_dxdy
        mesh_instance.edge_cross_dxdy = edge_cross_dxdy
        mesh_instance.gradient_sca = gradient_sca
        mesh_instance.gradient_vec = gradient_vec

        return mesh_instance

# Example Usage
if __name__ == "__main__":
    from mesh_manager import RiseMesh
    import yaml

    def read_config(config_path):
        """
        Reads a YAML configuration file and returns the content as a dictionary.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration as a dictionary.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
        return None


    config_path = "config.yml"  # Update with the path to your config file
    config = read_config(config_path)

    mesh = RiseMesh(config)