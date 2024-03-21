from representation.rep_class import *
from representation.rep_utils import *
from representation.utils import *
from representation.generation import *
from representation.meshing import *
from autoencoder.autoencoder import *
from autoencoder.dataset import *
from local_test.example_materials import *

EPSILON = 1e-4

print("utils.py - triangle_line_intersection")
assert triangle_line_intersection(np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.zeros(3), np.ones(3))
print(".", end="")
assert triangle_line_intersection(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]), np.zeros(3), np.ones(3))
print(".", end="")
assert not triangle_line_intersection(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]), np.ones(3)*2, np.ones(3))
print(".", end="")
assert triangle_line_intersection(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]), np.array([0.1,0.1,1]), np.array([0.1,0.1,-1]))
print(".", end="")
assert not triangle_line_intersection(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,0]), np.array([1,0,0]))
print(".", end="")
assert not triangle_line_intersection(np.array([0,0,0]), np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([1,0,1]))
print(".", end="")
print()

print("rep_utils.py - euclidian_to_spherical")
expected = np.array([0.25, 0])
actual = np.array(euclidian_to_spherical(1,0,1))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
expected = np.array([np.arccos(3**(-0.5))/np.pi, 0.125])
actual = np.array(euclidian_to_spherical(1,1,1))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
expected = np.array([0.75, 0.5])
actual = np.array(euclidian_to_spherical(-1,0,-1))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
expected = np.zeros(2)
actual = np.array(euclidian_to_spherical(0,0,0))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
print()

print("rep_utils.py - spherical_to_euclidian")
expected = np.array([1,0,1])/2**0.5
actual = np.array(spherical_to_euclidian(np.pi/4,0))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
expected = np.array([1,1,1])/3**0.5
actual = np.array(spherical_to_euclidian(np.arccos(3**(-0.5)), np.pi/4))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
expected = np.array([-1,0,-1])/2**0.5
actual = np.array(spherical_to_euclidian(3*np.pi/4,np.pi))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
print()

print("rep_utils.py - project_onto_cube")
expected = np.array([1,0.5,1])
actual = np.array(project_onto_cube(*spherical_to_euclidian(np.pi/4,0)))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
expected = np.array([1,1,1])
actual = np.array(project_onto_cube(*spherical_to_euclidian(np.arccos(3**(-0.5)), np.pi/4)))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
expected = np.array([0,0.5,0])
actual = np.array(project_onto_cube(*spherical_to_euclidian(3*np.pi/4,np.pi)))
assert np.all(np.abs(actual-expected) < EPSILON), f"Got {actual}, but expected {expected}."
print(".", end="")
print()

NUM_NODES = 7
print("rep_utils.py - edge_adj_index")
expected = 0
actual = edge_adj_index(0,1)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = 5
actual = edge_adj_index(0,6)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = EDGE_ADJ_SIZE-1
actual = edge_adj_index(5,6)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = 13
actual = edge_adj_index(2,5)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = 13
actual = edge_adj_index(5,2)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
print()

print("rep_utils.py - face_adj_index")
expected = 0
actual = face_adj_index(0,1,2)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = 4
actual = face_adj_index(0,1,6)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = FACE_ADJ_SIZE-1
actual = face_adj_index(4,5,6)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = 15+4+1
actual = face_adj_index(1,3,5)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = 15+4+1
actual = face_adj_index(5,3,1)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
expected = 15+4+1
actual = face_adj_index(3,5,1)
assert expected == actual, f"Got {actual}, but expected {expected}."
print(".", end="")
print()