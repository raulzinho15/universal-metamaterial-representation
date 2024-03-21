import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[1]))

from representation.rep_class import *
from representation.rep_utils import *
from representation.utils import *
from representation.generation import *
from representation.meshing import *
from autoencoder.autoencoder import *
from autoencoder.dataset import *
from local_test.example_materials import *


print("utils.py triangle_line_intersection")
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