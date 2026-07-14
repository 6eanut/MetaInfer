"""fusedmoe-evolve task plugin: 4-phase openevolve-driven optimization.

Each MetaInfer iteration runs one full OpenEvolve run:
  A_prepare -> B_evolve -> C_validate -> D_review -> A_prepare (next iter)
"""

from .. import register
from .plugin import PLUGIN

register(PLUGIN)
