from matplotlib.colors import Colormap
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import utils
from .heatmaps import ValueGrid
from .heatmaps import color_canvas
from .functions import GaussFunction2D
from typing import *


def mapvalues2mol(mol: Chem.Mol,
                  atom_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
                  bond_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
                  atom_width: float = 0.3,
                  bond_width: float = 0.25,
                  bond_length: float = 0.5,
                  canvas: Optional[rdMolDraw2D.MolDraw2D] = None,
                  grid_resolution=None,
                  value_lims: Optional[Sequence[float]] = None,
                  color: Union[str, Colormap] = "bwr",
                  padding: Optional[Sequence[float]] = None) -> rdMolDraw2D:
    """A function to map weights of atoms and bonds to the drawing of a RDKit molecular depiction.

    For each atom and bond of depicted molecule a Gauss-function, centered at the respective object, is created and
    scaled by the corresponding weight. Gauss-functions of atoms are circular, while Gauss-functions of bonds can be
    distorted along the bond axis. The value of each pixel is determined as the sum of all function-values at the pixel
    position. Subsequently the values are mapped to a color and drawn onto the canvas.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object which is displayed.
    atom_weights: Optional[Union[Sequence[float], np.ndarray]]
        Array of weights for atoms.
    bond_weights: Optional[Union[Sequence[float], np.ndarray]]
        Array of weights for bonds.
    atom_width: float
        Value for the width of displayed atom weights.
    bond_width: float
        Value for the width of displayed bond weights (perpendicular to bond-axis).
    bond_length: float
        Value for the length of displayed bond weights (along the bond-axis).
    canvas: Optional[rdMolDraw2D.MolDraw2D]
        RDKit canvas the molecule and heatmap are drawn onto.
    grid_resolution: Optional[Sequence[int]]
        Number of pixels of x- and y-axis.
    value_lims: Optional[Sequence[float]]
        Lower and upper limit of displayed values. Values exceeding limit are displayed as maximum (or minimum) value.
    color: Union[str, Colormap]
        Matplotlib colormap or string referring to a matplotlib colormap
    padding: Optional[Sequence[float]]
        Increase of heatmap size, relative to size of molecule. Usually the heatmap is increased by 100% in each axis
        by padding 50% in each side.

    Returns
    -------
    rdMolDraw2D.MolDraw2D
        Drawing of molecule and corresponding heatmap.
    """

    # Assigning default values
    if atom_weights is None:
        atom_weights = np.zeros(len(mol.GetAtoms()))

    if bond_weights is None:
        bond_weights = np.zeros(len(mol.GetBonds()))

    if not canvas:
        canvas = rdMolDraw2D.MolDraw2DCairo(800, 450)
        draw_opt = canvas.drawOptions()
        draw_opt.padding = 0.2
        draw_opt.bondLineWidth = 3
        canvas.SetDrawOptions(draw_opt)

    if grid_resolution is None:
        grid_resolution = [canvas.Width(), canvas.Height()]

    if padding is None:
        # Taking padding from DrawOptions
        draw_opt = canvas.drawOptions()
        padding = [draw_opt.padding * 2, draw_opt.padding * 2]

    # Validating input
    if not len(atom_weights) == len(mol.GetAtoms()):
        raise ValueError("len(atom_weights) is not equal to number of bonds in mol")

    if not len(bond_weights) == len(mol.GetBonds()):
        raise ValueError("len(bond_weights) is not equal to number of bonds in mol")

    # Setting up the grid
    xl, yl = utils.get_mol_lims(mol)  # Limit of molecule
    xl, yl = list(xl), list(yl)

    # Extent of the canvas is approximated by size of molecule scaled by ratio of canvas height and width.
    # Would be nice if this was directly accessible...
    mol_height = yl[1] - yl[0]
    mol_width = xl[1] - xl[0]

    height_to_width_ratio_mol = mol_height / mol_width
    height_to_width_ratio_canvas = canvas.Height() / canvas.Width()

    if height_to_width_ratio_mol < height_to_width_ratio_canvas:
        mol_height_new = canvas.Height() / canvas.Width() * mol_width
        yl[0] -= (mol_height_new - mol_height) / 2
        yl[1] += (mol_height_new - mol_height) / 2
    else:
        mol_width_new = canvas.Width() / canvas.Height() * mol_height
        xl[0] -= (mol_width_new - mol_width) / 2
        xl[1] += (mol_width_new - mol_width) / 2

    xl = utils.pad(xl, padding[0])  # Increasing size of x-axis
    yl = utils.pad(yl, padding[1])  # Increasing size of y-axis
    v_map = ValueGrid(xl, yl, grid_resolution[0], grid_resolution[1])

    conf = mol.GetConformer(0)

    # Adding Gauss-functions centered at atoms
    for i, _ in enumerate(mol.GetAtoms()):
        if atom_weights[i] == 0:
            continue
        pos = conf.GetAtomPosition(i)
        coords = pos.x, pos.y
        f = GaussFunction2D(center=coords, std1=atom_width, std2=atom_width, scale=atom_weights[i], rotation=0)
        v_map.add_function(f)

    # Adding Gauss-functions centered at bonds (position between the two bonded-atoms)
    for i, b in enumerate(mol.GetBonds()):  # type: Chem.Bond
        if bond_weights[i] == 0:
            continue
        a1 = b.GetBeginAtom().GetIdx()
        a1_pos = conf.GetAtomPosition(a1)
        a1_coords = np.array([a1_pos.x, a1_pos.y])

        a2 = b.GetEndAtom().GetIdx()
        a2_pos = conf.GetAtomPosition(a2)
        a2_coords = np.array([a2_pos.x, a2_pos.y])

        diff = a2_coords - a1_coords
        angle = np.arctan2(diff[0], diff[1])

        bond_center = (a1_coords + a2_coords) / 2

        f = GaussFunction2D(center=bond_center, std1=bond_width, std2=bond_length, scale=bond_weights[i],
                            rotation=angle)
        v_map.add_function(f)

    # Evaluating all functions at pixel positions to obtain pixel values
    v_map.evaluate()

    # Greating color-grid from the value grid.
    c_grid = v_map.map2color(color, v_lim=value_lims)
    # Drawing the molecule and erasing it to initialize the grid
    canvas.DrawMolecule(mol)
    canvas.ClearDrawing()
    # Adding the Colormap to the canvas
    color_canvas(canvas, c_grid)
    # Adding the molecule to the canvas
    canvas.DrawMolecule(mol)
    return canvas


def get_depiction_limits(mol: Chem.Mol,
                         atom_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
                         bond_weights: Optional[Union[Sequence[float], np.ndarray]] = None,
                         atom_width: float = 0.3,
                         bond_width: float = 0.25,
                         bond_length: float = 0.5,
                         canvas: Optional[rdMolDraw2D.MolDraw2D] = None,
                         grid_resolution=None,
                         padding: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """Dry run of `mapvalues2mol` in order to obtain value limits of depiction.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule object which is displayed.
    atom_weights: Optional[Union[Sequence[float], np.ndarray]]
        Array of weights for atoms.
    bond_weights: Optional[Union[Sequence[float], np.ndarray]]
        Array of weights for bonds.
    atom_width: float
        Value for the width of displayed atom weights.
    bond_width: float
        Value for the width of displayed bond weights (perpendicular to bond-axis).
    bond_length: float
        Value for the length of displayed bond weights (along the bond-axis).
    canvas: Optional[rdMolDraw2D.MolDraw2D]
        RDKit canvas the molecule and heatmap are drawn onto.
    grid_resolution: Optional[Sequence[int]]
        Number of pixels of x- and y-axis.
    padding: Optional[Sequence[float]]
        Increase of heatmap size, relative to size of molecule. Usually the heatmap is increased by 100% in each axis
        by padding 50% in each side.

    Returns
    -------
    Tuple[float, float]
        Value limits of depiction
    """
    if atom_weights is None:
        atom_weights = np.zeros(len(mol.GetAtoms()))

    if bond_weights is None:
        bond_weights = np.zeros(len(mol.GetBonds()))

    if not canvas:
        canvas = rdMolDraw2D.MolDraw2DCairo(800, 450)
        draw_opt = canvas.drawOptions()
        draw_opt.padding = 0.2
        draw_opt.bondLineWidth = 3
        canvas.SetDrawOptions(draw_opt)

    if grid_resolution is None:
        grid_resolution = [canvas.Width(), canvas.Height()]

    if padding is None:
        # Taking padding from DrawOptions
        draw_opt = canvas.drawOptions()
        padding = [draw_opt.padding * 2, draw_opt.padding * 2]

    # Validating input
    if not len(atom_weights) == len(mol.GetAtoms()):
        raise ValueError("len(atom_weights) is not equal to number of bonds in mol")

    if not len(bond_weights) == len(mol.GetBonds()):
        raise ValueError("len(bond_weights) is not equal to number of bonds in mol")

    # Setting up the grid
    xl, yl = utils.get_mol_lims(mol)  # Limit of molecule
    xl, yl = list(xl), list(yl)

    # Extent of the canvas is approximated by size of molecule scaled by ratio of canvas height and width.
    # Would be nice if this was directly accessible...
    mol_height = yl[1] - yl[0]
    mol_width = xl[1] - xl[0]

    height_to_width_ratio_mol = mol_height / mol_width
    height_to_width_ratio_canvas = canvas.Height() / canvas.Width()

    if height_to_width_ratio_mol < height_to_width_ratio_canvas:
        mol_height_new = canvas.Height() / canvas.Width() * mol_width
        yl[0] -= (mol_height_new - mol_height) / 2
        yl[1] += (mol_height_new - mol_height) / 2
    else:
        mol_width_new = canvas.Width() / canvas.Height() * mol_height
        xl[0] -= (mol_width_new - mol_width) / 2
        xl[1] += (mol_width_new - mol_width) / 2

    xl = utils.pad(xl, padding[0])  # Increasing size of x-axis
    yl = utils.pad(yl, padding[1])  # Increasing size of y-axis
    v_map = ValueGrid(xl, yl, grid_resolution[0], grid_resolution[1])

    conf = mol.GetConformer(0)

    # Adding Gauss-functions centered at atoms
    for i, _ in enumerate(mol.GetAtoms()):
        if atom_weights[i] == 0:
            continue
        pos = conf.GetAtomPosition(i)
        coords = pos.x, pos.y
        f = GaussFunction2D(center=coords, std1=atom_width, std2=atom_width, scale=atom_weights[i], rotation=0)
        v_map.add_function(f)

    # Adding Gauss-functions centered at bonds (position between the two bonded-atoms)
    for i, b in enumerate(mol.GetBonds()):  # type: Chem.Bond
        if bond_weights[i] == 0:
            continue
        a1 = b.GetBeginAtom().GetIdx()
        a1_pos = conf.GetAtomPosition(a1)
        a1_coords = np.array([a1_pos.x, a1_pos.y])

        a2 = b.GetEndAtom().GetIdx()
        a2_pos = conf.GetAtomPosition(a2)
        a2_coords = np.array([a2_pos.x, a2_pos.y])

        diff = a2_coords - a1_coords
        angle = np.arctan2(diff[0], diff[1])

        bond_center = (a1_coords + a2_coords) / 2

        f = GaussFunction2D(center=bond_center, std1=bond_width, std2=bond_length, scale=bond_weights[i],
                            rotation=angle)
        v_map.add_function(f)

    # Evaluating all functions at pixel positions to obtain pixel values
    v_map.evaluate()

    # Greating color-grid from the value grid.
    return v_map.values.min(), v_map.values.max()