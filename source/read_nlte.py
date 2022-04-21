import numpy as np
import os
from scipy.interpolate import interp1d
from sys import exit
import pickle
import glob

def write_departures_forTS(fileName, tau, depart, abund):
    """
    Writes NLTE departure coefficients into the file compatible
    with TurboSpectrum

    Parameters
    ----------
    fileName : str
        name of the file in which to write the departure coefficients
    tau : np.array
        depth scale (e.g. TAU500nm)
    depart : np.ndarray
        departure coefficients
    abund : float
        chemical element abundance
    """

    ndep = len(tau)
    nk = len(depart)
    with open(fileName, 'w') as f:
        """  Comment lines below are requested by TS """
        for i in range(8):
            f.write('# parameter 1.0 1.0\n')

        f.write(f"{abund:.3f}\n")
        f.write(f"{ndep:.0f}\n")
        f.write(f"{nk:.0f}\n")
        for t in tau:
            f.write(F"{t:15.8E}\n")

        for i in range(ndep):
            f.write( f"{'  '.join(str(depart[j,i]) for j in range(nk))} \n" )


def read_departures_forTS(fileName):
    """
    Reads NLTE departure coefficients from the input file compatible
    with TurboSpectrum

    Parameters
    ----------
    fileName : str
        name of the file in which to write the departure coefficients
    tau : np.array
        depth scale (e.g. TAU500nm)
    depart : np.ndarray
        departure coefficients
    abund : float
        chemical element abundance
    """
    with open(fileName, 'r') as f:
        data = [ l for l in f.readlines() if not l.startswith('#') ]

    abund = float( data[0])
    ndep = int( data[1])
    nk = int( data[2] )
    
    tau = np.loadtxt(fileName, skiprows=11, max_rows = ndep)
    depart = np.loadtxt(fileName, skiprows=11+ndep).T
# TODO depart.T shoud be
    return abund, tau, depart

def read_binary_grid(grid_file, pointer=1):
    """
    Reads a record at specified position from the binary NLTE grid
    (grid of departure coefficients)

    Parameters
    ----------
    grid_file : str
        path to the binary NLTE grid
    pointer : int
        bitwise start of the record as read from the auxiliarly file

    Returns
    -------
    ndep : int
        number of depth points in the model atmosphere used to solve for NLTE
    nk : int
        number of energy levels in the model atom used to solved for NLTE
    depart : array
        NLTE departure coefficients of shape (ndep, nk)
    tau : array
        depth scale (e.g. TAU500nm) in the model atmosphere
        used to solve for NLTE of shape (ndep)
    """
    with open(grid_file, 'rb') as f:
        # -1 since Python stars at 0
        pointer = pointer - 1

        f.seek(pointer)
        atmosStr = f.readline(500)#.decode('utf-8', 'ignore').strip()
        ndep = int.from_bytes(f.read(4), byteorder='little')
        nk = int.from_bytes(f.read(4), byteorder='little')
        tau  = np.log10(np.fromfile(f, count = ndep, dtype='f8'))
        depart = np.fromfile(f, dtype='f8', count=ndep*nk).reshape(nk, ndep)

    return ndep, nk, depart, tau


def read_fullNLTE_grid(bin_file, aux_file, rescale=False, depthScale=None, saveMemory = False):
    """
    Reads the full binary NLTE grid of departure coefficients
    Note: binary grids can reach up to 100 Gb in size, therefore it is
    possible to only read half of the records
    randomly distributed across the grid

    Parameters
    ----------
    bin_file : str
        path to the binary NLTE grid
    aux_file : str
        path to the complementary auxilarly file
    rescale : boolean
        whether or not to bring the departure coefficients onto a depth
        scale different from the depth scale in the model atmosphere
        used in solving for NLTE. False by default
    depthScale : np.array
        depth scale to which departure coefficients will be rescaled
        if rescale is set to True
    saveMemory : boolean
        whether or not to save the memory by only reading random
        half of the records in the NLTE grid

    Returns
    -------
    data : dict
        contains NLTE departure coefficients as well as the depth scale of the
        model atmosphere and parameters used in RT
        (e.g. Teff, log(g) of model atmosphere)
    """
    if rescale and isinstance(depthScale, type(None)):
            raise Warning(f"To re-scale NLTE departure coefficients, \
please supply new depth scale.")

    aux = np.genfromtxt(aux_file, \
    dtype = [('atmos_id', 'S500'), ('teff','f8'), ('logg','f8'), \
            ('feh', 'f8'), ('alpha', 'f8'), ('mass', 'f8'), \
            ('vturb', 'f8'), ('abund', 'f8'), ('pointer', 'i8')])

    data = {}
    if saveMemory:
        pos = np.random.randint(0, len(aux['atmos_id']), \
                                size=int(len(aux['atmos_id'])/2))
        for k in aux.dtype.names:
            data.update( { k : aux[k][pos] })
    else:
        for k in aux.dtype.names:
            data.update( { k : aux[k] })
    """
    Get size of records from the first records
    assuming same size across the grid (i.e. model atmospheres had the same
    depth dimension and same model atom was used consistently)
    """
    p = data['pointer'][0]
    ndep, nk, depart, tau = read_binary_grid(bin_file, pointer=p)
    if rescale:
        departShape = ( len(data['pointer']), nk, len(depthScale))
    else:
        departShape = ( len(data['pointer']), nk, ndep)
    data.update( {
                'depart' : np.full(departShape, np.nan),
                'depthScale' : np.full((departShape[0],departShape[-1]), np.nan)
                } )
    for i in range(len( data['pointer'])):
        p = data['pointer'][i]
        ndep, nk, depart, tau = read_binary_grid(bin_file, pointer=p)
    # TODO! add comments and pass to the spectral output
        if rescale:
            f_int = interp1d(tau, depart, fill_value='extrapolate')
            depart = f_int(depthScale)
            tau = depthScale
        data['depart'][i] = depart
        data['depthScale'][i] = tau
    return data


def find_distance_to_point(point, grid):
    """

    Find the closest record in the NLTE grid to the supplied point
    based on quadratic distance.
    If several records are at the same distance
    (might happen if e.g. one abudnance was included twice),
    the first one is picked

    Parameters
    ----------
    point : dict
        coordinates of the input point. Only coordinates provided here will
        be used to compute the distance
    grid : dict
        NLTE grid of departure coefficients as read by read_fullNLTE_grid()

    Returns
    -------
    pos : int
        position of the closest point found in the grid
    comment : str
        notifies if more than one point at the minimum distance was found
        and which one was picked
    """
    dist = 0
    for k in point:
        dist += ((grid[k] - point[k])/max(grid[k]))**2
    dist = np.sqrt(dist)
    pos = np.where(dist == min(dist) )[0]
    if len(pos) > 1:
        comment = f"Found more than one 'closets' points to: \n"
        comment += '\n'.join(f"{k} = {point[k]}" for k in point) + '\n'
        comment += f"{grid['atmos_id'][pos]}\n"
        comment += f"Adopted departure coefficients \
at pointer = {grid['pointer'][pos[0]]}\n"
        return pos[0], comment
    else: return pos[0], ''
