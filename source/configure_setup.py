import numpy as np
import os
import shutil
from sys import argv, exit
import datetime
import glob
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
# local
from model_atm_interpolation import get_all_ma_parameters, NDinterpolateGrid,preInterpolationTests
from read_nlte import read_fullNLTE_grid, find_distance_to_point
from atmos_package import model_atmosphere
from read_nlte import write_departures_forTS, read_departures_forTS
import cProfile
import pstats
from chemical_elements import ChemElement

def gradient3rdOrder(f):
    for i in range(3):
        f = np.gradient(f, edge_order=2)
    return f

def in_hull(p, hull):
    """
    Is triangulation-based interpolation to this point possible?

    Parameters
    ----------
    p : dict
        point to test, contains coordinate names as keys and their values, e.g.
        p['teff'] = 5150
    hull :

    Returns
    -------
    whether point is inside the hull
    """
    return hull.find_simplex(p) >= 0

def restoreDepartScaling(depart, el):
    """
    Departure coefficients are normalised and brought to the log scale
    for the ease of interpolation.
    This functions brings them back to the initial units

    Parameters
    ----------
    depart : np.ndarray
        normalised departure coefficients
    el : ChemElement
        chemical element corresponding to the departure coeffcicients
        (scaling is the same for all departure coefficients of the same
        chemical element)

    Returns
    -------
    np.ndarray
        Departure coefficient in original units
        as read from the binary NLTE grid
    """
    return 10**(depart * el.DepartScaling)

def read_random_input_parameters(file):
    """
    Read strictly formatted input parameters
    Example of input file:
    ....
    Teff logg Vturb FeH Fe H O # Ba
    5535  2.91  0.0  -1.03  6.470  12.0   9.610 # 2.24
    7245  5.74  0.0  -0.50  7.000  12.0   8.009 # -2.2
    ....
    Input file must include Teff, logg, Vturb, and FeH

    Parameters
    ----------
    file : str
        path to the input file

    Returns
    -------
    input_par : dict
        contains parameters requested for spectral synthesis,
        both fundamental (e.g. Teff, log(g), [Fe/H], micro-turbulence)
        and individual chemical abundances

    freeParams : list
        parameters describing model atmosphere
    """
    data =[ l.split('#')[0] for l in open(file, 'r').readlines() \
                            if not (l.startswith('#') or l.strip()=='') ]
    header = data[0].replace("'","")

    freeParams = [ s.lower() for s in header.split()[:4] ]
    for k in ['teff', 'logg', 'vturb', 'feh']:
        if k not in freeParams:
            print(f"Could not find {k} in the input file {file}. \
Please check requested input.")
            exit()
            # TODO: better exceptions/error tracking
    values =  np.array( [ l.split() for l in data[1:] ] ).astype(float)
    input_par = {
                'teff':values[:, 0], 'logg':values[:, 1], 'vturb':values[:, 2],\
                'feh':values[:,3], 'elements' : {}
                }

    elIDs = header.split()[4:]
    for i in range(len(elIDs)):
        el = ChemElement(elIDs[i].capitalize())
        el.abund = values[:, i+4]
        input_par['elements'][elIDs[i]] =  el

    input_par['count'] = len(input_par['teff'])
    input_par['comments'] = np.full(input_par['count'], '', dtype='U5000')

    if 'Fe' not in input_par['elements']:
        print(f"WARNING: input contains [Fe/H], but no A(Fe).\
Setting A(Fe) to 7.5")
        el = ChemElement('Fe')
        el.abund = input_par['feh'] + 7.5
        input_par['elements'][el.ID] = el

    absAbundCheck = np.array([ el.abund / 12. for el in input_par['elements'].values() ])
    if (absAbundCheck < 0.0).any():
        print(f"Warning: abundances must be supplied relative to H, \
on log12 scale. Please double check input file '{file}'")
    # TODO: move free parameters as a sub-dictinary of the return
    return input_par, freeParams

class setup(object):
    """
    Describes the setup requested for computations

    Parameters
    ----------
    file : str
        path to the configuration file, './config.txt' by default

    """
    def __init__(self, file='./config.txt'):
        if 'cwd' not in self.__dict__.keys():
            self.cwd = f"{os.getcwd()}/"
        self.debug = 0
        self.ncpu  = 1
        self.nlte = 0
        self.safeMemory = 250 # Gb


        "Read all the keys from the config file"
        for line in open(file, 'r').readlines():
            line = line.strip()
            if not line.startswith('#') and len(line)>0:
                if not '+=' in line:
                    k, val = line.split('=')
                    k, val = k.strip(), val.strip()
                    if val.startswith("'") or val.startswith('"'):
                        self.__dict__[k] = val[1:-1]
                    elif val.startswith("["):
                        if '[' in val[1:]:
                            if not k in self.__dict__ or len(self.__dict__[k]) == 0:
                                self.__dict__[k] = []
                            self.__dict__[k].append(val)
                        else:
                            self.__dict__[k] = eval('np.array(' + val + ')')
                    elif '.' in val:
                        self.__dict__[k] = float(val)
                    else:
                        self.__dict__[k] = int(val)
                elif '+=' in line:
                    k, val = line.split('+=')
                    k, val = k.strip(), val.strip()
                    if len(self.__dict__[k]) == 0:
                        self.__dict__[k] = []
                    self.__dict__[k].append(val)

        if 'inputParams_file' in self.__dict__:
            self.inputParams, self.freeInputParams = read_random_input_parameters(self.inputParams_file)
        else:
            print("Missing file with input parameters: inputParams_file")

        if 'nlte_config' in self.__dict__:
            """ Read provided NLTE grids and model atoms
             and match to the elements requested in the input file"""
            for l in self.nlte_config:
                l = l.replace('[','').replace(']','').replace("'","")
                elID, files = l.split(':')[0].strip().capitalize(),\
                                [f.strip() for f in l.split(':')[-1].split(',')]
                if 'nlte_grids_path' in self.__dict__:
                    files = [ f"{self.nlte_grids_path.strip()}/{f}" for f in files]
                files = [ f.replace('./', self.cwd) if f.startswith('./') \
                                                    else f  for f in files]

                if (elID not in self.inputParams['elements']) and self.debug:
                    print(f"NLTE data is provided for {elID}, \
but it is not a free parameter in the input file {self.inputParams_file}.")
                else:
                    el = self.inputParams['elements'][elID]
                    el.nlte = True
                    el.nlteGrid = files[0]
                    el.nlteAux = files[1]
                    el.modelAtom = files[2]

            "TS needs to access model atoms from the same path for all elements"
            if 'modelAtomsPath' not in self.__dict__.keys():
                self.modelAtomsPath = f"{self.cwd}/modelAtoms_links/"
                if os.path.isdir(self.modelAtomsPath):
                    shutil.rmtree(self.modelAtomsPath)
                os.mkdir(self.modelAtomsPath)

                "Link provided model atoms to this directory"
                for el in self.inputParams['elements'].values():
                    if el.nlte:
                        dst = self.modelAtomsPath + el.modelAtom.split('/')[-1]
                        os.symlink(el.modelAtom, dst )

        """ Any element to be treated in NLTE eventually? """
        for el in self.inputParams['elements'].values():
            if el.nlte:
                self.nlte = True
                break

        if 'nlte_config' not in self.__dict__ or not self.nlte:
            print(f"{50*'*'}\n Note: all elements will be computed in LTE!\n \
To set up NLTE, use 'nlte_config' flag\n {50*'*'}")

        """ Create a directory to save spectra"""
        # TODO: all directory creation should be done at the same time
        today = datetime.date.today().strftime("%b-%d-%Y")
        self.spectraDir = self.cwd + f"/spectra-{today}/"
        if not os.path.isdir(self.spectraDir):
            os.mkdir(self.spectraDir)

# TODO: write model atmospheres to files Here
# than if no interpolation is required, those files can be passed directly to TS


        "Temporary directories for NLTE files"
        for el in self.inputParams['elements'].values():
            if el.nlte:
                el.departDir = self.cwd + f"/{el.ID}_nlteDepFiles/"
                if not  os.path.isdir(el.departDir):
                    os.mkdir(el.departDir)

        if 'depthScale' not in self.__dict__:
            self.depthScaleNew = np.linspace(-5, 2, 60)
        else: self.depthScaleNew = np.array(depthScale[0], depthScale[1], depthScale[2])

        self.interpolate()

        "Some formatting required by TS routines"
        self.createTSinputFlags()


    def interpolate(self):
        """
        Here we interpolate grids of model atmospheres and
        grids of NLTE departures to each requested point
        and prepare all the files (incl. writing) in advance,
        i.e. before starting spectral computations with TS

        This decision was made to ensure that interpolation will not
        cause troubles in the middle of large expensive computations
        such as computing hundreds of thousands of model spectra for surveys
        like 4MOST and WEAVE
        """
        self.interpolator = {}

        """
        Read grid of model atmospheres, conduct checks
        and eventually interpolate to every requsted point

        Model atmospheres are stored in the memory at this point
        (self.inputParams['modelAtmInterpol'])
        in contrast to NLTE departure coefficients,
        which are significantly large
        """
        interpolCoords = self.prepInterpolation_MA()
        self.interpolateAllPoints_MA()
        del self.interpolator['modelAtm']


        """
        Go over each NLTE departure grids
        (one at a time to avoid memory overflow),
        read, conduct checks and eventually interpolate to every requested point

        Each set of departure coefficients is written in the file at this point,
        since storing all of them in the memory is not possible
        These files serve as input to TS 'as is' later anyways.

        Departure coefficients are rescaled to the same depth scale \
        as in each model atmosphere
        """
        for elID, el in self.inputParams['elements'].items():
            if el.nlte:
                print(el.ID)
                search = []
                el.departFiles = np.full(self.inputParams['count'], None)
                for i in range(len(el.abund)):
                    departFile = el.departDir + \
                            f"/depCoeff_{el.ID}_{el.abund[i]:.3f}_{i}.dat"
                    el.departFiles[i] = departFile
                    if not os.path.isfile(departFile):
                        search.append(False)
                    else:
                      search.append(True)
                if np.array(search).all():
                    print(f"Will re-use interpolated departure coefficients found under {el.departDir}")
                else:
                    self.prepInterpolation_NLTE(el, interpolCoords, \
                        rescale = True, depthScale = self.depthScaleNew)
                    self.interpolateAllPoints_NLTE(el)
                    del el.nlteData
                    del el.interpolator
# TODO: move the four routines below into model_atm_interpolation

    def prepInterpolation_MA(self):
        """
        Read grid of model atmospheres and NLTE grids of departures
        and prepare interpolating functions
        Store for future use
        """

        " Over which parameters (== coordinates) to interpolate?"
        interpolCoords = ['teff', 'logg', 'feh'] # order should match input file!
        if 'vturb' in self.inputParams:
            interpolCoords.append('vturb')

        "Model atmosphere grid"
        if self.debug: print("preparing model atmosphere interpolator...")
        modelAtmGrid = get_all_ma_parameters(self.atmos_path,  self.depthScaleNew,\
                                        format = self.atmos_format, debug=self.debug)
        passed  = preInterpolationTests(modelAtmGrid, interpolCoords, \
                                        valueKey='structure', dataLabel = 'model atmosphere grid' )
        if not passed:
            exit()
        interpFunction, normalisedCoord = NDinterpolateGrid(modelAtmGrid, interpolCoords, \
                                        valueKey='structure')
        """
        Create hull object to test whether each of the requested points
        are within the original grid
        Interpolation outside of hull returns NaNs, therefore skip those points
        """
        hull = Delaunay(np.array([ modelAtmGrid[k] / normalisedCoord[k] for k in interpolCoords ]).T)

        self.interpolator['modelAtm'] = {'interpFunction' : interpFunction, \
                                        'normCoord' : normalisedCoord, \
                                        'hull': hull}
        del modelAtmGrid
        return interpolCoords


    def prepInterpolation_NLTE(self, el, interpolCoords, rescale = False, depthScale = None):
        """
        Read grid of departure coefficients
        in nlteData 0th element is tau, 1th--Nth are departures for N levels
        """
        if self.debug:
            print(f"reading grid {el.nlteGrid}...")

        el.nlteData = read_fullNLTE_grid( el.nlteGrid, el.nlteAux, \
                                    rescale=rescale, depthScale = depthScale, safeMemory = self.safeMemory )
        t = np.where(el.nlteData['depart'] > 100000)
        print(f"{len(np.unique(t[0]))} models")
        print(f"levels: {np.unique(t[1])}")
        print(f"depth: {np.unique(t[2])}")
        #el.comment += el.nlteData['comment']
        del el.nlteData['comment']

        """ Scaling departure coefficients for the most efficient interpolation """
        pos = np.isnan(np.log10(el.nlteData['depart']+ 1.e-20))
        print(f"{np.sum(pos)} points become NaN under log10")

        el.nlteData['depart'] = np.log10(el.nlteData['depart']+ 1.e-20)
        pos = np.where(np.isnan(el.nlteData['depart']))

        el.DepartScaling = np.max(np.max(el.nlteData['depart'], axis=1), axis=0)
        el.nlteData['depart'] = el.nlteData['depart'] / el.DepartScaling

        # """ Stack departure coefficients and depth scale for consistent interpolation """
        # el.nlteData['departNew'] = np.full((np.shape(el.nlteData['depart'])[0], np.shape(el.nlteData['depart'])[1]+1, np.shape(el.nlteData['depart'])[2]), np.nan)
        # for i in range(len(el.nlteData['pointer'])):
        #     el.nlteData['departNew'][i] = np.vstack([el.nlteData['depthScale'][i], el.nlteData['depart'][i]])
        # el.nlteData['depart'] = el.nlteData['departNew'].copy()
        # del el.nlteData['departNew']
        # del el.nlteData['depthScale']

#
        """
        If element is Fe, than [Fe/H] == A(Fe) with an offset,
        so one of the parameters needs to be excluded to avoid degeneracy
        Here we omit [Fe/H] dimension but keep A(Fe)
        """
        if len(np.unique(el.nlteData['feh'])) == len(np.unique(el.nlteData['abund'])): # it is probably Fe
            if el.isFe:
                interpolCoords_el = [c for c in interpolCoords if c!='feh']
                indiv_abund = np.unique(el.nlteData['abund'])
            else:
                print(f"abundance of {el.ID} is coupled to metallicity, \
but element is not Fe (for Fe A(Fe) == [Fe/H] is acceptable)")
                exit()
        elif len(np.unique(el.nlteData['abund'])) == 1 : # it is either H or no iteration ovr abundance was included in computations of NLTE grids
                interpolCoords_el = interpolCoords.copy()
                indiv_abund = np.unique(el.nlteData['abund'])
        else:
            interpolCoords_el = interpolCoords.copy()
            indiv_abund = np.unique(el.nlteData['abund'] - el.nlteData['feh'])

        """
        Here we use Delaunay triangulation to interpolate over
        fund. parameters like Teff, log(g), [Fe/H], etc,
        and direct linear interpolation for abundance,
        since it is regularly spaced by construction.
        This saves a lot of time.
        """
        el.interpolator = {
                'abund' : [], 'interpFunction' : [], 'normCoord' : [], 'tri':[]
        }

        """ Split the NLTE grid into chuncks of the same abundance """
        subGrids = {
                'abund':np.zeros(len(indiv_abund)), \
                'nlteData':np.empty(len(indiv_abund), dtype=dict)
        }
        for i in range(len(indiv_abund)):
            subGrids['abund'][i] = indiv_abund[i]
            if el.isFe or el.isH:
                mask = np.where( np.abs(el.nlteData['abund'] - \
                                subGrids['abund'][i]) < 0.001)[0]
            else:
                mask = np.where( np.abs(el.nlteData['abund'] - \
                        el.nlteData['feh'] - subGrids['abund'][i]) < 0.001)[0]
            subGrids['nlteData'][i] = {
                        k: el.nlteData[k][mask] for k in el.nlteData
            }

        """
        Run tests and eventually build an interpolating function
        for each sub-grid of constant abundance
        Delete intermediate data
        """
        for i in range(len(subGrids['abund'])):
            ab = subGrids['abund'][i]
            passed = preInterpolationTests(subGrids['nlteData'][i], \
                                        interpolCoords_el, \
                                        valueKey='depart', \
                                        dataLabel=f"NLTE grid {el.ID}")
            if passed:
                interpFunction, normalisedCoord  = \
                    NDinterpolateGrid(subGrids['nlteData'][i], \
                        interpolCoords_el,\
                        valueKey='depart')

                el.interpolator['abund'].append(ab)
                el.interpolator['interpFunction'].append(interpFunction)
                el.interpolator['normCoord'].append(normalisedCoord)
                #el.interpolator['tri'].append(tri)
            else:
                print("Failed pre-interpolation tests, see above")
                print(f"NLTE grid: {el.ID}, A({el.ID}) = {ab}")
                exit()
        del subGrids


    def interpolateAllPoints_MA(self):
        """
        Python parallelisation libraries can not send more than X Gb of data between processes
        To avoid that, interpolation at each requested point is done before the start of computations
        """
        if self.debug: print(f"Interpolating to each of {self.inputParams['count']} requested points...")

        "Model atmosphere grid"
        self.inputParams.update({'modelAtmInterpol' : np.full(self.inputParams['count'], None) })

        countOutsideHull = 0
        for i in range(self.inputParams['count']):
            point = [ self.inputParams[k][i] / self.interpolator['modelAtm']['normCoord'][k] \
                    for k in self.interpolator['modelAtm']['normCoord'] ]
            if not in_hull(np.array(point).T, self.interpolator['modelAtm']['hull']):
                countOutsideHull += 1
            else:
                values =  self.interpolator['modelAtm']['interpFunction'](point)[0]
                self.inputParams['modelAtmInterpol'][i] = values
        if countOutsideHull > 0 and self.debug:
            print(f"{countOutsideHull}/{self.inputParams['count']}requested \
points are outside of the model atmosphere grid.\
No computations will be done for those")

    def interpolateAllPoints_NLTE(self, el):
        """
        Interpolate to each requested abundance of element (el)
        Write departure coefficients to a file
        that will be used as input to TS later
        """
        el.departFiles = np.full(self.inputParams['count'], None)
        for i in range(len(el.abund)):
                departFile = el.departDir + \
                        f"/depCoeff_{el.ID}_{el.abund[i]:.3f}_{i}.dat"
#            if not os.path.isfile(departFile):
                x, y = [], []
                # TODO: introduce class for nlte grid and set exceptions if grid wasn't rescaled
                tau = self.depthScaleNew
                for j in range(len(el.interpolator['abund'])):
                    point = [ self.inputParams[k][i] / el.interpolator['normCoord'][j][k] \
                             for k in el.interpolator['normCoord'][j] if k !='abund']
                    ab = el.interpolator['abund'][j]
                    departAb = el.interpolator['interpFunction'][j](point)[0]
                    #simp = el.interpolator['tri'][j].simplices
                    #psimp = el.interpolator['tri'][j].find_simplex(point)
                    #print(j, simp[psimp])
                    #df = f"./testInterpolDepart/Na_{j:.0f}_" + '_'.join( f"{s:.0f}" for s in simp[psimp]) + '.dat'
                    #write_departures_forTS(df, tau, departAb, el.interpolator['abund'][j])
                    if not np.isnan(departAb).all():
                        x.append(ab)
                        y.append(departAb)
                x, y = np.array(x), np.array(y)
                """
                Now interpolate linearly along abundance axis
                If only one point is present (e.g. A(H) is always 12),
                take departure coefficient at that abundance
                """
                if len(x) >= 2:
                    if not el.isFe or el.isH:
                        abScale = el.abund[i] - self.inputParams['feh'][i]
                    else:
                        abScale = el.abund[i]
                    if abScale > min(x) and abScale < max(x):
                        depart = interp1d(x, y, axis=0)(abScale)
                        depart = restoreDepartScaling(depart, el)
                    else:
                        depart = np.nan
                    #print('NaNs?', np.isnan(depart).any())
                elif len(x) == 1 and el.isH:
                    print(f'only one point at abundandance={x} found, will accept depart coeff.')
                    depart = y[0]
                    depart = restoreDepartScaling(depart, el)
                    #print('NaNs?', np.isnan(depart).any())
                else:
                    print(f"Found no departure coefficients \
at A({el.ID}) = {el.abund[i]}, [Fe/H] = {self.inputParams['feh'][i]} at i = {i}")
                    depart = np.nan

                """
                Check that no non-linearities are present
                """
                nonLin = False
                if not np.isnan(depart).all():
                    for ii in range(np.shape(depart)[0]):
                        if (gradient3rdOrder( depart[ii] ) > 0.01).any():
                            depart = np.nan
                            nonLin = True
                            self.inputParams['comments'][i] += f"Non-linear behaviour in the interpolated departure coefficients \
of {el.ID} found. Will be using the closest data from the grid instead of interpolated values.\n"
                            break
                if not nonLin:
                    print(f'no weird behaviour encountered for {el.ID} at abund={ el.abund[i]:.2f}')
                else:
                    print(f"non-linearities for {el.ID} at abund={el.abund[i]:.2f}")
                """
                If interpolation failed e.g. if the point is outside of the grid,
                find the closest point in the grid and take a departure coefficient
                for that point
                """
                if np.isnan(depart).all():
                    if self.debug:
                        print(f"attempting to find the closest point the in the grid of departure coefficients")
# TODO: move the four routines below into model_atm_interpolation
                    point = {}
                    for k in el.interpolator['normCoord'][0]:
                        point[k] = self.inputParams[k][i]
                    if 'abund' not in point:
                        point['abund'] = el.abund[i]
                    pos, comment = find_distance_to_point(point, el.nlteData)
                    depart = el.nlteData['depart'][pos]
                    depart = restoreDepartScaling(depart, el)
                    tau = el.nlteData['depthScale'][pos]

                    for k in el.interpolator['normCoord'][0]:
                        if ( np.abs(el.nlteData[k][pos] - point[k]) / point[k] ) > 0.5:
                            for k in el.interpolator['normCoord'][0]:
                                self.inputParams['comments'][i] += f"{k} = {el.nlteData[k][pos]}\
 (off by {point[k] - el.nlteData[k][pos] }) \n"

                write_departures_forTS(departFile, tau, depart, el.abund[i])
                el.departFiles[i] = departFile
                self.inputParams['comments'][i] += el.comment




    def createTSinputFlags(self):
        self.ts_input = { 'PURE-LTE':'.false.', 'MARCS-FILE':'.false.', 'NLTE':'.false.',\
        'NLTEINFOFILE':'', 'LAMBDA_MIN':4000, 'LAMBDA_MAX':9000, 'LAMBDA_STEP':0.05,\
         'MODELOPAC':'./OPAC', 'RESULTFILE':'' }


        """ At what wavelenght range to compute a spectrum? """
        self.lam_start, self.lam_end = min(self.lam_end, self.lam_start), \
                                    max(self.lam_end, self.lam_start)
        self.wave_step = np.mean([self.lam_start, self.lam_end]) / self.resolution
        self.ts_input['LAMBDA_MIN'] = self.lam_start
        self.ts_input['LAMBDA_MAX'] = self.lam_end
        self.ts_input['LAMBDA_STEP'] = self.wave_step
        self.ts_input['ts_root'] = self.ts_root


        """ Linelists """
        if type(self.linelist) == np.array or type(self.linelist) == np.ndarray:
            pass
        elif type(self.linelist) == str:
            self.linelist = np.array([self.linelist])
        else:
            print(f"Can not understand the 'linelist' flag: {self.linelist}")
            exit(1)
        llFormatted = []
        for path in self.linelist:
            if '*' in path:
                llFormatted.extend( glob.glob(path) )
            else:
                llFormatted.append(path)
        self.linelist = llFormatted
        print(f"Linelist(s) will be read from: {' ; '.join(str(x) for x in self.linelist)}")

        self.ts_input['NFILES'] = len(self.linelist)
        self.ts_input['LINELIST'] = '\n'.join(self.linelist)


        "Any element in NLTE?"
        if self.nlte:
            self.ts_input['NLTE'] = '.true.'
