# external
import os
from sys import argv
import shutil
import subprocess
import numpy as np
import pickle
import glob
import time
import datetime
# local
from atmos_package import read_atmos_marcs, model_atmosphere

def compute_babsma(ts_input, atmos, modelOpacFile, quite=True):
    """
    Creates input for the babsma.f routine and executes it
    babsma.f routine computes opacities for the give model atmosphere
    which are then used by the bsyn.f routine

    Parameters
    ----------
    ts_input : dict
        contains TS input flags
        must include the following flags:
            'MARCS-FILE'('.true.' or '.false.'),
            'ts_root' (path to TS executables bsyn.f and babsma.f)
    atmos : model_atmosphere
        for which model atmosphere to compute the opacities
    modelOpacFile : str
        where to store computed opacities
    quite : boolean
        controls details printout of the progress info
    """

    babsma_conf = F""" \
'MODELINPUT:'    '{atmos.path}'
'MARCS-FILE:' '{ts_input['MARCS-FILE']}'
'MODELOPAC:' '{modelOpacFile}'
'METALLICITY:'    '{atmos.feh:.3f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
    """

    time0 = time.time()
    cwd = os.getcwd()
    os.chdir(ts_input['ts_root'])
    pr = subprocess.Popen(['./exec/babsma_lu'], stdin=subprocess.PIPE, \
        stdout=open(set.cwd + '/babsma.log', 'w'), stderr=subprocess.STDOUT )
    pr.stdin.write(bytes(babsma_conf, 'utf-8'))
    pr.communicate()
    pr.wait()
    os.chdir(cwd)
    if not quite:
        print(F"babsma: {time.time()-time0} seconds")

def compute_bsyn(ts_input, elementalAbundances, atmos, modelOpacFile, specResultFile, nlteInfoFile=None, quite = True):
    """
    Creates input for the bsyn.f routine and executes it
    bsyn.f runs spectral synthesis based on the opacities
    computed previsously by babsma.f

    Parameters
    ----------
    ts_input : dict
        contains TS input flags
        must include the following flags:
            'NLTE' ('.true.' or '.false.'),
            'LAMBDA_MIN', 'LAMBDA_MAX', 'LAMBDA_STEP',
            'MARCS-FILE'('.true.' or '.false.'),
            'NFILES' (how many linelists provided, integer),
            'LINELIST' (separated with new line),
            'ts_root' (path to TS executables bsyn.f and babsma.f)
    elementalAbundances : list
        contains atomic numbers and abundances of the elements requested for spectral synthesis,
        e.g.
        [ [26, 7.5], [8, 8.76] ]
    atmos : model_atmosphere
        for which model atmosphere to compute the opacities
    modelOpacFile : str
        path to the file storing opacities computed by babsma
        returned by compute_babsma
    specResultFile : str
        where to save computed spectrum
    nlteInfoFile : str
        path to the configuration file that controls inclusion of NLTE to TS
        returned by create_NlteInfoFile
        if None, spectrum will be computed in LTE
    quite : boolean
        controls details printout of the progress info
    """

    bsyn_config = F""" \
'NLTE :'          '{ts_input['NLTE']}'
'LAMBDA_MIN:'    '{ts_input['LAMBDA_MIN']:.3f}'
'LAMBDA_MAX:'    '{ts_input['LAMBDA_MAX']:.3f}'
'LAMBDA_STEP:'   '{ts_input['LAMBDA_STEP']:.3f}'
'INTENSITY/FLUX:' 'Flux'
'MARCS-FILE:' '{ts_input['MARCS-FILE']}'
'MODELOPAC:'        '{modelOpacFile}'
'RESULTFILE :'    '{specResultFile}'
'HELIUM     :'    '0.00'
'NFILES   :' '{ts_input['NFILES']}'
{ts_input['LINELIST']}
"""
    if atmos.spherical:
        bsyn_config = bsyn_config + f" 'SPHERICAL:'  '.true.' "
    else:
        bsyn_config = bsyn_config + f" 'SPHERICAL:'  '.false.' "

    bsyn_config = bsyn_config + f"""\
 30
 300.00
 15
 1.30
"""

    if not isinstance(nlteInfoFile,  type(None)):
        bsyn_config = bsyn_config + f"'NLTEINFOFILE:' '{nlteInfoFile}' \n"

    bsyn_config = bsyn_config +\
            f"'INDIVIDUAL ABUNDANCES:'   '{len(elementalAbundances)}' \n"
    for i in range(len(elementalAbundances)):
        z, abund = elementalAbundances[i]
        bsyn_config = bsyn_config + f" {z:.0f} {abund:5.3f} \n"

    """ Run bsyn """
    time0 = time.time()
    cwd = os.getcwd()
    os.chdir(ts_input['ts_root'])
    pr = subprocess.Popen(['./exec/bsyn_lu'], stdin=subprocess.PIPE, \
        stdout=open(cwd + '/bsyn.log', 'w'), stderr=subprocess.STDOUT )
    pr.stdin.write(bytes(bsyn_config, 'utf-8'))
    pr.communicate()
    pr.wait()
    os.chdir(cwd)
    if not quite:
        print(F"bsyn: {time.time()-time0} seconds")

def create_NlteInfoFile(elementalConfig, modelAtomsPath='', departureFilesPath='', filePath='./nlteinfofile.txt'):
    """
    Creates configuration file that controls inclusion of NLTE
    for requsted elements into spectral synthesis

    Parameters
    ----------
    elementalConfig : list
        contains IDs, atomic number, abundances, NLTE flag,
        and departure coefficient file + model atom ID if NLTE is True,
        for the elements requested for spectral synthesis
        e.g.
        [
            ['Fe', 26, 7.5, True, './depart_Fe.dat', 'atom.fe607c'],
            ['O', 8, 8.76, False, '', '']
        ]
    modelAtomsPath : str
        path to the model atoms, since TS requires all the model atoms
        to be provided in the same directory
        can be symbolic links
    departureFilesPath : str
        path to directory containing departure files in TS format
        can be set to empty string, then paths to individual departure files
        have to be absolute
    filePath : str
        where to write the file
    """
    with open(filePath, 'w') as nlte_info_file:
        nlte_info_file.write('# created on \n')
        nlte_info_file.write('# path for model atom files ! this comment line has to be here !\n')
        nlte_info_file.write(F"{modelAtomsPath} \n")

        nlte_info_file.write('# path for departure files ! this comment line has to be here !\n')
        nlte_info_file.write(F"{departureFilesPath} \n")
        nlte_info_file.write('# atomic (non)LTE setup \n')
        for i in range(len(elementalConfig)):
            id, z, abund, nlte, departFile, modelAtom = elementalConfig[i]
            if nlte:
                model_atom_id = modelAtom
                nlte_info_file.write(F"{z}  '{id}'  'nlte' '{modelAtom}'  '{departFile}' 'ascii' \n")
            else:
                nlte_info_file.write(F"{z}  '{id}'  'lte' ''  '' '' \n")

def parallel_worker(set, ind):
    """
    Responsible for organising computations and talking to TS
    Creates model atmosphers, opacity file (by running babsma.f),
    NLTE control file if NLTE is requested fot at least one element,
    computes the spectrum ( by calling bsyn.f),
    and finally cleans up by removing temporary files


    Parameters
    ----------
    set: setup
        requested configuration
    ind : list or np.array of int
        positional indexes of stellar labels and individual abundances
        compuations will be done consequently for each index
    """
    tempDir = f"{set.cwd}/job_{set.jobID}_{min(ind)}_{max(ind)}/"
    if os.path.isdir(tempDir):
        shutil.rmtree(tempDir)
    os.mkdir(tempDir)
    today = datetime.date.today().strftime("%b-%d-%Y")

    elements = set.inputParams['elements'].values()

    for i in ind:
        # TODO: move writing intrpolated model somewhere else, maybe even right after intrpolation
        atmos = model_atmosphere()
        if not isinstance(set.inputParams['modelAtmInterpol'][i], type(None)):
            atmos.depth_scale, atmos.temp, atmos.ne, atmos.vturb = \
                set.inputParams['modelAtmInterpol'][i]
            set.inputParams['modelAtmInterpol'][i] = None
            atmos.temp, atmos.ne = 10**(atmos.temp), 10**(atmos.ne)
            atmos.depth_scale_type = 'TAU500'
            atmos.feh, atmos.logg = set.inputParams['feh'][i], set.inputParams['logg'][i]
            atmos.spherical = False
            atmos.id = f"interpol_{i:05d}_{set.jobID}"
            atmos.path = f"{tempDir}/atmos.{atmos.id}"

            atmos.write(atmos.path, format = 'ts')

            """ Compute model atmosphere opacity with babsma.f"""
            modelOpacFile = F"{set.ts_input['ts_root']}/opac_{atmos.id}_{set.jobID}"
            compute_babsma(set.ts_input, atmos, modelOpacFile, set.debug)

            """ Compute the spectrum """
            specResultFile = specResultFile + f"{['NLTE' if set.nlte else 'LTE'][0]}"

            header = f"computed with TS NLTE v.20 \n\
by E.Magg (emagg at mpia dot de) \n\
Date: {today} \n\
Input parameters: \n\
"
            header += '\n'.join( f"{k} = {set.inputParams[k][i]}" for  k in set.freeInputParams)
            header += '\n'.join(f"A({el.ID}) = {el.abund[i]} {['NLTE' if el.nlte else 'LTE']}" for el in elements)

            "Create NLTE info file"
            if set.nlte:
                for el in elements:
                    nlteInfoFile   = f"{tempDir}/NLTEinfoFile_{set.jobID}.txt"
                    elementalConfig = []
                    for el in set.inputParams['elements'].values():
                        if el.nlte:
                            if not isinstance(el.departFiles[i], type(None)):
                                cnfg = [
                                        el.ID, el.Z, el.abund[i],
                                        el.nlte, el.departFiles[i],
                                        el.modelAtom.split('/')[-1]
                                        ]
                            else:
                                cnfg = [ el.ID, el.Z, el.abund[i], False, '', '']
                                set.inputParams['comments'][i] += f"\
failed to create departure file for {el.ID} at A({el.ID}) = {el.abund[i]}. \
Treated in LTE instead."
                        else:
                            cnfg = [ el.ID, el.Z, el.abund[i], False, '', '']
                    elementalConfig.append( cnfg )

            create_NlteInfoFile(elementalConfig, set.modelAtomsPath, '', nlteInfoFile)

            "Run bsyn.f for spectral synthesis"
            elementalConfig = [ [el.Z, el.abund[i]] for el in set.inputParams['elements'].values() ]
            compute_bsyn(
                        set.ts_input, elementalConfig, \
                        atmos, modelOpacFile, specResultFile, \
                        nlteInfoFile, set.debug
            )

            """ Add header, comments and save the spectrum to the common output directory """
            if os.path.isfile(specResultFile) and os.path.getsize(specResultFile) > 0:
                with open(f"{set.spectraDir}/{specResultFile.split('/')[-1]}", 'w') as moveSpec:
                    for l in header.split('\n'):
                        moveSpec.write('#' + l + '\n')
                    for l in set.inputParams['comments'][i].split('\n'):
                        moveSpec.write('#' + l + '\n')
                    moveSpec.write('#\n')
                    for l in open(specResultFile, 'r').readlines():
                        moveSpec.write(l)
                os.remove(specResultFile)

            """ Clean up """
            os.remove(atmos.path)
            os.remove(modelOpacFile)
            os.remove(modelOpacFile+'.mod')

if __name__ == '__main__':
    if len(argv) > 1:
        conf_file = argv[1]
    else:
        print("Usage: ./run_ts.py ./configFile.txt")
        exit()
    set = setup(file = conf_file)
    parallel_worker(set, np.arange(len(set)))
    exit(0)
