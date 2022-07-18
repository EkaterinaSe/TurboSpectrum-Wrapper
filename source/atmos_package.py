import numpy as np
from copy import deepcopy

"""
    Read and manipulate model atmospheres
"""
# TODO: change lists to arrays!
def read_atmos_marcs(self, file):
    """
    Read a model atmosphere in marcs format

    Parameters
    ----------
    self : model_atmosphere
        empty initialised model atmosphere object
    file : str
        from which file to read the model
    """

    # Boltzmann constant
    k_B = 1.38064852E-16

    data = []
    for line in  open(file, 'r').readlines():
        data.append(line.strip())
    # MARCS model atmosphere are by default strictly formatted
    self.id = data[0]
    if self.id.startswith('p'):
        self.pp = True
        self.spherical = False
    elif self.id.startswith('s'):
        self.spherical = True
        self.pp = False
    else:
        print(f"Could not understand model atmosphere geometry for {file}")

    self.teff = float(data[1].split()[0])
    self.flux = float(data[2].split()[0])
    self.logg = np.log10( float(data[3].split()[0]) )
    self.vturb = float(data[4].split()[0])
    self.mass = float(data[5].split()[0])
    self.feh, self.alpha = np.array(data[6].split()[:2]).astype(float)
    self.X, self.Y, self.Z = np.array(data[10].split()[:3]).astype(float)

    # read structure
    for line in data:
        if 'Number of depth points' in line:
            self.ndep = int(line.split()[0])
    self.k, self.tau500, self.height, self.temp, self.ne = [], [], [], [], []
    for line in data[25:25+self.ndep]:
        spl = np.array( line.split() ).astype(float)
        self.k.append(spl[0])
        self.tau500.append(spl[2])
        self.height.append(spl[3])
        t = spl[4]
        self.temp.append(t)
        pe = spl[5]
        ne =  pe / t / k_B
        self.ne.append(ne)

    self.vturb = np.full(self.ndep, self.vturb )
    self.vmac = np.zeros(self.ndep)
    # add comments
    self.header = "Converted from MARCS formatted model atmosphere %s" %( file.split('/')[-1].strip() )

    return


def read_atmos_m1d(self, file):
    """
    Read a model atmosphere in the MULTI2.4 format (e.g. 'atmos.sun').
    MULTI format is short on info, so some guessing for parameters is done

    Parameters
    ----------
    self : model_atmosphere
        empty initialised model atmosphere object
    file : str
        from which file to read the model
    """
    lines = [ l.strip() for l in open( file , 'r').readlines() if not l.startswith('*')]
    data = []
    for l in lines:
        if 'Teff' in l:
            self.teff = float(l.split()[-1].split('=')[-1])
        else:
            data.append( l )

    # read header
    self.id = data[0]
    self.depth_scale_type = data[1]
    self.logg = float(data[2])
    self.ndep = int(data[3])
    # read structure
    self.depth_scale, self.temp, self.ne, self.vmac, self.vturb = [],[],[],[],[]
    for line in data[ 4 : ]:
        spl = np.array(line.split()).astype(float)
        self.depth_scale.append( spl[0] )
        self.temp.append( spl[1] )
        self.ne.append( spl[2] )
        self.vmac.append( spl[3] )
        self.vturb.append( spl[4] )
    # guess for the info that's not provided in the model atmosphere file:
    if not 'teff' in self.__dict__.keys():
        self.teff   = np.nan
    self.X      = np.nan
    self.Y      = np.nan
    self.Z      = np.nan
    self.mass   = np.nan
    # add comments here
    self.header = f"Read from M1D formatted model atmosphere {file.split('/')[-1].strip()}"

    return self


def write_atmos_m1d(atmos, file):
    """
    Write a model atmosphere in the MULTI2.4 format (e.g. 'atmos.sun').
    MULTI format is short on info, so some guessing for parameters is done

    Parameters
    ----------
    atmos : model_atmosphere

    file : str
        output file
    """

    with open(file, 'w') as f:
        f.write( f"* {atmos.header} \n" )

        f.write( f"%s {atmos.id}\n" )
        f.write( f"* Depth scale: log(tau500nm) (T), log(column mass) (M), height [km] (H)\n" )
        f.write( f"{atmos.depth_scale_type} \n" )
        f.write( f"* log(g) \n {atmos.logg:.3f} \n" )
        f.write( f"* Teff = {atmos.teff:.1f}\n")
        f.write( f"* Number of depth points \n {atmos.ndep:.0f} \n" )

        f.write("* depth scale, temperature, N_e, Vmac, Vturb \n")

        for i in range(atmos.ndep):
            s = '\t'.join(f"{ar[i]}" for ar in [atmos.depth_scale, \
                                                atmos.temp, atmos.ne, \
                                                atmos.vmac, atmos.vturb])
            f.write( f"{s}\n" )

def write_atmos_m1d4TS(atmos, file):
    """
    Write model atmosphere in MULTI 1D input format, i.e. atmos.*
    input:
    (object of class model_atmosphere): atmos
    (string) file: path to output file1
    """

    with open(file, 'w') as f:
        f.write(f"{atmos.id}\n" )
        f.write(f"{atmos.depth_scale_type}\n" )
        f.write(f"* LOG (G) \n {atmos.logg} \n")
        f.write(f"* NDEP \n {atmos.ndep} \n" )
        # write structure
        f.write("* depth scale, temperature, N_e, Vmac, Vturb \n")
        for i in range(len(atmos.depth_scale)):
            f.write("%15.8E %15.5f %15.5E %10.3f %10.3f\n" \
                %( atmos.depth_scale[i], atmos.temp[i], atmos.ne[i], atmos.vmac[i], atmos.vturb[i] ) )


def write_dscale_m1d(atmos, file):
    """
    Write MULTI1D DSCALE input file with depth scale to be used for NLTE computations
    """
    with open(file, 'w') as f:
        # write formatted header
        f.write("%s \n" %(atmos.id) )
        f.write("* Depth scale: log(tau500nm) (T), log(column mass) (M), height [km] (H)\n %s \n" %(atmos.depth_scale_type) )
        f.write("* Number of depth points, top point \n %.0f %10.4E \n" %(atmos.ndep, atmos.depth_scale[0]) )
        # write structure
        for i in range(len(atmos.depth_scale)):
            f.write("%15.5E \n" %( atmos.depth_scale[i] ) )



    return


class model_atmosphere(object):
    # def __init__():
        # pass
    def read(self, file='atmos.sun', format='m1d'):
        """
        Model atmosphere for NLTE calculations
        input:
        (string) file: file with model atmosphere, default: atmos.sun
        (string) format: m1d, marcs, stagger, see function calls below
        """
        if format.lower() == 'marcs':
            read_atmos_marcs(self, file)
            # print("Setting depth scale to tau500")
            self.depth_scale_type = 'TAU500'
            self.depth_scale = self.tau500
        elif format.lower() == 'm1d':
            read_atmos_m1d(self, file)
            try:
                feh = float(self.id.split('_z')[-1].split('_a')[0])
                alpha = float(self.id.split('_a')[-1].split('_c')[0])
                self.feh = feh
                self.alpha = alpha
                #print(F"Guessed [Fe/H]={self.feh}, [alpha/Fe]={self.alpha}")
            except:
                try:
                    feh = float(self.id.split('m')[-1].split('_')[0])
                    self.feh = feh
                    self.alpha = self.feh
                except:
                    print("WARNING: [Fe/H] and [alpha/Fe] are unknown")

                    self.feh = np.nan
                    self.alpha = np.nan
        elif format.lower() == 'stagger':
#            print(F"Guessing [Fe/H] and [alpha/Fe] from the file name {self.id}..")
            read_atmos_m1d(self, file)
            teff = float(self.id.split('g')[0].replace('t',''))
            if teff != 5777:
                teff = teff*1e2

            feh = float(self.id[-2:]) /10
            if self.id[-3] == 'm':
                feh = feh * (-1)
            elif self.id[-3] == 'p':
                pass
            else:
                raise Warning("WARNING: [Fe/H] and [alpha/Fe] are unknown. Stopped")
            self.feh = feh
            self.alpha = self.feh
            self.teff = teff
            #print(F"Guessed [Fe/H]={self.feh}, [alpha/Fe]={self.alpha}")
        else:
            raise Warning("Unrecognized format of model atmosphere: %s" %(format) )

    def FillIn(self):
        if 'logg' not in self.__dict__.keys():
            self.logg  = np.nan
        if 'teff' not in self.__dict__.keys():
            self.teff  = np.nan
        if 'header' not in self.__dict__.keys():
            self.header  = ''
        if 'ndep' not in self.__dict__.keys():
            self.ndep = len(self.depth_scale)
        if 'vmac' not in self.__dict__.keys():
            self.vmac = np.zeros( len(self.depth_scale ))


    def copy(self):
        return deepcopy(self)

    def write(self, path, format = 'm1d'):
        self.FillIn()
        if format == 'm1d':
            write_atmos_m1d(self, path)
        elif format == 'ts':
            write_atmos_m1d4TS(self, path)
        else:
            raise Warning(f"Format {format} not supported for writing yet.")



if __name__ == '__main__':
    atmos = model_atmosphere('./atmos.sun_marcs_t5777_4.44_0.00_vmic1_new', format='m1d')
    write_atmos_m1d(atmos, file='atmos.test_out')
    write_dscale_m1d(atmos, file='dscale.test_out')
    atmos = model_atmosphere('./sun.mod', format='Marcs')
    write_atmos_m1d(atmos, file='atmos.test_out')
    write_dscale_m1d(atmos, file='dscale.test_out')
    exit(0)
