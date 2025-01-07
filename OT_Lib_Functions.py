# We make use of the software package OTlib.py, which implements the approach of Sambridge et al. (2022).
# https://github.com/msambridge/waveform-ot/tree/main

# Reference paper: Malcolm Sambridge, Andrew Jackson, y Andrew P Valentine,
# «Geophysical Inversion and Optimal Transport»,
# Geophysical Journal International 231, n.º 1 (21 de junio de 2022): 172-98,
# https://doi.org/10.1093/gji/ggac151.


# waveform-ot libraries
from Imported_Libraries import OTlib as OT
from Imported_Libraries import ricker_util as ru 


# waveform-ot functions
# https://github.com/msambridge/waveform-ot/blob/main/Point_mass_demo_Fig_5.ipynb
# https://github.com/msambridge/waveform-ot/blob/main/Ricker_Figs_1_7.ipynb


def compute_OT(simulated_data, observed_data):
    # set up initial and final PDFs simple point masses
    fx = np.linspace(0, nx*nz, nx*nz)
    gx = np.linspace(0, nx*nz, nx*nz)
    f = np.abs(simulated_data.flatten())
    g = np.abs(observed_data.flatten())

    # set up PDF objects
    source = OT.OTpdf((f,fx)) # set up source object
    target = OT.OTpdf((g,gx)) # set up target object

    w_1 = np.round(OT.wasser(source,target,distfunc='W1')[0],4)
    w_2 = np.round(OT.wasser(source,target,distfunc='W2')[0],4)

    return (w_1 + w_2)


def compute_OT_2(simulated_data, observed_data):
    fx = np.linspace(0, nx*nz, nx*nz)
    gx = np.linspace(0, nx*nz, nx*nz)
    f = observed_data.flatten()
    g = simulated_data.flatten()

    wfobs, wfobs_target = ru.BuildOTobjfromWaveform(fx,f,grid,lambdav=lambdav)

    wfs,wfsource = ru.BuildOTobjfromWaveform(gx,g,grid,lambdav=lambdav)

    w_1 = ru.CalcWasserWaveform(wfsource,wfobs_target,wfs,distfunc='W1')
    w_2 = ru.CalcWasserWaveform(wfsource,wfobs_target,wfs,distfunc='W2')

    return (w_1 + w_2)


def compute_OT_3(simulated_data, observed_data):
    w = 0
    for i in range(nx):
        fx = np.linspace(0, nx, nx)
        gx = np.linspace(0, nx, nx)
        f = observed_data[:,i]
        g = simulated_data[:,i]

        wfobs, wfobs_target = ru.BuildOTobjfromWaveform(fx,f,grid,lambdav=lambdav)

        wfs,wfsource = ru.BuildOTobjfromWaveform(gx,g,grid,lambdav=lambdav)

        w_1 = ru.CalcWasserWaveform(wfsource,wfobs_target,wfs,distfunc='W1')
        w_2 = ru.CalcWasserWaveform(wfsource,wfobs_target,wfs,distfunc='W2')

        w = w + (w_1 + w_2)

        return w
    