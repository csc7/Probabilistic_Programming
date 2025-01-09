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

    # choose grid to use
    nugrid,ntgrid=40,512 # This is the discretization of the waveform window used to evaluated the density surface
    lambdav = 1 #0.03    # This is the distance scale factor used to calculate the density function (eqn. 17 of Sambridge et al. (2
    trange = [0, nx]
    grid = (trange[0], trange[1], -1.0, 1.0, nugrid,ntgrid) # specify grid for fingerprint # -0.8, 1.2

    # set up initial and final PDFs simple point masses
    fx = np.linspace(0, nx*nz, nx*nz)
    gx = np.linspace(0, nx*nz, nx*nz)
    f = np.abs(simulated_data.T.flatten())
    g = np.abs(observed_data.T.flatten())

    wf_f, OT_pdf_f = ru.BuildOTobjfromWaveform(fx,f,grid,lambdav=lambdav)
    wf_g, OT_pdf_g = ru.BuildOTobjfromWaveform(gx,g,grid,lambdav=lambdav)

    #ru.fp.plot_LS(wf_f.dfield,wf_f,None,None,'Fingerprint of observed data','grey','grey',aspect=True) 
    #ru.fp.plot_LS(wf_g.dfield,wf_g,None,None,'Fingerprint of simulated data','grey','grey',aspect=True) 

    w_1 = ru.CalcWasserWaveform(OT_pdf_f, OT_pdf_g, wf_g, distfunc='W1', deriv=False, returnmarg=False)
    w_2 = ru.CalcWasserWaveform(OT_pdf_f, OT_pdf_g, wf_g, distfunc='W1', deriv=False, returnmarg=False)

    return (w_2)


def compute_OT_2(simulated_data, observed_data):

    # choose grid to use
    nugrid,ntgrid=40,512 # This is the discretization of the waveform window used to evaluated the density surface
    lambdav = 1 #0.03    # This is the distance scale factor used to calculate the density function (eqn. 17 of Sambridge et al. (2
    trange = [0, nx]
    grid = (trange[0], trange[1], -1.0, 1.0, nugrid,ntgrid) # specify grid for fingerprint # -0.8, 1.2

    fx = np.linspace(0, nx*nz, nx*nz)
    gx = np.linspace(0, nx*nz, nx*nz)
    f = observed_data.T.flatten()
    g = simulated_data.T.flatten()

    wf_f, OT_pdf_f = ru.BuildOTobjfromWaveform(fx,f,grid,lambdav=lambdav)
    wf_g, OT_pdf_g = ru.BuildOTobjfromWaveform(gx,g,grid,lambdav=lambdav)

    #ru.fp.plot_LS(wf_f.dfield,wf_f,None,None,'Fingerprint of observed data','grey','grey',aspect=True) 
    #ru.fp.plot_LS(wf_g.dfield,wf_g,None,None,'Fingerprint of simulated data','grey','grey',aspect=True) 

    w_1 = ru.CalcWasserWaveform(OT_pdf_f, OT_pdf_g, wf_g, distfunc='W1', deriv=False, returnmarg=False)
    w_2 = ru.CalcWasserWaveform(OT_pdf_f, OT_pdf_g, wf_g, distfunc='W1', deriv=False, returnmarg=False)

    return (w_2)


def compute_OT_3(simulated_data, observed_data):
    w = 0
    # choose grid to use
    nugrid,ntgrid=40,512 # This is the discretization of the waveform window used to evaluated the density surface
    lambdav = 1 #0.03    # This is the distance scale factor used to calculate the density function (eqn. 17 of Sambridge et al. (2
    trange = [0, nx]
    grid = (trange[0], trange[1], -1.0, 1.0, nugrid,ntgrid) # specify grid for fingerprint # -0.8, 1.2

    for i in range(nx):
        fx = np.linspace(0, nx, nx)
        gx = np.linspace(0, nx, nx)
        f = observed_data.T[:,i]
        g = simulated_data.T[:,i]

        wf_f, OT_pdf_f = ru.BuildOTobjfromWaveform(fx,f,grid,lambdav=lambdav)
        wf_g, OT_pdf_g = ru.BuildOTobjfromWaveform(gx,g,grid,lambdav=lambdav)

        #ru.fp.plot_LS(wf_f.dfield,wf_f,None,None,'Fingerprint of observed data','grey','grey',aspect=True) 
        #ru.fp.plot_LS(wf_g.dfield,wf_g,None,None,'Fingerprint of simulated data','grey','grey',aspect=True) 

        w_1 = ru.CalcWasserWaveform(OT_pdf_f, OT_pdf_g, wf_g, distfunc='W1', deriv=False, returnmarg=False)
        w_2 = ru.CalcWasserWaveform(OT_pdf_f, OT_pdf_g, wf_g, distfunc='W1', deriv=False, returnmarg=False)

        w = (w_2)

        return w
    