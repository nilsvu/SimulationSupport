# fmt: off
# isort: skip_file

"""
This code was only very slightly changed from the SpEC implementation and should
be modernized. It was kept in its original form to make transitioning from SpEC
to this package easier.
Changes made:
- Inlined 'Compute_OrbitalFrequency' from BbhDiagnosticsImpl.py and
  'SmoothData' from Utils.py to remove SpEC dependencies.
- Replaced Utils.norm -> np.linalg.norm
- Removed SpEC-specific H5 and IdParams file reading, main function, and CLI.
  These features remain in SpEC.
"""

from __future__ import division
import os
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, size, sin, cos, array, pi, cross, dot, mean
from numpy.random import rand
from scipy import optimize, signal
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
import VarPro
from VarPro import RankError
from io import StringIO


"""
Performs eccentricity fitting, and outputs the estimated eccentricity
and updated initial data parameters.
"""

def error(msg):
    e_msg = "\n############# ERROR #############\n{}\n".format(msg)
    raise Exception(e_msg)


def warning(msg):
    os.sys.stderr.write("\n############# WARNING #############\n")
    os.sys.stderr.write("{}\n".format(msg))

# Inlined from Utils.py
def SmoothData(t,
               data,
               width,
               Deriv=0,
               conv_factor=4.,
               cut_off=None,
               Tstart=None,
               DT=None,
               dt_tol=1e-12):
    """Smooth the data (t,data) using Gaussian convolution.
Derivatives do not seem to work well when timesteps are much larger than 1.
It also does not work (though it should) for unequally spaced timesteps.
The smooth data will be truncated on both sides based on the smoothing width.
"""
    import math
    import numpy as np

    # Convert to expected objects
    width = float(width)
    try:
        data.shape
    except AttributeError:
        data = np.array(data)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]  #promote the array with singleton dim
    if len(data.shape) != 2:
        error("data is expected to be 2d array, not {}d".format(len(
            data.shape)))

    try:
        t.shape
    except AttributeError:
        t = np.array(t)

    convolution_width = width * conv_factor
    if cut_off == None:
        cut_width = width * conv_factor
    else:
        cut_width = width * cut_off

    # compute average time-spacing
    def ComputeAvgDt(t, dt_tol):
        dt = np.diff(t)
        aveDt = np.mean(dt)
        MaxDtDiff = max(abs(dt - aveDt))
        if MaxDtDiff > dt_tol:
            error(
                "Times are not equally spaced within dt_tol={}.\n".format(
                    dt_tol) +
                "Maximum deviation from the average is {}.".format(MaxDtDiff))
        return aveDt

    aveDt = ComputeAvgDt(t, dt_tol)

    # set up moving window
    if (t[-1] - t[0]) <= 6 * width:
        error("Total time-interval must be at least 6x larger than width")

    # setup output times
    if DT == None:
        t_out = t
    else:
        Tstart = t[0] if (Tstart is None) else Tstart
        if Tstart < t[0]:
            warn("--Tstart smaller than first time-step %g" % t[0])
        t_out = np.arange(Tstart, t[-1], step=DT)

    def K(x, width):
        return 1. / np.sqrt(math.pi) / width * np.exp(-x * x / (width * width))

    def K1(x, width):
        return -2. * x / np.sqrt(np.pi) / width**3 * np.exp(-x * x /
                                                            (width * width))

    def K2(x, width):
        return (4*x*x-2*width**2)/np.sqrt(np.pi)/width**5 \
               * np.exp(-x*x/(width**2))

    idxmin = 0
    idxmax = 0

    # limit t_out to range so that the kernel fits
    t_out = t_out[(t_out >= t[0] + cut_width) * (t_out <= t[-1] - cut_width)]

    output_array = np.empty((len(t_out), 1 + len(data[0])))
    for curr_idx in range(len(t_out)):
        curr_t = t_out[curr_idx]
        # update bounds for convolution
        while (t[idxmin] < curr_t - convolution_width):
            idxmin = idxmin + 1
        while idxmax < len(t) - 1 and t[idxmax] < curr_t + convolution_width:
            idxmax = idxmax + 1

        if (t[idxmax] > curr_t + convolution_width):
            idxmax = idxmax - 1

        if (abs(curr_t - t[0]) < cut_width or abs(curr_t - t[-1]) < cut_width):
            continue

        if (idxmax == idxmin):
            error("For t=%f, no data-points in interval.\n" % curr_t)

        # compute sums
        # s = Sum Kf,  w=Sum K
        f12 = K(curr_t - t[idxmin:idxmax + 1], width)
        w = np.sum(aveDt * (f12[0:-1] + f12[1:]))
        s = np.sum(aveDt * (data[idxmin:idxmax] * f12[0:-1].reshape(
            (-1, 1)) + data[idxmin + 1:idxmax + 1] * f12[1:].reshape((-1, 1))),
                   axis=0)

        # s1= Sum K1f, w1=Sum K1
        if (Deriv >= 1):
            f12 = K1(curr_t - t[idxmin:idxmax + 1], width)
            w1 = np.sum(aveDt * (f12[0:-1] + f12[1:]))
            s1 = np.sum(aveDt * (data[idxmin:idxmax] * f12[0:-1].reshape(
                (-1, 1)) + data[idxmin + 1:idxmax + 1] * f12[1:].reshape(
                    (-1, 1))),
                        axis=0)

        # s2= Sum K2f, w2=Sum K2
        if (Deriv >= 2):
            f12 = K2(curr_t - t[idxmin:idxmax + 1], width)
            w2 = np.sum(aveDt * (f12[0:-1] + f12[1:]))
            s2 = np.sum(aveDt * (data[idxmin:idxmax] * f12[0:-1].reshape(
                (-1, 1)) + data[idxmin + 1:idxmax + 1] * f12[1:].reshape(
                    (-1, 1))),
                        axis=0)

        ### combine for answer ###
        if (Deriv == 0):
            result = np.concatenate(((curr_t, ), s / w))
        elif (Deriv == 1):
            result = np.concatenate(((curr_t, ), s1 / w - s / w**2 * w1))
        elif (Deriv == 2):
            # Note: This formula produces visible noise
            # for non-uniformly spaced data even when the
            # 'w1' terms are included, which sould take
            # correct for non-uniform spacing.
            # (Harald, Nov 29, 2006)
            result = np.concatenate(((curr_t, ), s2 / w - 2. * s1 * w1 / w**2 -
                                     s * w2 / w**2 + 2. * s * w1 * w1 / w**3))
        output_array[curr_idx, :] = result[:]
    return output_array

# Inlined from BbhDiagnosticsImpl.py
def Compute_OrbitalFrequency(xA, xB, N, method="Fit", NSamples=None):
    r"""Given numpy arrays for the black hole locations xA, xB,
    perform fits to 2N+1 data-points around each point, and from
    the fit compute the instantaneous orbital frequency,
            Omega = r\times \dot{r} / r^2
    return t,Omega
    """

    def FitData(data, N, NSamples=None):
      """given a numpy array data with time as first column, perform fits 
        covering N points before and after each data-point for each column.
        return the fitted values and their first time-derivatives as a 
        numpy arrays, with first column time"""
      # collect output data, first column being time
      last_idx=len(data)-N-1

      if NSamples==None:
        step=1
      else:
        step=max(int(last_idx/NSamples),1)

      # The output times
      t_final = data[N:last_idx:step,0]

      x_tmp = []
      v_tmp = []
      for idx in range(N, last_idx, step):
        # Time at which we want the result
        T = data[idx,0]

        x = data[idx-N:idx+N+1,0]-T # Shift back to t=0
        y = data[idx-N:idx+N+1,1:]  # Fit all the columns at once!
        p0 = np.polyfit(x,y,2)
        x_tmp.append(p0[2]) # p0[2] is the constant part of fit
        v_tmp.append(p0[1]) # p0[1] is the linear part

      return np.column_stack((t_final,x_tmp)), np.column_stack((t_final,v_tmp))

    def FilterData(data,N):
      assert False, "The FilterData function is poorly tested"
      from scipy.ndimage import gaussian_filter1d
      mode = 'reflect'
      return gaussian_filter1d(data,N,mode=mode),\
             gaussian_filter1d(data,N,mode=mode,order=1)

    def SmoothDataImpl(data,N):
      t   = data[:,0]
      dat = data[:,1:]
      dt_interp = None
      return SmoothData(t,dat,N,DT=dt_interp), \
             SmoothData(t,dat,N,DT=dt_interp,Deriv=1)

    def SplineData(data):
      """Interpolating spline (no smoothing)"""
      from scipy.interpolate import splrep,splev
      t   = data[:,0]
      spline_x = splrep(t, data[:,1])
      spline_y = splrep(t, data[:,2])
      spline_z = splrep(t, data[:,3])
      dx = splev(t, spline_x, der=1)
      dy = splev(t, spline_y, der=1)
      dz = splev(t, spline_z, der=1)
      v = np.vstack((t,dx,dy,dz)).T
      return data, v

    if NSamples is not None and method!='Fit':
        error("NSamples only works with 'Fit'")

    if method=="Fit":
      data = np.column_stack((xA,xB[:,1:4]))
      xs_fit,vs = FitData(data,N,NSamples=NSamples)
      xA_fit = xs_fit[:,0:4]
      xB_fit = xs_fit[:,[0,4,5,6]]
      vA = vs[:,0:4]
      vB = vs[:,[0,4,5,6]]
    elif method=="Filter":
      xA_fit,vA=FilterData(xA,N)
      xB_fit,vB=FilterData(xB,N)
    elif method=="Smooth":
      xA_fit,vA=SmoothDataImpl(xA,N)
      xB_fit,vB=SmoothDataImpl(xB,N)
    elif method=="Spline":
      xA_fit,vA=SplineData(xA)
      xB_fit,vB=SplineData(xB)
    else:
      error("Don't know method '{}'".format(method))

    # Compute Orbital frequency (r x dr/dt)/r^2
    t  = xA_fit[:,0]
    dr = xA_fit[:,1:] - xB_fit[:,1:]
    dr2 = norm(dr,axis=1)**2   #slightly inefficient
    dv = vA[:,1:] - vB[:,1:]
    Omega = np.cross(dr,dv)/dr2[:,np.newaxis]
    return t,Omega

##########################################################################
### Functions for unloading input and calculating related quantities
##########################################################################

# Choose the earliest tmin **after** the junk radiation.
# This function computes the absolute value of the second derivative of
# omega, ODblDot; it then takes a running average of this abs. value
# over avgp points; separately, it averages ODblDot from time 500 until the
# end and calls this yfin; with this, it calculates drop which is the value of
# ODblDot @t=0 - yfin; a threshold of ODblDot, ythr, is the sum of yfin and
# threshp*drop; then tmin is the time at which ODblDot first goes under this
# threshold, ythr, + tshiftp(set @200); however, tmin is capped at 500;
# in other words, tmin=min(tmin, 500). (Author: Robert McGehee)
def FindTmin(t_temp2, dOmegadt, max_tmin):
  # Compute the 2nd deriv. of Omega, take its abs. value
  d2Omegadt = (dOmegadt[2:]-dOmegadt[0:-2])/(t_temp2[2:]-t_temp2[0:-2])
  t=t_temp2[1:-1]
  ODblDot=abs(d2Omegadt)

  # Compute the running average of ODblDot with a given avg param
  avgp=10       #How many points in the running average (has to be an even #)
  y= [0.00]*size(t)
  for i in range(0,avgp//2):
      for j in range(0,avgp+1):
          if j!=i:
              y[i]=y[i]+ODblDot[j]
      y[i]=y[i]/avgp
  for i in range(avgp//2,size(t)-avgp//2):
      for j in range(i-avgp//2,i+avgp//2+1):
          if j!=i:
              y[i]=y[i]+ODblDot[j]
      y[i]=y[i]/avgp
  for i in range(size(t)-avgp//2,size(t)):
      for j in range(size(t)-avgp-1,size(t)):
          if j!=i:
              y[i]=y[i]+ODblDot[j]
      y[i]=y[i]/avgp

  # Compute the average,yfin, of ODblDot from t=500 until the end
  sum=0.0000000
  for i in range(500,size(t)):
      sum=sum+ODblDot[i]
  yfin=sum/(size(t)-500)
  drop=ODblDot[0]-yfin #the drop in magnitude determines a threshold
  threshp=0.001        #parameter which controls how low the threshold ythr is
  ythr=yfin+threshp*drop

  # Loop through y to find first time where y < ythr
  k=0
  while (k<size(t)) and (y[k]>=ythr):
      k=k+1
  if k==size(t):
      k=k-1
  tshiftp=200          #parameter controls how much "safety" is added to tmin
  tmin=t[k]+tshiftp

  # Don't allow tmin to be arbitrarily large
  return min(tmin, max_tmin)

def ComputeOmegaAndDerivsFromFile(TrajA, TrajB):
  '''Compute Omega, dOmega, ellHat and nHat'''

  # While method="Smooth" appears to be more accurate, "Fit"
  # does a slightly better job of reducing eccentricity.
  t_raw, OmegaVec = Compute_OrbitalFrequency(TrajA, TrajB, 10, method='Fit')
  Omega_raw = norm(OmegaVec, axis=1)

  # Compute derivative
  dOmega    = (Omega_raw[2:]-Omega_raw[0:-2])/(t_raw[2:]-t_raw[0:-2])
  Omega     = Omega_raw[1:-1]
  OmegaVec  = OmegaVec[1:-1]
  t = t_raw[1:-1]

  return t,Omega,dOmega,OmegaVec

def interpolateVector(t_old,v,t_new):
  assert len(v[0])==3, \
    "This function expects a list of 3d vectors, got {}d".format(len(v[0]))
  res = np.zeros((len(t_new),3))

  for i in range(3):
    intp =InterpolatedUnivariateSpline(t_old,v[:,i],k=5)
    res[:,i]=intp(t_new)

  return res

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def GetVarsFromSpinData(sA, sB, XA, XB, mA, mB, OmegaVec, t, tmin):
  '''Compute quantities that depend on the mass and spin'''

  t_idx = find_nearest(t, tmin)
  Omega_tmin = norm(OmegaVec[t_idx,:])

  alpha_intrp, S_0_perp_n = ComputeSpinTerms(XA,XB,sA,sB,mA,mB,OmegaVec,t)

  nu = mA * mB / (mA + mB) ** 2
  m = mA + mB

  x_i = (m * Omega_tmin) ** (2. / 3)
  T_merge = tmin + 5 / 256.*m ** 2 / nu * x_i ** (-4)
  Amp = (5. / 256.*m ** 2 / nu) ** (3. / 8) * m

  return alpha_intrp, S_0_perp_n, T_merge, Amp

def ComputeSpinTerms(TrajA, TrajB, sA, sB, mA, mB, OmegaVec, t):
    t_s = sA[:,0]

    def PointwiseDot(A,B):
      '''Dot product for each element in two lists of vectors'''
      return np.array([np.dot(a,b) for a,b in zip(A,B)])
    def PointwiseNormalize(A):
      '''Normalization for each element in a list of vectors'''
      return A/norm(A, axis=1)[:,np.newaxis]

    ellHat    = PointwiseNormalize(OmegaVec)
    nHat_raw  = PointwiseNormalize(TrajA[:,1:4] - TrajB[:,1:4])
    nHat      = PointwiseNormalize(interpolateVector(TrajA[:, 0],
                                                     nHat_raw, t))
    lambdaHat = PointwiseNormalize(cross(ellHat, nHat, axis=1))

    S_0 = (1 + mB / mA) * sA[:,1:4] + (1 + mA / mB) * sB[:,1:4]
    S_0 = interpolateVector(t_s, S_0, t)
    S_0_perp = S_0 - PointwiseDot(S_0, ellHat)[:,np.newaxis] * ellHat
    S_0_perp_n = norm(S_0_perp, axis=1)

    # alpha is an angle that appears in rewriting spin-induced
    # oscillations in orbital quantities for spin-precessing
    # binaries.
    # See https://arxiv.org/abs/1012.1549 (paper 1)
    # and https://arxiv.org/abs/2410.05531 (paper 2)
    # Note that alpha here is NOT equal to what is called alpha in
    # either paper 1 or paper 2.  Instead, alpha here is equal to
    # \bar(alpha) in paper 2.  Note also that Eq. 49 in paper 1 should
    # not have a factor of 2 in the denominator.
    #
    # Harald suspects that the alpha-calculation becomes
    # inaccurate if S_0_perp is very close to zero, i.e. for
    # nearly aligned-spin systems.
    alpha = np.arctan2(PointwiseDot(S_0_perp, lambdaHat), \
                       PointwiseDot(S_0_perp, nHat))
    alpha_intrp = InterpolatedUnivariateSpline(t, alpha, k=5)

    # Sanity check
    # Checks Eq. 48 in paper 1.
    # Note that in the notation of paper 2,
    # sin(2*bar(alpha)) = 2 sin(2\bar\Omega t + \gamma).
    expected = PointwiseDot(S_0_perp, nHat) \
               * PointwiseDot(S_0_perp, lambdaHat)  # LHS of Eq. (48)
    got = sin(2 * alpha) * S_0_perp_n ** 2 * 0.5    # rewrite w/ alpha

    # If the absolute magnitude of these expressions is small,
    # then they don't matter in the fit, therefore, the rtol value
    # is allowed to be large.  (value 1e-3 chosen to avoid the
    # errors reported in in
    # https://github.com/sxs-collaboration/vacuum-call-notes/
    #    issues/152#issuecomment-2607354382)
    np.testing.assert_allclose(got,expected,rtol=1e-3,atol=1e-8)

    return alpha_intrp, S_0_perp_n

def computeOmegaGuessAndFilterTraj(t, dOmega_dt, IDparam_omega0,
                                   pad_length=10000, peak_cutoff_factor=.2):
  '''
  Take an FFT of the trajectory data, and from that calculate an initial guess
  for the omega fit parameter and a version of the trajectory with high
  frequency content removed. This modified trajectory data is used for fitting
  if '--freq_filter' is specified when running.

  Input:
    pad_length -- number of 0s to pad each end of time domain data with
    peak_cutoff_factor -- amplitude below which local maxima in frequency
                          spectrum are ignored for peak finding, as a
                          fraction of the global maximum

  Returns the omega guess, time array for the filtered trajectory, and the
  filtered trajectory. If filtering was unsuccessful, return Nones instead of
  the time array and trajectory.
  '''

  t_grid = t
  pre_FFT = dOmega_dt
  # If input time steps are not equally spaced, interpolate
  # omega dot onto an even time grid.
  if not np.all(t[1:]-t[:-1] == t[1]-t[0]):
    t_grid = np.linspace(t[0], t[-1], len(t))
    pre_FFT = CubicSpline(t, dOmega_dt)(t_grid)
  # Apply window function, detrending, and zero padding to trajectory data
  window = np.hamming(len(pre_FFT))
  detrend_diff = pre_FFT - signal.detrend(pre_FFT)
  signal_length = len(pre_FFT) + 2*pad_length
  pre_FFT = signal.detrend(pre_FFT)*window
  pre_FFT = np.pad(pre_FFT, pad_length)

  # Transform the processed trajectory data into frequency space
  # and compute the frequency space grid
  post_FFT = np.abs(np.fft.rfft(pre_FFT))
  FFT_freq = np.fft.rfftfreq(signal_length, d=t_grid[1]-t_grid[0])*2*pi

  peak_indices,_ = signal.find_peaks(post_FFT,
                                  height=peak_cutoff_factor*np.max(post_FFT))
  minima_indices,_ = signal.find_peaks(-post_FFT)
  peak_freqs = FFT_freq[peak_indices]
  minimum_freqs = FFT_freq[minima_indices]
  viable_peaks = peak_freqs[[f >= 0.6*IDparam_omega0 and
                             f <= 1.4*IDparam_omega0 for f in peak_freqs]]

  # Pick peak of FFT in frequency space to be guess for omega_0. Only consider
  # frequencies within 40% of the initial data omega0. If there is more than
  # one peak or there are no peaks in this range, default to 0.8*omega0 as a
  # guess and don't filter.
  if len(viable_peaks) != 1:
    return 0.8*IDparam_omega0, None, None
  # Filtering can only be done if a minimum occurs after the guess peak
  # frequency. To filter we set the frequency spectrum for frequencies higher
  # than the first minimum after guess peak to 0.
  elif len(minimum_freqs[minimum_freqs > viable_peaks[0]]) > 0:
    filter_threshold = minimum_freqs[minimum_freqs > viable_peaks[0]][0]
    filtered_FFT = np.fft.rfft(pre_FFT)
    filtered_FFT[FFT_freq > filter_threshold] = 0
    filtered_dOmegadt = (np.fft.irfft(filtered_FFT, n=signal_length)
                         .real[pad_length:-pad_length] / window) \
                         + detrend_diff
    return viable_peaks[0], t_grid, filtered_dOmegadt
  else:
    return viable_peaks[0], None, None

##########################################################################
### Functions for performing fits and computing updates
##########################################################################

# fit the data (t,y) to the model F[p,t], by least-squares fitting the params
# p.
def fit(t, y, F, p0, bounds, jac, name):

    # t,y -- arrays of equal length, having t and y values of data
    # F   -- function: F(p,t) taking parameters p and set of t-values
    # p0  -- starting values of parameters
    residual = lambda p,t,y: F(p,t) - y
    # the jacobian function passed to least_squares must have the form
    # jac(p, t, y)
    jacfunc = lambda p,t,y: [jac(p,t_i) for t_i in t]
    res = optimize.least_squares(
      residual,
      method = "dogbox",
      x0 = p0,
      jac = jacfunc,
      bounds = bounds,
      args = (t,y),
    )

    if not res.success:
      if name == "F2cos2_SS":
        error("minimize failed in {}: {}".format(name, res.message))
      else:
        warning("minimize failed in {}: {}".format(name, res.message))

    return res.x, res.cost, res.success

#TODO: this should be updated now that we bound omega
def CheckPeriastronAdvance(omega,Omega0,name,ecc,ecc_str,summary):
  # See discussion just before section IV of arXiv:1012.1549.  The
  # idea here is that omega/Omega0 should be exactly equal to 1 for a
  # pure Newtonian orbit, but for GR or PN orbits omega/Omega0 should
  # be slightly less than 1 because of periastron advance.
  # Furthermore, omega/Omega0 should be independent of eccentricity
  # for small eccentricity.
  #
  # So if omega/Omega0 is much larger than 1, then this is unphysical
  # because it represents negative periastron advance.
  # If omega/Omega0 is much smaller than 1, then this is also very large
  # periastron advance and unphysical (one can argue where the cutoff should
  # go).
  #
  # The big question is what should we do if we encounter these
  # unphysical situations. Currently we flag for a human to look at
  # by setting ecc=9.99999.  Note that such a large value of eccentricity
  # will currently cause EccReduce.pm to report a failure.
  #
  # These situations might occur when
  # eccentricity is so small that we cannot find an accurate fit.
  # Currently we have no reliable way of detecting inaccurate fits
  # (other than the res/B check); but with varpro we will have an
  # error estimate for ecc, so perhaps the periastron check will be
  # encountered less often in that case.
  #
  if omega/Omega0<0.5 or omega/Omega0>1.2:
    ecc = 9.99999
    ecc_str = str(ecc)
    err="OmegaDotEccRemoval: {name}-fit resulted in large (>1.2) or " \
        "negative (<0.5) periastron advance:\nomega/Omega0={omegafrac}." \
        " This is likely wrong (or a bad omega fit from too small ecc " \
        "oscillation amplitudes),\nso eccentricity was set to {ecc}\n" \
        .format(name=name,omegafrac=omega/Omega0,ecc=ecc)
    warning(err)
    summary.write(err)
  return ecc,ecc_str

def ComputeUpdate(Omega0, adot0, D0,
                  Tc, B, omega,
                  phi, phi_tref,
                  name, tmin, tmax, rms,
                  Improved_Omega0_update, check_periastron_advance,
                  params_output_dir, Source, summary,
                  B_std_dev = None, omega_std_dev = None):
    '''
    Computes and returns eccentricity and corrections to be added to initial
    data parameters Omega_0, adot, and D. Corrected initial data removes
    spurious eccentricity from the evolution. Only two out of the three
    corrections returned should be applied to initial data before restarting
    evolution. See arXiv:1012.1549 for derivation.
    '''

    delta_adot0=B/(2.*Omega0)*cos(phi)
    delta_Omega0=-B*omega/(4.*Omega0*Omega0)*sin(phi)
    if(Improved_Omega0_update):
        # extra factor Omega0/omega in delta_Omega0
        delta_Omega0=-B/(4.*Omega0)*sin(phi)

    delta_D0=-B*D0*omega*sin(phi)/(2*Omega0*(Omega0**2+2/D0**3))
    ecc=B/(2.*Omega0*omega)
    ecc_str="{:7.7f}".format(ecc)
    ecc_std_dev = None
    if (B_std_dev != None) and (omega_std_dev != None):
      ecc_std_dev = sqrt(B**2 * omega_std_dev**2 + omega**2 * B_std_dev**2) \
                  / (2 * Omega0 * omega * omega)

    if rms/B>0.4:
      # See discussion just before section IV of arXiv:1012.1549.
      # The idea here is that if the oscillations are very small
      # (meaning that B is small enough that it is smaller than the residual
      # of the fit), then the fit effectively does not see the oscillations.
      # So the eccentricity estimate is not very accurate, and we report
      # a bound here.
      #
      # When we switch to varpro, we will presumably have an error bound
      # on B, and instead of rms/B > 0.4 we can use some criterion like
      # (error bound on B) < B.
      summary.write("Large residual of ecc-fit, report bound on ecc\n")
      # for a sine-wave, the rms is 1/2 its amplitude.  Therefore, assume
      # that the amplitude of a bad-fit is 2*rms.  Double that, for safety
      # and because we have to disregard the term omega/Omega0
      ecc=4.*rms/(2.*Omega0*Omega0)
      ecc_str="<{:.1e}".format(ecc)
    elif check_periastron_advance:
      ecc,ecc_str = CheckPeriastronAdvance(omega,Omega0,name,
                                           ecc,ecc_str,summary)
    pi=np.pi
    summary.write("%s:  %+11.8f    %+11.8f     " \
                  "%+9.6f   %s    %9.6f       %9.6f\n"
        %(name,delta_Omega0,delta_adot0,delta_D0,ecc_str,
          (phi-pi/2.)%(2.*pi),
          (phi_tref-pi/2.)%(2.*pi)  # mean anomaly, in [0, 2pi]
      ))

    if params_output_dir:
      f = open(os.path.join(params_output_dir, "Params_%s.dat"%name), 'w')
      f.write("# EccRemoval.py utilizing orbital frequency "\
              "Omega, fit %s\n"%name)
      f.write("# Source file=%s\n" % Source)
      f.write("# Omega0=%10.8g, adot0=%10.8g, D0=%10.8g\n"%(Omega0,adot0,D0))
      f.write("# Fitting interval [tmin,tmax]=[%g,%g]\n"%(tmin,tmax))
      f.write("# oscillatory part of fit: (B,omega,phi)=(%g,%g,%g)\n"
              %(B,omega,phi))
      f.write("# ImprovedOmega0Update=%s\n"%Improved_Omega0_update)
      f.write("# [1] = Omega0\n")
      f.write("# [2] = 1e4 adot0\n")
      f.write("# [3] = D0\n")
      f.write("# [4] = ecc\n")
      f.write("%10.12f\t%10.12f\t%10.12f\t%10.12f\n"
              %(Omega0, 1e4*adot0, D0, ecc))
      f.write("%10.12f\t%10.12f\t%10.12f\t%10.12f\n"
              %(Omega0+delta_Omega0, 1e4*(adot0+delta_adot0),
                D0+delta_D0, 0.))
      f.close()

      f = open("Fit_%s.dat"%name, 'w')
      f.write("# EccRemoval.py utilizing orbital frequency Omega, fit %s\n"
              %name)
      f.write("# Source file=%s\n" % Source)
      f.write("# Omega0=%10.8g, adot0=%10.8g\n"%(Omega0,adot0))
      f.write("# Fitting interval [tmin,tmax]=[%g,%g]\n"%(tmin,tmax))
      f.write("# oscillatory part of fit: (B,omega,phi)=(%g,%g,%g)\n"
              %(B,omega,phi))
      f.write("# [1] = Tstart\n")
      f.write("# [2] = Tend\n")
      f.write("# [3] = Tc\n")
      f.write("# [4] = B\n")
      f.write("# [5] = omega\n")
      f.write("# [6] = sin(phi)\n")
      f.write("# [7] = rms residual\n")
      f.write("# [8] = rms residual/B\n")
      f.write("# [9] = omega/Omega0\n")
      f.write("# [10] = ecc\n")

      f.write("%g %g %g %g %g %g %g %g %g %g\n"
              %(tmin, tmax, Tc, B, omega, sin(phi),
                rms, rms/B, omega/Omega0, ecc))
      f.close()

    return ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev

def plot_fitted_function(p, F, x, y, idxFit, idxPlot, idxZoom, name, style,
                         axes):
    # add the plot to all four panels
    # p, F: params and fitting function
    # x, y - complete data
    # idxFit -- indices used in fit, for indicating fit-interval
    # idxPlot -- indices to be plotted in left windows
    # idxZoom -- indices to be plotted in right windows
    # name -- string placed into legend
    # style - plot-style

    xBdry=array([x[idxFit][0], x[idxFit][-1]])
    yBdry=array([y[idxFit][0], y[idxFit][-1]])

    ((ax1,ax2),(ax3,ax4)) = axes

    ## Top-left plot: fit
    ax1.plot(x[idxPlot],1e6*F(p,x[idxPlot]),style, label=name)

    # add point at begin and end of fit interval:
    ax1.plot(xBdry,1e6*F(p,xBdry),'o')

    # bottom-left plot: residual
    data=1e6*(F(p,x)-y)
    ax3.plot(x[idxPlot],data[idxPlot],style,label=name)
    ax3.plot(xBdry,1e6*(F(p,xBdry)-yBdry),'o')
    ylim3=ax3.get_ylim()
    miny=min(ylim3[0], min(data[idxFit]))
    maxy=max(ylim3[1], max(data[idxFit]))
    ax3.set_ylim(miny, maxy)
    ax3.set_title("1e6 residual")

    ## Top-right plot: zoom of fit
    ax2.plot(x[idxZoom],1e6*F(p,x[idxZoom]),style, label=name)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15,1.3)
           ,labelspacing=0.25,handletextpad=0.0,fancybox=True
           )

    # add point at begin of fit interval:
    ax2.plot([xBdry[0]],[1e6*F(p,xBdry)[0]],'o')

    # bottom-right plot: zoom of residual
    ax4.plot(x[idxZoom],1e6*(F(p,x[idxZoom])-y[idxZoom]),style,label=name)
    ax4.plot([xBdry[0]],[(1e6*(F(p,xBdry)-yBdry))[0]],'o')
    ax4.set_title("1e6 residual")

    return

def make_full_plot(Source,t,dOmegadt,tmin,tmax,idxFit,
                   params,funcs,names,linestyles,plotfilename):
  #=== set plotting intervals
  idxPlot= (t> tmin - 0.2*(tmax-tmin)) & (t< tmax + 0.35*(tmax-tmin) )
  idxZoom=(t< tmin + 0.2*(tmax-tmin))
  tPlot=t[idxPlot]
  dOmegadtPlot=dOmegadt[idxPlot]

  fig,axes = plt.subplots(2,2)
  fig.text(0.5, 0.95,
           "%s [%g, %g]" % (os.path.basename(Source), tmin, tmax),
           color='b', size='large', ha='center')

  ((ax1,ax2),(ax3,ax4)) = axes

  #=== Top left plot dOmega/dt ===
  ax1.plot(tPlot[tPlot >= tmin], 1e6 * dOmegadtPlot[tPlot >= tmin], 'k',
                 label="dOmega/dt", linewidth=2)
  xlim1=ax1.get_xlim()
  ylim1=ax1.get_ylim()
  ax1.plot(tPlot, 1e6 * dOmegadtPlot, 'k',label="dOmega/dt", linewidth=2)
  xlim2=ax1.get_xlim()
  ax1.set_xlim(xlim2)
  ax1.set_ylim(ylim1)
  ax1.set_title("1e6 dOmega/dt")

  #==== bottom left ====
  # set x-axes to top-left scale, and y-axes to something small
  # as initial conditions for adding line by line above
  ax3.set_xlim(xlim1)
  ax3.set_ylim([-1e-10, 1e-10])

  #==== Top right plot -- zoom of dOmega/dt ====
  ax2.plot(t[idxZoom], 1e6 * dOmegadt[idxZoom],
           'k', linewidth=2)  #  label="dOmega/dt",
  ax2.set_title("1e6 dOmega/dt           .") # extra space to avoid legend

  # Plot individual fits
  for (p,f,name,linestyle) in zip(params,funcs,names,linestyles):
    plot_fitted_function(p, f, t, dOmegadt, idxFit, idxPlot,
                         idxZoom, name, linestyle, axes)

  # zoom out of the y-axis in the lower left panel by 15%
  ylim1 = ax3.get_ylim()
  Deltay=ylim1[1]-ylim1[0]
  ax3.set_ylim([ylim1[0]-0.15*Deltay, ylim1[1]+0.15*Deltay])

  # adjust margins of figure
  fig.subplots_adjust(left=0.09, right=0.95, bottom=0.07, hspace=0.25)

  plt.savefig(plotfilename)

class FitBounds:
  """Set bounds for some of the variables being fit"""
  def __init__(self, tmax, IDparam_omega0):
    # Time to coalescence must be greater than the fit interval
    self.Tc    = [tmax+2.0*np.spacing(abs(tmax)), np.inf]
    # Frequency should be positive and similar to the initial frequency
    self.omega = [0.6*IDparam_omega0, 1.4*IDparam_omega0]
    # Amplitude of the cos term is chosen to be positive (negative is phi+=pi)
    self.B     = [1e-16, np.inf]
    # phi is a phase, trivially bounded
    self.phi   = [0, 2*pi]
    # For various variables that have no limits
    self.none  = [-np.inf, np.inf]

def performNonspinFits(t,dOmegadt,idxFit,tFit,dOmegadtFit,
                       tmin,tmax,tref,IDparam_omega0,
                       IDparam_D0,IDparam_adot0,q,omega_guess,opt_tmin,
                       opt_improved_Omega0_update,check_periastron_advance,
                       params_output_dir,plot_output_dir,Source,summary):
  '''
  Fit the eccentricity estimator given in arXiv:1012.1549 to the given
  trajectory, not including spin terms, and compute initial data from the
  resulting best fit parameters. A series of fits is performed, each one
  including a new term from the estimator model. Returns eccentricity and
  initial data corrections based on the best fit to the complete model.
  '''

  # Set bounds for some of the variables
  lim = FitBounds(tmax, omega_guess)

  # 0PN approximation
  Tmerger = 5. / (64.*(1. - ((q-1.) / (q+1.))**2.)*IDparam_omega0**(8./3.))

  # fit a0(T-t)^(-11./8)
  F1 = lambda p,t: p[1]*(p[0]-t)**(-11/8)
  p0 = [Tmerger, 1e-5]
  pBounds = list(zip(*[lim.Tc, lim.none]))
  jac = lambda p,t: [ (-11/8)*p[1]*(p[0]-t)**(-19/8),
                      (p[0]-t)**(-11/8) ]
  pF1, rmsF1, F1_status = fit(tFit, dOmegadtFit, F1, p0, pBounds, jac, "F1")

  #================

  F1cos1 = lambda p,t: p[1]*(p[0]-t)**(-11/8) + p[2]*cos(p[3]*t+p[4])
  jac = lambda p,t: [ (-11/8)*p[1]*(p[0]-t)**(-19/8),
                      (p[0]-t)**(-11/8),
                      cos(p[3]*t + p[4]),
                      -p[2]*sin(p[3]*t + p[4])*t,
                      -p[2]*sin(p[3]*t + p[4]) ]

  rmsF1cos1 = 2*rmsF1
  pF1cos1 = pF1
  # try a few initial guesses for the phase to ensure good convergence
  for phi in range(0,6):
    p0 = [pF1[0], pF1[1], rmsF1, omega_guess, phi]
    pBounds = list(zip(*[lim.Tc, lim.none, lim.B, lim.omega, lim.phi]))
    ptemp, rmstemp, F1cos1_status = fit(tFit, dOmegadtFit, F1cos1,
                                        p0, pBounds, jac, "F1cos1")
    if(rmstemp<rmsF1cos1):
        rmsF1cos1=rmstemp
        pF1cos1=ptemp

  #================

  F1cos2 = lambda p,t: p[1]*(p[0]-t)**(-11/8) + p[2]*cos(p[3]*t+p[4]+p[5]*t*t)
  jac = lambda p,t: [ (-11/8)*p[1]*(p[0]-t)**(-19/8),
                      (p[0]-t)**(-11/8),
                      cos(p[3]*t + p[4] + p[5]*t*t),
                      -p[2]*sin(p[3]*t + p[4] + p[5]*t*t)*t,
                      -p[2]*sin(p[3]*t + p[4] + p[5]*t*t),
                      -p[2]*sin(p[3]*t + p[4] + p[5]*t*t)*t*t ]
  p0 = [pF1cos1[0], pF1cos1[1], pF1cos1[2], pF1cos1[3], pF1cos1[4], 0]
  pBounds = list(zip(*[lim.Tc, lim.none, lim.B,
                       lim.omega, lim.phi, lim.none]))
  pF1cos2, rmsF1cos2, F1cos2_status = fit(tFit, dOmegadtFit, F1cos2,
                                          p0, pBounds, jac, "F1cos2")

  #================

  # F2cos2 = a0(Tc-t)^(-11/8)+a1(Tc-t)^(-13/8)+Bcos(omega t+phi+b t^2)
  F2cos2 = lambda p,t: p[1]*(p[0]-t)**(-11/8) + p[2]*(p[0]-t)**(-13/8) \
                       + p[3]*cos(p[4]*t+p[5]+p[6]*t*t)
  jac = lambda p,t: [ (-11/8)*p[1]*(p[0]-t)**(-19/8) + \
                        (-13/8)*p[2]*(p[0]-t)**(-21/8),
                      (p[0]-t)**(-11/8),
                      (p[0]-t)**(-13/8),
                      cos(p[4]*t + p[5] + p[6]*t*t),
                      -p[3]*sin(p[4]*t + p[5] + p[6]*t*t)*t,
                      -p[3]*sin(p[4]*t + p[5] + p[6]*t*t),
                      -p[3]*sin(p[4]*t + p[5] + p[6]*t*t)*t*t ]
  p0 = [pF1cos2[0], pF1cos2[1], 0., pF1cos2[2],
        pF1cos2[3], pF1cos2[4], pF1cos2[5]]
  pBounds = list(zip(*[lim.Tc, lim.none, lim.none, lim.B, lim.omega,
                       lim.phi, lim.none]))
  pF2cos2, rmsF2cos2, F2cos2_status = fit(tFit, dOmegadtFit, F2cos2,
                                          p0, pBounds, jac, "F2cos2")

  #==== output all three residuals
  summary.write("tmin=%f  (determined by "%tmin)
  if(opt_tmin==None):
    summary.write(" fit)\n")
  else:
    summary.write(" option)\n")

  summary.write("""
FIT SUCCESS
    F1  %d
    F1cos1  %d
    F1cos2  %d
    F2cos2  %d
""" % (F1_status, F1cos1_status, F1cos2_status, F2cos2_status))

  summary.write("""
RESIDUALS
    F1cos1 rms=%g   \tF1cos2 rms=%g
    F2cos2 rms=%g

DIAGNOSTICS
                Tc         B     sin(phi)    rms/B   omega/Omega0
""" %(rmsF1cos1,rmsF1cos2,rmsF2cos2,))

  # add first two fields for remaining fits
  tmp="%-8s %10.1f   %6.2e    %6.4f     %5.3f      %5.3f\n"
  summary.write(tmp%("F1cos1", pF1cos1[0], pF1cos1[2], sin(pF1cos1[4]),
                               rmsF1cos1/pF1cos1[2],
                               pF1cos1[3]/IDparam_omega0))
  summary.write(tmp%("F1cos2", pF1cos2[0], pF1cos2[2], sin(pF1cos2[4]),
                               rmsF1cos2/pF1cos2[2],
                               pF1cos2[3]/IDparam_omega0))
  summary.write(tmp%("F2cos2", pF2cos2[0], pF2cos2[3], sin(pF2cos2[5]),
                               rmsF2cos2/pF2cos2[3],
                               pF2cos2[4]/IDparam_omega0))

  #==== compute updates and generate EccRemoval_FIT.dat files
  summary.write("""
ECCENTRICITY AND UPDATES
         delta_Omega0   delta_adot0     delta_D0     ecc      """\
"""mean_anomaly(0)  mean_anomaly(tref)
""")

  ComputeUpdate(IDparam_omega0, IDparam_adot0, IDparam_D0,
                pF1cos1[0], pF1cos1[2], pF1cos1[3],
                pF1cos1[4], pF1cos1[3]*tref+pF1cos1[4],
                # The above two arguments are arg of cos(..) at t=0 and tref
                "F1cos1",tmin,tmax,rmsF1cos1,
                opt_improved_Omega0_update, check_periastron_advance,
                params_output_dir, Source, summary)

  ComputeUpdate(IDparam_omega0, IDparam_adot0, IDparam_D0,
                pF1cos2[0], pF1cos2[2], pF1cos2[3],
                pF1cos2[4], pF1cos2[4]+pF1cos2[3]*tref+pF1cos2[5]*tref*tref,
                "F1cos2",tmin,tmax,rmsF1cos2,
                opt_improved_Omega0_update, check_periastron_advance,
                params_output_dir, Source, summary)

  ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev = ComputeUpdate(
    IDparam_omega0, IDparam_adot0, IDparam_D0, pF2cos2[0], pF2cos2[3],
    pF2cos2[4], pF2cos2[5], pF2cos2[5]+pF2cos2[4]*tref+pF2cos2[6]*tref*tref,
    "F2cos2",tmin,tmax,rmsF2cos2, opt_improved_Omega0_update,
    check_periastron_advance, params_output_dir, Source, summary)

  if plot_output_dir:
    make_full_plot(Source,t,dOmegadt,tmin,tmax,idxFit,
                   [pF1cos1,pF1cos2,pF2cos2],
                   [F1cos1,F1cos2,F2cos2],
                   ['F1cos1','F1cos2','F2cos2'],
                   [':',':',"--"],
                   os.path.join(plot_output_dir, "FigureEccRemoval.pdf"))

  return ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev

def performSpinFits(t,dOmegadt,idxFit,tFit,OmegaFit,dOmegadtFit,T_merge,
                    Amp,alpha_intrp,S_0_perp_n,tmin,tmax,tref,IDparam_omega0,
                    IDparam_D0,IDparam_adot0,omega_guess,opt_tmin,
                    opt_improved_Omega0_update,check_periastron_advance,
                    params_output_dir,plot_output_dir,Source,summary):
  '''
  Fit the eccentricity estimator given in arXiv:1012.1549 to the given
  trajectory, including spin terms, and compute initial data from the
  resulting best fit parameters. A series of fits is performed, each one
  including a new term from the estimator model. Returns eccentricity and
  initial data corrections based on the best fit to the complete model.
  '''

  # Set bounds for some of the variables
  lim = FitBounds(tmax, omega_guess)

  # Fit *Omega*
  OmegaFunc = lambda p,t: p[1]*(p[0]-t)**(-3/8)   # power law form for Omega
  jac = lambda p,t: [ (-3/8)*p[1]*(p[0]-t)**(-11/8),
                      (p[0]-t)**(-3/8) ]
  p0 = [T_merge, Amp]
  pBounds = list(zip(*[lim.Tc, lim.none]))
  pOmega, rmsOmega, Omega_status = fit(tFit, OmegaFit, OmegaFunc,
                                       p0, pBounds, jac, "OmegaFunc")
  FitTc = pOmega[0]  # Fit time to coalescence

  #================

  F1_SS = lambda p,t: p[0]*(FitTc-t)**(-11/8)
  jac = lambda p,t: [ (FitTc-t)**(-11/8) ]
  p0 = [1e-5]
  pBounds = list(zip(*[lim.none]))
  pF1_SS, rmsF1_SS, F1_SS_status = fit(tFit, dOmegadtFit, F1_SS,
                                       p0, pBounds, jac, "F1_SS")

  #================

  # This function is *only* used to provide an initial guess for F1cos1_SS
  # We use it because generally the amplitude and time to merger one obtains
  # with the "usual" F1cos1 is highly degenerate and can give strange values.
  F1cos1_helper = lambda p,t: p[0]*(FitTc-t)**(-11/8) \
                              + p[1]*cos(p[2]*t + p[3])
  jac = lambda p,t: [ (FitTc-t)**(-11/8),
                      cos(p[2]*t + p[3]),
                      -p[1]*sin(p[2]*t + p[3])*t,
                      -p[1]*sin(p[2]*t + p[3]) ]

  rmsF1cos1_helper = 2*rmsF1_SS
  # try a few initial guesses for the phase to ensure good convergence
  for phi in range(0, 6):
    p0 = [pF1_SS[0], rmsF1_SS, omega_guess, phi]
    pBounds = list(zip(*[lim.none, lim.B, lim.omega, lim.phi]))
    ptemp, rmstemp,status = fit(tFit, dOmegadtFit, F1cos1_helper,
                                p0, pBounds, jac, "F1cos1_helper")

    if(rmstemp < rmsF1cos1_helper):
      rmsF1cos1_helper = rmstemp
      pF1cos1_helper = ptemp

  #================

  rmsF1cos1_SS = 3*rmsF1_SS

  F1cos1_SS = lambda p,t : p[0]*(FitTc-t)**(-11/8) + p[1]*cos(p[2]*t + p[3]) \
                           - p[4]*sin(2*alpha_intrp(t) + p[5])
  jac = lambda p,t: [ (FitTc-t)**(-11/8),
                      cos(p[2]*t + p[3]),
                      -p[1]*sin(p[2]*t + p[3])*t,
                      -p[1]*sin(p[2]*t + p[3]),
                      -sin(2*alpha_intrp(t) + p[5]),
                      -p[4]*cos(2*alpha_intrp(t) + p[5]) ]
  for phi in range(0, 6):
    p0 = [pF1cos1_helper[0], pF1cos1_helper[1],
          pF1cos1_helper[2], pF1cos1_helper[3],
          0.5*S_0_perp_n[0]**2 * (IDparam_omega0/IDparam_D0)**2, phi]
    pBounds = list(zip(*[lim.none, lim.B, lim.omega,
                         lim.phi, lim.none, lim.phi]))
    ptemp, rmstemp, F1cos1_SS_status = fit(tFit, dOmegadtFit, F1cos1_SS,
                                           p0, pBounds, jac, "F1cos1_SS")

    if(rmstemp <= rmsF1cos1_SS):
      rmsF1cos1_SS = rmstemp
      pF1cos1_SS = ptemp

  #================

  F1cos2_SS = lambda p,t: p[0]*(FitTc-t)**(-11/8) \
                          + p[1]*cos(p[2]*t + p[3] + p[4]*t*t) \
                          - p[5]*sin(2*alpha_intrp(t) + p[6])
  jac = lambda p,t: [ (FitTc-t)**(-11/8),
                      cos(p[2]*t + p[3] + p[4]*t*t),
                      -p[1]*sin(p[2]*t + p[3] + p[4]*t*t)*t,
                      -p[1]*sin(p[2]*t + p[3] + p[4]*t*t),
                      -p[1]*sin(p[2]*t + p[3] + p[4]*t*t)*t*t,
                      -sin(2*alpha_intrp(t) + p[6]),
                      -p[5]*cos(2*alpha_intrp(t) + p[6]) ]
  p0 = [pF1cos1_SS[0], pF1cos1_SS[1], pF1cos1_SS[2], pF1cos1_SS[3],
        0, pF1cos1_SS[4],pF1cos1_SS[5]]
  pBounds = list(zip(*[lim.none, lim.B, lim.omega, lim.phi,
                       lim.none, lim.none, lim.phi]))
  pF1cos2_SS, rmsF1cos2_SS, F1cos2_SS_status = fit(tFit, dOmegadtFit,
                                                   F1cos2_SS, p0, pBounds,
                                                   jac, "F1cos2_SS")

  #================

  # POSSIBLE IMPROVEMENT:
  # rewrite p[6] sin(2*alpha+p[7])   as   p6*sin() + p7*cos()
  F2cos2_SS = lambda p,t: p[0]*(FitTc-t)**(-11/8) + p[1]*(FitTc-t)**(-13/8) \
                          + p[2]*cos(p[3]*t + p[4] + p[5]*t*t) \
                          - p[6]*sin(2*alpha_intrp(t) + p[7])

  jac = lambda p,t: [ (FitTc-t)**(-11/8),
                      (FitTc-t)**(-13/8),
                      cos(p[3]*t + p[4] + p[5]*t*t),
                      -p[2]*sin(p[3]*t + p[4] + p[5]*t*t)*t,
                      -p[2]*sin(p[3]*t + p[4] + p[5]*t*t),
                      -p[2]*sin(p[3]*t + p[4] + p[5]*t*t)*t*t,
                      -sin(2*alpha_intrp(t) + p[7]),
                      -p[6]*cos(2*alpha_intrp(t) + p[7]) ]
  p0 = [pF1cos2_SS[0], pF1cos2_SS[0]/100, pF1cos2_SS[1], pF1cos2_SS[2],
        pF1cos2_SS[3], pF1cos2_SS[4], pF1cos2_SS[5], pF1cos2_SS[6]]
  pBounds = list(zip(*[lim.none, lim.none, lim.B, lim.omega,
                       lim.phi, lim.none, lim.none, lim.phi]))
  pF2cos2_SS, rmsF2cos2_SS, F2cos2_SS_status = fit(tFit, dOmegadtFit,
                                                   F2cos2_SS, p0,
                                                   pBounds, jac, "F2cos2_SS")

  #==== output all three residuals
  summary.write("tmin=%f  (determined by " % tmin)
  if(opt_tmin == None):
    summary.write(" fit)\n")
  else:
    summary.write(" option)\n")

  summary.write("""
FIT SUCCESS
    FOmega %d
    F1  %d
    F1cos1_SS  %d
    F1cos2_SS  %d
    F2cos2_SS  %d
""" % (Omega_status, F1_SS_status, F1cos1_SS_status,
       F1cos2_SS_status, F2cos2_SS_status))

  summary.write("""
RESIDUALS
    F1cos1 rms=%g   \tF1cos2 rms=%g
    F2cos2 rms=%g

DIAGNOSTICS
                Tc         B     sin(phi)    rms/B   omega/Omega0
""" % (rmsF1cos1_SS, rmsF1cos2_SS, rmsF2cos2_SS))

  # add first two fields for remaining fits
  tmp = "%-8s %10.1f   %6.2e    %6.4f     %5.3f      %5.3f\n"

  summary.write(tmp % ("F1cos1_SS", FitTc, pF1cos1_SS[1], sin(pF1cos1_SS[3]),
                       rmsF1cos1_SS / pF1cos1_SS[1],
                       pF1cos1_SS[2] / IDparam_omega0))
  summary.write(tmp % ("F1cos2_SS", FitTc, pF1cos2_SS[1], sin(pF1cos2_SS[3]),
                       rmsF1cos2_SS / pF1cos2_SS[1],
                       pF1cos2_SS[2] / IDparam_omega0))
  summary.write(tmp % ("F2cos2_SS", FitTc, pF2cos2_SS[2], sin(pF2cos2_SS[4]),
                     rmsF2cos2_SS/pF2cos2_SS[2],
                     pF2cos2_SS[3]/IDparam_omega0))

  #==== compute updates and generate EccRemoval_FIT.dat files
  summary.write("""
ECCENTRICITY AND UPDATES
             delta_Omega0   delta_adot0     delta_D0     ecc     """\
"""mean_anomaly(0)  mean_anomaly(tref)
""")

  ComputeUpdate(IDparam_omega0, IDparam_adot0, IDparam_D0,
              FitTc, pF1cos1_SS[1], pF1cos1_SS[2],
              pF1cos1_SS[3], pF1cos1_SS[2]*tref+pF1cos1_SS[3],
                # Above 2 arguments are arg of cos(..) at t=0 and tref
              "F1cos1_SS", tmin, tmax, rmsF1cos1_SS,
              opt_improved_Omega0_update, check_periastron_advance,
              params_output_dir, Source, summary)

  ComputeUpdate(IDparam_omega0, IDparam_adot0, IDparam_D0,
                FitTc, pF1cos2_SS[1], pF1cos2_SS[2],
                pF1cos2_SS[3],
                pF1cos2_SS[3]+pF1cos2_SS[2]*tref+pF1cos2_SS[4]*tref*tref,
                "F1cos2_SS", tmin, tmax, rmsF1cos2_SS,
                opt_improved_Omega0_update, check_periastron_advance,
                params_output_dir, Source, summary)

  ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev = ComputeUpdate(
    IDparam_omega0, IDparam_adot0, IDparam_D0, FitTc, pF2cos2_SS[2],
    pF2cos2_SS[3], pF2cos2_SS[4],
    pF2cos2_SS[4]+pF2cos2_SS[3]*tref+pF2cos2_SS[5]*tref*tref, "F2cos2_SS",
    tmin, tmax, rmsF2cos2_SS, opt_improved_Omega0_update,
    check_periastron_advance, params_output_dir, Source, summary)

  if plot_output_dir:
    make_full_plot(Source,t,dOmegadt,tmin,tmax,idxFit,
                   [pF1cos1_SS,pF1cos2_SS,pF2cos2_SS],
                   [F1cos1_SS,F1cos2_SS,F2cos2_SS],
                   ['F1cos1_SS','F1cos2_SS','F2cos2_SS'],
                   [':',':',"--"],
                   os.path.join(plot_output_dir, "FigureEccRemoval_SS.pdf"))

  return ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev

def performNonspinVarPro(t,dOmegadt,idxFit,tFit,dOmegadtFit,
                         tmin,tmax,tref,IDparam_omega0,IDparam_D0,
                         IDparam_adot0,q,omega_guess,opt_tmin,
                         opt_improved_Omega0_update,check_periastron_advance,
                         params_output_dir,plot_output_dir,Source,summary):
  '''
  Fit the eccentricity estimator given in arXiv:1012.1549 to the given
  trajectory, not including spin terms, and compute initial data from the
  resulting best fit parameters. Uses the variable projection fitting
  algorithm and a reparameterized form of the eccentricity estimator. Returns
  eccentricity, initial data corrections based on the best fit parameters, and
  estimated standard deviation of eccentricity.
  '''

  # Set bounds for some of the variables
  lim = FitBounds(tmax, omega_guess)

  # 0PN approximation
  Tmerger = 5. / (64.*(1. - ((q-1.) / (q+1.))**2.)*IDparam_omega0**(8./3.))

  pNonlin_0 = np.array([Tmerger, omega_guess, 0])
  # bounds given as tuple of lower bounds and upper bounds on nonlinear params
  pNonlin_bounds = ([tFit[-1]+2.0*np.spacing(abs(tFit[-1])),
                     0.6*IDparam_omega0, -np.inf],
                    [np.inf, 1.4*IDparam_omega0, np.inf])

  # Model to fit to
  nonspin_F = lambda p,t: p[3]*(p[0]-t)**(-11/8) + p[4]*(p[0]-t)**(-13/8) \
                          + p[5]*cos(p[1]*t+p[2]*t*t) \
                          - p[6]*sin(p[1]*t + p[2]*t*t)

  def nonspinFPhi(alpha, t):
    Phi = np.empty([t.shape[0], 4])
    Phi[:,0] = (alpha[0] - t)**(-11./8)
    Phi[:,1] = (alpha[0] - t)**(-13./8)
    Phi[:,2] = cos(alpha[1]*t + alpha[2]*t*t)
    Phi[:,3] = -sin(alpha[1]*t + alpha[2]*t*t)

    dPhi = np.empty([t.shape[0], 6])
    dPhi[:,0] = (-11./8)*(alpha[0] - t)**(-19./8)
    dPhi[:,1] = (-13./8)*(alpha[0] - t)**(-21./8)
    dPhi[:,2] = np.multiply(t, Phi[:,3])
    dPhi[:,3] = np.multiply(t*t, Phi[:,3])
    dPhi[:,4] = np.multiply(t, -Phi[:,2])
    dPhi[:,5] = np.multiply(t*t, -Phi[:,2])

    # the nth column in dPhi is dPhi_i/dalpha_j,
    # where i=Ind[0][n] and j=Ind[1][n]
    Ind = np.array([[0,1,2,2,3,3],[0,0,1,2,1,2]])

    return Phi, dPhi, Ind

  # Same as above function, but no -13/8 radiation-reaction term.
  def nonspinFPhiF1(alpha, t):
    Phi = np.empty([t.shape[0], 3])
    Phi[:,0] = (alpha[0] - t)**(-11./8)
    Phi[:,1] = cos(alpha[1]*t + alpha[2]*t*t)
    Phi[:,2] = -sin(alpha[1]*t + alpha[2]*t*t)

    dPhi = np.empty([t.shape[0], 5])
    dPhi[:,0] = (-11./8)*(alpha[0] - t)**(-19./8)
    dPhi[:,1] = np.multiply(t, Phi[:,2])
    dPhi[:,2] = np.multiply(t*t, Phi[:,2])
    dPhi[:,3] = np.multiply(t, -Phi[:,1])
    dPhi[:,4] = np.multiply(t*t, -Phi[:,1])

    # the nth column in dPhi is dPhi_i/dalpha_j,
    # where i=Ind[0][n] and j=Ind[1][n]
    Ind = np.array([[0,1,1,2,2],[0,1,2,1,2]])

    return Phi, dPhi, Ind

  # Perform fit using fancy variable projection routine
  try:
    pNonlin_nonspin, pLin_nonspin, res_nonspin, rms_nonspin, \
      dOmega_dt_est_nonspin, corr_mtx_nonspin, param_std_dev_nonspin = \
        VarPro.varpro(tFit, dOmegadtFit, np.ones(len(tFit)), pNonlin_0, 4,
                      lambda alpha = None: nonspinFPhi(alpha,tFit),
                      pNonlin_bounds)
  except RankError:
    # Varpro failed with RankError.  This is probably because the
    # radiation-reaction terms are degenerate. So try again with F1cos2_SS.
    pNonlin_nonspin, pLin_nonspin_tmp, res_nonspin, rms_nonspin, \
      dOmega_dt_est_nonspin, corr_mtx_nonspin, param_std_dev_nonspin = \
        VarPro.varpro(tFit, dOmegadtFit, np.ones(len(tFit)), pNonlin_0, 3,
                      lambda alpha = None: nonspinFPhiF1(alpha,tFit),
                      pNonlin_bounds)
    # pLin_nonspin has an extra parameter compared to pLin_nonspin_tmp.
    # Fill the equivalent pLin_nonspin.
    pLin_nonspin = [pLin_nonspin_tmp[0],0.0,pLin_nonspin_tmp[1],
                    pLin_nonspin_tmp[2]]

  # Calculate quantities for correction formulae
  B_nonspin = np.sqrt(pLin_nonspin[2]**2 + pLin_nonspin[3]**2)
  B_std_dev = sqrt(pLin_nonspin[2]**2 * param_std_dev_nonspin[2]**2 \
                 + pLin_nonspin[3]**2 * param_std_dev_nonspin[3]) / B_nonspin
  phi_nonspin = np.arctan2(pLin_nonspin[3], pLin_nonspin[2])
  # phi should be constrained to (0, 2pi),
  # but np.arctan2 is constrained to (-pi, pi)
  if phi_nonspin < 0:
    phi_nonspin += 2*np.pi
  # obtain actual RMS from residual value given by variable projection
  rms_nonspin = .5*rms_nonspin*rms_nonspin

  #==== output residual
  summary.write("tmin=%f  (determined by "%tmin)
  if(opt_tmin==None):
    summary.write(" fit)\n")
  else:
    summary.write(" option)\n")

  summary.write("""
RESIDUALS
    F2cos2 rms=%g

DIAGNOSTICS
                Tc         B     sin(phi)    rms/B   omega/Omega0
""" %(rms_nonspin))

  # add first two fields for remaining fits
  tmp="%-8s %10.1f   %6.2e    %6.4f     %5.3f      %5.3f\n"
  summary.write(tmp%("F2cos2", pNonlin_nonspin[0], B_nonspin,
                     sin(phi_nonspin), rms_nonspin/B_nonspin,
                     pNonlin_nonspin[1]/IDparam_omega0))

  #==== compute updates and generate EccRemoval_FIT.dat files
  summary.write("""
ECCENTRICITY AND UPDATES
         delta_Omega0   delta_adot0     delta_D0     ecc      """\
"""mean_anomaly(0)  mean_anomaly(tref)
""")

  ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev = ComputeUpdate(
    IDparam_omega0, IDparam_adot0, IDparam_D0, pNonlin_nonspin[0], B_nonspin,
    pNonlin_nonspin[1], phi_nonspin,
    phi_nonspin+pNonlin_nonspin[1]*tref+pNonlin_nonspin[2]*tref*tref,
    "F2cos2", tmin, tmax, rms_nonspin, opt_improved_Omega0_update,
    check_periastron_advance, params_output_dir, Source, summary,
    B_std_dev = B_std_dev, omega_std_dev = param_std_dev_nonspin[5])

  if plot_output_dir:
    make_full_plot(Source,t,dOmegadt,tmin,tmax,idxFit,
                   [np.concatenate((pNonlin_nonspin, pLin_nonspin))],
                   [nonspin_F],
                   ['nonspin fit'],
                   ["--"],
                   os.path.join(plot_output_dir, "FigureEccRemoval.pdf"))

  return ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev

def performSpinVarPro(t,dOmegadt,idxFit,tFit,dOmegadtFit,alpha_intrp,tmin,tmax,
                      tref,IDparam_omega0,IDparam_D0,IDparam_adot0,q,
                      omega_guess,opt_tmin,opt_improved_Omega0_update,
                      check_periastron_advance,params_output_dir,
                      plot_output_dir,Source,summary):
  '''
  Fit the eccentricity estimator given in arXiv:1012.1549 to the given
  trajectory, including spin terms, and compute initial data from the
  resulting best fit parameters. Uses the variable projection fitting
  algorithm and a reparameterized form of the eccentricity estimator. Returns
  eccentricity, initial data corrections based on the best fit parameters, and
  estimated standard deviation of eccentricity.
  '''

  # 0PN approximation
  Tmerger = 5. / (64.*(1. - ((q-1.) / (q+1.))**2.)*IDparam_omega0**(8./3.))

  pNonlin_0 = np.array([Tmerger, omega_guess, 0])
  # bounds given as tuple of lower bounds and upper bounds on nonlinear params
  pNonlin_bounds = ([tFit[-1]+2.0*np.spacing(abs(tFit[-1])),
                     0.6*IDparam_omega0, -np.inf],
                    [np.inf, 1.4*IDparam_omega0, np.inf])

  # Model to fit to
  spin_F = lambda p,t: p[3]*(p[0]-t)**(-11/8) + p[4]*(p[0]-t)**(-13/8) \
                       + p[5]*cos(p[1]*t+p[2]*t*t) \
                       - p[6]*sin(p[1]*t + p[2]*t*t) \
                       - p[7]*sin(2*alpha_intrp(t)) \
                       - p[8]*cos(2*alpha_intrp(t))

  def spinFPhi(alpha, t):
    Phi = np.empty([t.shape[0], 6])
    Phi[:,0] = (alpha[0] - t)**(-11./8)
    Phi[:,1] = (alpha[0] - t)**(-13./8)
    Phi[:,2] = cos(alpha[1]*t + alpha[2]*t*t)
    Phi[:,3] = -sin(alpha[1]*t + alpha[2]*t*t)
    Phi[:,4] = -sin(2*alpha_intrp(t))
    Phi[:,5] = -cos(2*alpha_intrp(t))

    dPhi = np.empty([t.shape[0], 6])
    dPhi[:,0] = (-11./8)*(alpha[0] - t)**(-19./8)
    dPhi[:,1] = (-13./8)*(alpha[0] - t)**(-21./8)
    dPhi[:,2] = np.multiply(t, Phi[:,3])
    dPhi[:,3] = np.multiply(t*t, Phi[:,3])
    dPhi[:,4] = np.multiply(t, -Phi[:,2])
    dPhi[:,5] = np.multiply(t*t, -Phi[:,2])

    # the nth column in dPhi is dPhi_i/dalpha_j,
    # where i=Ind[0][n] and j=Ind[1][n]
    Ind = np.array([[0,1,2,2,3,3],[0,0,1,2,1,2]])

    return Phi, dPhi, Ind

  # Same as above function, but no -13/8 radiation-reaction term.
  def spinFPhiF1(alpha, t):
    Phi = np.empty([t.shape[0], 5])
    Phi[:,0] = (alpha[0] - t)**(-11./8)
    Phi[:,1] = cos(alpha[1]*t + alpha[2]*t*t)
    Phi[:,2] = -sin(alpha[1]*t + alpha[2]*t*t)
    Phi[:,3] = -sin(2*alpha_intrp(t))
    Phi[:,4] = -cos(2*alpha_intrp(t))

    dPhi = np.empty([t.shape[0], 5])
    dPhi[:,0] = (-11./8)*(alpha[0] - t)**(-19./8)
    dPhi[:,1] = np.multiply(t, Phi[:,2])
    dPhi[:,2] = np.multiply(t*t, Phi[:,2])
    dPhi[:,3] = np.multiply(t, -Phi[:,1])
    dPhi[:,4] = np.multiply(t*t, -Phi[:,1])

    # the nth column in dPhi is dPhi_i/dalpha_j,
    # where i=Ind[0][n] and j=Ind[1][n]
    Ind = np.array([[0,1,1,2,2],[0,1,2,1,2]])

    return Phi, dPhi, Ind

  # Perform fit using fancy variable projection routine
  try:
    pNonlin_spin, pLin_spin, res_spin, rms_spin, dOmega_dt_est_spin, \
      corr_mtx_spin, param_std_dev_spin = \
        VarPro.varpro(tFit, dOmegadtFit, np.ones(len(tFit)), pNonlin_0, 6,
                      lambda alpha = None: spinFPhi(alpha,tFit),
                      pNonlin_bounds)
  except RankError:
    # Varpro failed with RankError.  This is probably because the
    # radiation-reaction terms are degenerate. So try again with F1cos2_SS.
    pNonlin_spin, pLin_spin_tmp, res_spin, rms_spin, dOmega_dt_est_spin, \
      corr_mtx_spin, param_std_dev_spin = \
        VarPro.varpro(tFit, dOmegadtFit, np.ones(len(tFit)), pNonlin_0, 5,
                      lambda alpha = None: spinFPhiF1(alpha,tFit),
                      pNonlin_bounds)
    # pLin_spin has an extra parameter compared to pLin_spin_tmp.
    # Fill the equivalent pLin_spin.
    pLin_spin = [pLin_spin_tmp[0],0.0,pLin_spin_tmp[1],
                 pLin_spin_tmp[2],pLin_spin_tmp[3],pLin_spin_tmp[4]]


  # Calculate quantities for correction calculation
  B_spin = np.sqrt(pLin_spin[2]**2 + pLin_spin[3]**2)
  B_std_dev = sqrt(pLin_spin[2]**2 * param_std_dev_spin[2]**2 \
                 + pLin_spin[3]**2 * param_std_dev_spin[3]) / B_spin
  phi_spin = np.arctan2(pLin_spin[3], pLin_spin[2])
  # phi should be constrained to (0, 2pi),
  # but np.arctan2 is constrained to (-pi, pi)
  if phi_spin < 0:
    phi_spin += 2*np.pi
  # obtain actual RMS from residual value given by variable projection
  rms_spin = .5*rms_spin*rms_spin

  #==== output residual
  summary.write("tmin=%f  (determined by " % tmin)
  if(opt_tmin == None):
    summary.write(" fit)\n")
  else:
    summary.write(" option)\n")

  summary.write("""
RESIDUALS
    F2cos2 rms=%g

DIAGNOSTICS
                Tc         B     sin(phi)    rms/B   omega/Omega0
""" % (rms_spin))

  # add first two fields for remaining fits
  tmp = "%-8s %10.1f   %6.2e    %6.4f     %5.3f      %5.3f\n"

  summary.write(tmp%("F2cos2_SS", pNonlin_spin[0], B_spin, sin(phi_spin),
                      rms_spin/B_spin, pNonlin_spin[1]/IDparam_omega0))

  #==== compute updates and generate EccRemoval_FIT.dat files
  summary.write("""
ECCENTRICITY AND UPDATES
             delta_Omega0   delta_adot0     delta_D0     ecc     """\
"""mean_anomaly(0)  mean_anomaly(tref)
""")

  ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev = ComputeUpdate(
    IDparam_omega0, IDparam_adot0, IDparam_D0, pNonlin_spin[0], B_spin,
    pNonlin_spin[1], phi_spin,
    phi_spin+pNonlin_spin[1]*tref+pNonlin_spin[2]*tref*tref, "F2cos2_SS",
    tmin, tmax, rms_spin, opt_improved_Omega0_update, check_periastron_advance,
    params_output_dir, Source, summary,
    B_std_dev = B_std_dev, omega_std_dev = param_std_dev_spin[7])

  if plot_output_dir:
    make_full_plot(Source,t,dOmegadt,tmin,tmax,idxFit,
                   [np.concatenate((pNonlin_spin, pLin_spin))],
                   [spin_F],
                   ['spin fit'],
                   ["--"],
                   os.path.join(plot_output_dir, "FigureEccRemoval_SS.pdf"))

  return ecc, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev

def performAllFits(IDparam_omega0, IDparam_adot0, IDparam_D0, XA, XB, mA, mB,
                   sA, sB, t, Omega, dOmegadt, OmegaVec, tmin, tmax, tref,
                   opt_freq_filter, opt_varpro, opt_type, opt_tmin,
                   opt_improved_Omega0_update, check_periastron_advance,
                   params_output_dir=None, plot_output_dir=None, Source=None):
  '''
  Fit an eccentricity estimator to the given trajectory. Return the
  eccentricity, final initial data corrections, and output for printing.

  If variable projection is enabled, also return estimated uncertainty of
  eccentricity. Otherwise, return None.
  '''

  summary = StringIO()
  q = max(mA/mB, mB/mA) # q > 1 by convention

  idxFit=  (t>=tmin) & (t<=tmax)
  tFit=t[idxFit]
  OmegaFit=Omega[idxFit]
  dOmegadtFit=dOmegadt[idxFit]

  omega_guess, filtered_t, filtered_dOmegadt = \
    computeOmegaGuessAndFilterTraj(tFit, dOmegadtFit, IDparam_omega0)
  if (opt_freq_filter and filtered_t is not None):
    tFit = filtered_t
    dOmegadtFit = filtered_dOmegadt

  # First, do the usual fits
  if opt_varpro:
    nonspin_ecc, nonspin_delta_Omega0, nonspin_delta_adot0, \
    nonspin_delta_D0, nonspin_ecc_std_dev = \
      performNonspinVarPro(t,dOmegadt,idxFit,tFit,dOmegadtFit,tmin,tmax,tref,
                           IDparam_omega0,IDparam_D0,IDparam_adot0,q,
                           omega_guess,opt_tmin,opt_improved_Omega0_update,
                           check_periastron_advance,params_output_dir,
                           plot_output_dir,Source,summary)
  else:
    nonspin_ecc, nonspin_delta_Omega0, nonspin_delta_adot0, \
    nonspin_delta_D0, nonspin_ecc_std_dev = \
      performNonspinFits(t,dOmegadt,idxFit,tFit,dOmegadtFit,tmin,tmax,tref,
                         IDparam_omega0,IDparam_D0,IDparam_adot0,q,omega_guess,
                         opt_tmin,opt_improved_Omega0_update,
                         check_periastron_advance,params_output_dir,
                         plot_output_dir,Source,summary)

  # Now, do fits that include spin-spin interactions,
  # see Buonnano et al., 2010 (arXiv 1012.1549v2).

  # Everything below is only for BBH (since NS have negligible spins)
  spin_ecc, spin_delta_Omega0, spin_delta_adot0, spin_delta_D0, \
  spin_ecc_std_dev = \
    None, None, None, None, None
  if opt_type=="bbh" and sA is not None and sB is not None:
    alpha_intrp, S_0_perp_n, T_merge, Amp = \
      GetVarsFromSpinData(sA, sB, XA, XB, mA, mB, OmegaVec, t, tmin)

    if opt_varpro:
      spin_ecc, spin_delta_Omega0, spin_delta_adot0, spin_delta_D0, \
      spin_ecc_std_dev = \
        performSpinVarPro(t,dOmegadt,idxFit,tFit,dOmegadtFit,alpha_intrp,tmin,
                          tmax,tref,IDparam_omega0,IDparam_D0,IDparam_adot0,
                          q,omega_guess,opt_tmin,opt_improved_Omega0_update,
                          check_periastron_advance,params_output_dir,
                          plot_output_dir,Source,summary)
    else:
      spin_ecc, spin_delta_Omega0, spin_delta_adot0, spin_delta_D0, \
      spin_ecc_std_dev = \
        performSpinFits(t,dOmegadt,idxFit,tFit,OmegaFit,dOmegadtFit,T_merge,
                        Amp,alpha_intrp,S_0_perp_n,tmin,tmax,tref,
                        IDparam_omega0,IDparam_D0,IDparam_adot0,omega_guess,
                        opt_tmin,opt_improved_Omega0_update,
                        check_periastron_advance,params_output_dir,
                        plot_output_dir,Source,summary)

  summary_string = summary.getvalue()
  summary.close()

  # If filtering is turned on and has succeeded, spin contribution is removed
  # from trajectory before fitting, so return corrections from nonspin fits
  # since spin fits aren't necessary
  if spin_ecc is None \
    or (opt_freq_filter and filtered_t is not None):
    return nonspin_ecc, nonspin_delta_Omega0, nonspin_delta_adot0, \
      nonspin_delta_D0, nonspin_ecc_std_dev, summary_string
  else:
    return spin_ecc, spin_delta_Omega0, spin_delta_adot0, spin_delta_D0, \
      spin_ecc_std_dev, summary_string
