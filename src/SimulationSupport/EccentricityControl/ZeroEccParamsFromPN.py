# fmt: off
# isort: skip_file

"""
This code was only very slightly changed from the SpEC implementation and should
be modernized. It was kept in its original form to make transitioning from SpEC
to this package easier.

Determines low eccentricity initial parameters (D0, Omega0, adot0) from the 3.5
Post-Newtonian expressions for r vs. omega and rdot vs. r for circular orbits
given some criterion to satisfy, i.e. desired initial separation, frequency, or
number of orbits.  The number of orbits is determined by integrating 3.5PN (T4)
expressions.

This will typically be a crude estimate for SpEC initial data parameters, but
should be good enough to start the eccentricity reduction process. The estimates
will be especially bad if the initial separation is small.
"""

from __future__ import print_function
from __future__ import division
from numpy import sqrt, array, log, arange, diff, where, cross, interp
from math import pi
import sys, os
import argparse
from scipy.optimize import fmin
from scipy.integrate import odeint


def error(msg):
    e_msg = "\n############# ERROR #############\n{}\n".format(msg)
    raise Exception(e_msg)


def warning(msg):
    os.sys.stderr.write("\n############# WARNING #############\n")
    os.sys.stderr.write("{}\n".format(msg))


###########################################
def omegaAndAdot(r,q,chiA,chiB,rPrime0):
    LHat = array([0.,0.,1.])
    chiAL = chiA.dot(LHat)
    chiBL = chiB.dot(LHat)
    chiAB = chiA.dot(chiB)
    rInv = 1./r
    mA = q/(1. + q)
    mB = 1./(1. + q)
    eta = mA*mB
    deltaM = mA - mB
    SL = LHat.dot(mA**2.*chiA + mB**2.*chiB)
    SigmaL = LHat.dot(mB*chiB - mA*chiA)

    # See equation 4.2 of http://arxiv.org/abs/1212.5520v1 for the PN expression for omega(r) for circular orbits.
    # Note that the 2.5PN term disagrees with equation 5.10, 5.11a, 5.11b of http://journals.aps.org/prd/pdf/10.1103/PhysRevD.74.104033
    # We use the version in equation 4.2 of the first reference since it is more recent.
    # We also include the spin-spin term (2PN order) from equation 4.5 of Equation 4.5 of http://arxiv.org/abs/gr-qc/9506022
    # Equation 228 of http://arxiv.org/abs/1310.1528 also gives omega(r), without spin terms.
    A1 = (-3. + eta)*rInv
    A1p5 = (-chiAL*(2.*mA*mA + 3.*eta) - chiBL*(2.*mB*mB + 3.*eta))*rInv*sqrt(rInv)
    A2 = (6. + 41*eta/4. + eta*eta - 1.5*eta*chiAB + 4.5*eta*chiAL*chiBL)*rInv*rInv
    A2p5 = ((22.5 - 13.5*eta)*SL + (13.5 - 6.5*eta)*deltaM*SigmaL )*(rInv**2.5)
    A3 = (-10. + (-75707./840 + 41*pi*pi/64. + 22.*log(r/rPrime0))*eta + 9.5*eta*eta + eta**3.)*rInv*rInv*rInv
    A3p5 = (1./8.)*(( -495. - 561.*eta - 51*eta*eta)*SL + (-297. - 341*eta - 21*eta*eta)*deltaM*SigmaL)*(rInv**3.5)
    omega = sqrt(rInv**3.*(1. + A1 + A1p5 + A2 + A2p5 + A3 + A3p5))


    # adot0 = (dr/dt)/r, given in equation 4.12 of
    # http://arxiv.org/abs/gr-qc/9506022
    B1 = -(1./336.)*(1751 + 588*eta)*rInv
    B1p5 = -( (7./12.)*(chiAL*(19.*mA*mA + 15.*eta) + chiBL*(19.*mB*mB + 15.*eta)) -4.*pi)*rInv*sqrt(rInv)
    B2 = (-5./48.)*eta*( 59.*chiAB - 173.*chiAL*chiBL )*rInv*rInv

    dr_dt = (-64./5.)*eta*rInv*rInv*rInv*(1 + B1 + B1p5 + B2)
    adot = dr_dt/r

    return omega,adot

###########################################
def nOrbitsAndTotalTime(q,chiA0,chiB0,omega0, cutoffFrequency=0.1):
    """
q = double                  # mass ratio (q >= 1)
chiA0 = double              # initial dimensionless spin A
chiB0 = double              # initial dimensionless spin B
omega0 = double             # initial orbital frequency
cutoffFrequency = double    # stop the PN evolution at this frequency
    """
    m1 = q/(1. + q)
    m2 = 1./(1. + q)
    delta = m1 - m2
    nu = (1. - delta**2.)/4.
    m2overm1 = m2/m1
    m1overm2 = m1/m2
    S10 = chiA0*m1*m1
    S20 = chiB0*m2*m2
    def pow(a,b):
        return a**b

    def dydt(y,dummyTime):
        v = y[1]
        S1 = y[2:5]
        S2 = y[5:8]
        LN = y[8:11]
        LNHat = LN/sqrt(LN.dot(LN))
        chi1 = S1/(m1*m1)
        chi2 = S2/(m2*m2)
        chis = 0.5*(chi1 + chi2)
        chia = 0.5*(chi1 - chi2)
        chischis = chis.dot(chis)
        chischia = chis.dot(chia)
        chiachia = chia.dot(chia)
        chisLNHat = chis.dot(LNHat)
        chiaLNHat = chia.dot(LNHat)
        S1Mag = sqrt(S1.dot(S1))
        S2Mag = sqrt(S2.dot(S2))
        chi1LNHat = chisLNHat + chiaLNHat
        chi2LNHat = chisLNHat - chiaLNHat

        # Taken from Triton, should be updated.  Could interface with GWFrames.
        dvdt2 = (-2.2113095238095237 - 2.75*nu)
        dvdt3 = (0.08333333333333333*(150.79644737231007 - 113.*chisLNHat - 113.*chiaLNHat*delta + 76.*chisLNHat*nu))
        dvdt4 = (0.00005511463844797178*(34103. - 44037.*chiachia + 135891.*pow(chiaLNHat,2) - 44037.*chischis + 135891.*pow(chisLNHat,2) - 
            88074.*chischia*delta + 271782.*chiaLNHat*chisLNHat*delta + 122949.*nu + 181440.*chiachia*nu - 544320.*pow(chiaLNHat,2)*nu - 
            5292.*chischis*nu + 756.*pow(chisLNHat,2)*nu + 59472.*pow(nu,2)))
        dvdt5 = (0.000496031746031746*(-39197.65153883985 - 63142.*chisLNHat - 4536.*chiachia*chisLNHat - 1512.*chischis*chisLNHat - 
            63142.*chiaLNHat*delta - 1512.*chiachia*chiaLNHat*delta - 4536.*chiaLNHat*chischis*delta - 149627.77490517468*nu + 
            185312.*chisLNHat*nu + 13608.*chiachia*chisLNHat*nu + 4536.*chischis*chisLNHat*nu + 97860.*chiaLNHat*delta*nu + 
            1512.*chiachia*chiaLNHat*delta*nu + 4536.*chiaLNHat*chischis*delta*nu - 53088.*chisLNHat*pow(nu,2)))
        dvdt6 = (2.385915084327783e-9*(6.745934508094527e10 - 1.3565475e8*chiachia + 5.3794125e8*pow(chiaLNHat,2) - 1.3565475e8*chischis - 
            4.937716571870764e10*chisLNHat + 2.684976525e10*pow(chisLNHat,2) - 4.937716571870764e10*chiaLNHat*delta - 
            2.713095e8*chischia*delta + 5.36995305e10*chiaLNHat*chisLNHat*delta + 2.6311824e10*pow(chiaLNHat,2)*pow(delta,2) - 
            6.931556164404614e10*nu + 2.28534075e9*chiachia*nu - 6.84146925e9*pow(chiaLNHat,2)*nu + 1.37598615e9*chischis*nu + 
            3.247920233941658e10*chisLNHat*nu - 3.548967345e10*pow(chisLNHat,2)*nu + 3.1187079e9*chischia*delta*nu - 
            4.01793777e10*chiaLNHat*chisLNHat*delta*nu + 2.53066275e8*pow(nu,2) - 6.2170416e9*chiachia*pow(nu,2) + 
            1.86511248e10*pow(chiaLNHat,2)*pow(nu,2) - 2.03742e7*chischis*pow(nu,2) + 8.8511346e9*pow(chisLNHat,2)*pow(nu,2) - 
            9.063285e8*pow(nu,3)))
        dvdt6Ln4v = (-16.304761904761904)
        dvdt7 = (-3.440012789087038 - 18.84955592153876*(chiachia + chischis - 3.*pow(chisLNHat,2) + 2.*chischia*delta - 
            3.*chiaLNHat*(chiaLNHat + 2.*chisLNHat*delta)) + 0.000036743092298647854*(-2.512188e6*pow(chisLNHat,3) - 
            7.536564e6*chiaLNHat*pow(chisLNHat,2)*delta + chiaLNHat*delta*(-2.529407e6 + 794178.*chiachia - 2.512188e6*pow(chiaLNHat,2) + 
            732942.*chischis + 1.649592e6*chischia*delta) + chisLNHat*(-2.529407e6 + 732942.*chiachia + 794178.*chischis + 
            1.649592e6*chischia*delta - 2.512188e6*pow(chiaLNHat,2)*(1. + 2.*pow(delta,2)))) + (186.31130043424588 + 
            75.39822368615503*chiachia - 226.1946710584651*pow(chiaLNHat,2) + 53.1875*pow(chisLNHat,3) + 369.5*pow(chiaLNHat,3)*delta + 
            0.00016534391534391533*chiaLNHat*(845827. - 738864.*chiachia + 29904.*chischis)*delta + 
            106.65277777777777*chiaLNHat*pow(chisLNHat,2)*delta - 0.000018371546149323927*chisLNHat*(-1.0772921e7 + 7.13097e6*chiachia - 
            2.3022846e7*pow(chiaLNHat,2) + 674730.*chischis + 1.914948e6*chischia*delta))*nu + 0.00016534391534391533*(1.1497600793607924e6 + 
            3.*(-398017. + 146076.*chiachia - 431424.*pow(chiaLNHat,2) - 1204.*chischis)*chisLNHat + 840.*pow(chisLNHat,3) + 
            7.*chiaLNHat*(-41551. + 108.*chiachia + 324.*chischis)*delta)*pow(nu,2) + 9.467592592592593*chisLNHat*pow(nu,3))
        
        omega1 = (v**5)*(0.75*(1-delta) + 0.5*nu + v*v*(0.5625*(1-delta)+1.25*nu*(1+0.5*delta)-0.041666666666666667*nu*nu +
            v*v*(0.84375 + delta*(-0.84375 + (4.875 - 0.15625*nu)*nu) + nu*(0.1875 + (-3.28125 - 0.020833333333333332*nu)*nu))))
        omega2 = (v**5)*(0.75*(1+delta) + 0.5*nu + v*v*(0.5625*(1+delta)+1.25*nu*(1-0.5*delta)-0.041666666666666667*nu*nu + 
            v*v*(0.84375 - delta*(-0.84375 + (4.875 - 0.15625*nu)*nu) + nu*(0.1875 + (-3.28125 - 0.020833333333333332*nu)*nu))))
        omegaLN = (v**5)*((2+1.5*m2overm1-1.5*(v/nu)*(S2Mag*chi2LNHat)) * S1 + (2+1.5*m1overm2-1.5*(v/nu)*(S1Mag*chi1LNHat)) * S2)
        
        s1Dot = cross(omega1*LNHat,S1)
        s2Dot = cross(omega2*LNHat,S2)
        LNDot = cross(omegaLN,LN)
        return [v**3,(6.4*nu)*(v**9)*(1. + v*v*(dvdt2 + v*(dvdt3 + v*(dvdt4 + v*(dvdt5 + v*(dvdt6 + dvdt6Ln4v*log(4.*v) + v*(dvdt7))))))),
                s1Dot[0],s1Dot[1],s1Dot[2],s2Dot[0],s2Dot[1],s2Dot[2],LNDot[0],LNDot[1],LNDot[2]]

    # odeint needs to know what output times we want.  Use twice the 0PN time to merger to make sure we integrate for enough time.
    timeToMerger0PN = 5./(256*nu*(omega0**(8./3.)))
    times = arange(0.,timeToMerger0PN*2,1.)
    phiAndVVsT = odeint(dydt,[0.,omega0**(1./3.),S10[0],S10[1],S10[2],S20[0],S20[1],S20[2],0.,0.,1.],times).T
    phiVsT = phiAndVVsT[0]
    omegaVsT = phiAndVVsT[1]**3.
    if max(omegaVsT) < cutoffFrequency: error("The PN integration did not reach the cutoff frequency in twice the 0PN time.")
    endIndex = where(omegaVsT > cutoffFrequency)[0][0]
    orbits = interp(cutoffFrequency, omegaVsT[endIndex-1:endIndex+1], phiVsT[endIndex-1:endIndex+1]/(2.*pi))
    totalTime = interp(cutoffFrequency, omegaVsT[endIndex-1:endIndex+1], times[endIndex-1:endIndex+1])
    return orbits, totalTime

###########################################
def nOrbits(q,chiA0,chiB0,omega0, cutoffFrequency=0.1):
    return nOrbitsAndTotalTime(q,chiA0,chiB0,omega0, cutoffFrequency=0.1)[0]

def totalTime(q,chiA0,chiB0,omega0, cutoffFrequency=0.1):
    return nOrbitsAndTotalTime(q,chiA0,chiB0,omega0, cutoffFrequency=0.1)[1]

###########################################
def fromOmega0(omega0,rPrime0_vals, q, chiAParsed, chiBParsed):
    d0Vals = []
    adotVals = []
    for rPrime0 in rPrime0_vals:
        def helperFunc(D0):
            return abs(omegaAndAdot(D0,q,chiAParsed,chiBParsed,rPrime0)[0]-omega0)
        d0 = fmin(helperFunc,10.)[0]
        omega,adot = omegaAndAdot(d0,q,chiAParsed,chiBParsed,rPrime0)
        d0Vals.append(d0)
        adotVals.append(adot)
        if abs(omega - omega0)/omega0 > 1.e-4:
            error('Failed to determine a D0 which recovers the desired Omega0 using fmin')
    orbits, tMerger = nOrbitsAndTotalTime(q,chiAParsed,chiBParsed,omega0)
    for i in range(len(rPrime0_vals)):
        print('##############################')
        print('Results for rPrime0 = ' + str(rPrime0_vals[i]) + ':')
        print('Omega0 = ' + str(omega0))
        print('D0 = ' + str(d0Vals[i]))
        print('adot0 = ' + str(adotVals[i]))
        print('Approximate nOrbits = ' + str(orbits))
        print('Approximate tMerger = ' + str(tMerger))
    return d0Vals[0], adotVals[0], orbits

###########################################
def main():
    rPrime0_vals = [1.,10.] # rPrime0 is a gauge choice, try a couple different values and see what the difference is.

    p = argparse.ArgumentParser(usage=__doc__)
    p1 = p.add_argument_group("required arguments")
    p1.add_argument("--q", type=float, required=True,
        help = "Mass ratio")
    p1.add_argument("--chiA", type=str, required=True, metavar="X,Y,Z",
        help="Dimensionless spin vector of black hole A")
    p1.add_argument("--chiB", type=str, required=True, metavar="X,Y,Z",
        help="Dimensionless spin vector of black hole B")
    p2 = p.add_argument_group("criterion arguments", "Choose one of the following")
    pm = p2.add_mutually_exclusive_group(required=True)
    pm.add_argument("--D0", type=float,
        help="Initial separation distance")
    pm.add_argument("--Omega0", type=float,
        help="Initial orbital angular frequency")
    pm.add_argument("--NOrbits", type=float,
        help="Number of orbits until merger. Note that this is approximate and could take a minute.")
    pm.add_argument("--tMerger", type=float,
        help="Time until merger. Note that this is approximate and could take a minute.")
    opts = p.parse_args()
    
    #Parse spins
    temp=opts.chiA.split(",")
    if(len(temp)!=3):
        error("chiA=%s does not have three components"%opts.chiA)
    chiAParsed=array([float(temp[0]), float(temp[1]), float(temp[2])])
    temp=opts.chiB.split(",")
    if(len(temp)!=3):
        error("chiB=%s does not have three components"%opts.chiB)
    chiBParsed=array([float(temp[0]), float(temp[1]), float(temp[2])])

    if opts.Omega0 != None:
        fromOmega0(opts.Omega0, rPrime0_vals, opts.q, chiAParsed, chiBParsed)

    if opts.NOrbits != None:
        # First find an omega0 that gives the right number of orbits, then use that to find D0, adot0.
        def helperFunc(args):
            omega0 = args[0]
            return abs(nOrbits(opts.q,chiAParsed,chiBParsed,omega0) - opts.NOrbits)
        omega = fmin(helperFunc,0.01)[0]
        fromOmega0(omega, rPrime0_vals, opts.q, chiAParsed, chiBParsed)

    if opts.tMerger != None:
        # First find an omega0 that gives the right time, then use that to find D0, adot0.
        def helperFunc(args):
            omega0 = args[0]
            return abs(totalTime(opts.q,chiAParsed,chiBParsed,omega0) - opts.tMerger)
        omega = fmin(helperFunc,0.01)[0]
        fromOmega0(omega, rPrime0_vals, opts.q, chiAParsed, chiBParsed)

    if opts.D0 != None:
        # Make sure to get the number of orbits for all cases before printing results, since odeint prints many warnings
        omegaAdotVals = [omegaAndAdot(opts.D0,opts.q,chiAParsed,chiBParsed,rPrime0) for rPrime0 in rPrime0_vals]
        orbitsAndTimes = [nOrbitsAndTotalTime(opts.q,chiAParsed,chiBParsed,omegaAdot[0]) for omegaAdot in omegaAdotVals]
        for i in range(len(rPrime0_vals)):
            print('###############################')
            print('Results for rPrime0 = ' + str(rPrime0_vals[i]) + ':')
            print('Omega0 = ' + str(omegaAdotVals[i][0]))
            print('D0 = ' + str(opts.D0))
            print('adot0 = ' + str(omegaAdotVals[i][1]))
            print('Approximate nOrbits = ' + str(orbitsAndTimes[i][0]))
            print('Approximate tMerger = ' + str(orbitsAndTimes[i][1]))

###########################################
if __name__ == "__main__":
    main()
