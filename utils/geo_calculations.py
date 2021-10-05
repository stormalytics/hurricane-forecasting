import numpy as np

from numpy import arctan as atan
from numpy import arctan2 as atan2
from numpy import cos as cos
from numpy import radians as radians
from numpy import sin as sin
from numpy import sqrt as sqrt
from numpy import tan as tan
from numpy import arcsin as arcsin
import time
from pprint import pprint


# Haversine Formula
# https://en.wikipedia.org/wiki/Haversine_formula

def haversine(coord1, coord2):

    coord1 = np.asarray(coord1).reshape((-1, 2))
    coord2 = np.asarray(coord2).reshape((-1, 2))

    R = 6378137.0  # radius at equator in meters (WGS-84)

    lat1 = coord1[:, 0]
    lon1 = coord1[:, 1]

    lat2 = coord2[:, 0]
    lon2 = coord2[:, 1]

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*arcsin(sqrt(a))

    return R * c



# Inverse Vincenty's Formulae
# https://en.wikipedia.org/wiki/Vincenty%27s_formulae

def vincenty_inverse(coord1, coord2, maxIter=1000, tol=10**-16):

    coord1 = np.asarray(coord1).reshape((-1, 2))
    coord2 = np.asarray(coord2).reshape((-1, 2))

    # constants

    a = 6378137.0        # radius at equator in meters (WGS-84)
    f = 1/298.257223563  # flattening of the ellipsoid (WGS-84)
    b = (1-f)*a

    # lat -> phi
    # lon -> L

    phi_1 = coord1[:, 0]
    L_1 = coord1[:, 1]

    phi_2 = coord2[:, 0]
    L_2 = coord2[:, 1]

    u_1 = atan((1-f)*tan(radians(phi_1)))
    u_2 = atan((1-f)*tan(radians(phi_2)))

    L = radians(L_2-L_1)

    # set initial value of lambda to L
    Lambda = L

    sin_u1 = sin(u_1)
    cos_u1 = cos(u_1)
    sin_u2 = sin(u_2)
    cos_u2 = cos(u_2)

    # iterative calculations
    iters = 0
    for i in range(0, maxIter):
        iters += 1

        cos_lambda = cos(Lambda)
        sin_lambda = sin(Lambda)
        sin_sigma = sqrt((cos_u2*sin(Lambda))**2 +
                         (cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)**2)
        cos_sigma = sin_u1*sin_u2+cos_u1*cos_u2*cos_lambda
        sigma = atan2(sin_sigma, cos_sigma)
        sin_alpha = (cos_u1*cos_u2*sin_lambda)/sin_sigma
        cos_sq_alpha = 1-sin_alpha**2
        cos2_sigma_m = cos_sigma-((2*sin_u1*sin_u2)/cos_sq_alpha)
        C = (f/16)*cos_sq_alpha*(4+f*(4-3*cos_sq_alpha))
        Lambda_prev = Lambda
        Lambda = L+(1-C)*f*sin_alpha*(sigma+C*sin_sigma *
                                      (cos2_sigma_m+C*cos_sigma*(-1+2*cos2_sigma_m**2)))

        # successful convergence
        diff = abs(Lambda_prev-Lambda)
        if np.amax(diff) <= tol:
            break

    u_sq = cos_sq_alpha*((a**2-b**2)/b**2)
    A = 1+(u_sq/16384)*(4096+u_sq*(-768+u_sq*(320-175*u_sq)))
    B = (u_sq/1024)*(256+u_sq*(-128+u_sq*(74-47*u_sq)))
    delta_sig = B*sin_sigma*(cos2_sigma_m+0.25*B*(cos_sigma*(-1+2*cos2_sigma_m**2)-(
        1/6)*B*cos2_sigma_m*(-3+4*sin_sigma**2)*(-3+4*cos2_sigma_m**2)))

    # output distance (in meters)
    meters = b*A*(sigma-delta_sig)

    # a1 is the initial bearing, or forward azimuth
    # a2 is the final bearing (in direction p1->p2)
    a1 = atan2(cos_u2*sin_lambda, cos_u1*sin_u2 - sin_u1*cos_u2*cos_lambda)
    a2 = atan2(cos_u1*sin_lambda, -1*cos_u1*sin_u2 + sin_u1*cos_u2*cos_lambda)

    km = meters/1000                    # distance in kilometers
    miles = meters*0.000621371          # distance in miles
    n_miles = miles*(6080.20/5280)      # distance in nautical miles
    return meters, a1, a2


if __name__ == "__main__":
    points_1 = []
    points_2 = []
    for x in range(10000):
        points_1.append([17.5, -56.4])
        points_2.append([18.28926911, -56.86163531])
        

    t0 = time.time()
    answer = vincenty_inverse(np.asarray(points_1), np.asarray(points_2))
    t1 = time.time()
    total_n = t1-t0
    pprint(answer[0]/1000)
    pprint(total_n)