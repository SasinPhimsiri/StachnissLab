import numpy as np 

def triangulate_points(xs, xss, K, R01, X1):
    """Triangulate a batch of rays

    Input:
        xs  - [N x 3] points in first image
        xss - [N x 3] points in second image
        K   - [3 x 3] calibration matrix
        R12 - [3 x 3] rotation of 2nd cam w.r.t. 1st cam
        X1  - [3 x 1] projection center of 2nd cam
      
    Output:
        X - [N x 3] 3D points in global frame
    """

    # compute direction in corresponding camera frame
    k_xs = np.dot(np.linalg.inv(K), xs.T)
    k_xss = np.dot(np.linalg.inv(K), xss.T)

    # rotate rays of 2nd cam s.t. they are represented in the coord system of 1st cam,
    k_xss = np.dot(R01, k_xss)

    # intersect points,
    X = np.zeros(k_xs.shape)
    X0 = np.array([0,0,0])
    for i in range(k_xss.shape[1]):
        X[:, i] = compute_intersection(X0, k_xs[:, i], X1, k_xss[:, i])

    return X.T


def compute_intersection(X0, r, X1, s):
    """ Compute the intersection of rays of two cameras

    Input:
        X0 - [3 x 1] position of 1st camera
        r  - [3 x 1] direction vector of 1st cam
        X1 - [3 x 1] position of 2nd camera
        s  - [3 x 1] direction vector of 2nd cam
    Output:
        X - [3 x 1] point of intersection
    """
    r = r / np.linalg.norm(r)
    s = s / np.linalg.norm(s)

    # derived from geometrical constraints\n",
    A = np.array([[np.dot(r.T,r), np.dot(-s.T,r)],
                  [np.dot(r.T,s), np.dot(-s.T,s)]])

    b = np.array([[np.dot((X1-X0).T, r)],
                  [np.dot((X1-X0).T, s)]])
    
    x = np.linalg.solve(A, b) # length of rays

    # X0 - point in r where the lines should intersect
    # X1 - point in s where the lines should intersect
    F = X0 + x[0] * r
    H = X1 + x[1] * s
    
    # we need a mean point if two lines do not intersect in 3D
    X = (F + H) / 2

    return X