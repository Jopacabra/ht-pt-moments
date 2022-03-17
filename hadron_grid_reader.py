# reads UrQMD grids

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

particle_data = np.genfromtxt('TestCases/Lattice.csv', delimiter=',', dtype=float)
particle_mass = np.genfromtxt('TestCases/Masses/OneTypeMass.csv', delimiter=',', dtype=float)

# position coordinates
x = particle_data[:, 2]
y = particle_data[:, 3]
# momentum components
px = particle_data[:, 4]
py = particle_data[:, 5]
# particle types
t = particle_data[:, 1]  # t = the type of particle

# Find mass of each particle
m = np.zeros((1, len(x)))
for i in range(0, len(x)):
    m[0, i] = particle_mass[int(t[i])]

# number of bins we want in each dimension:
bx = 15
by = 15

# bin sizes based on max and min values
# NOTE: a fudge factor F (F for Fudge) is added to make sure that the number of bins comes out correct
# This is done since if there were no F, the floor function used to bin the stuff would not count the
# particle with the largest x coordinate (or y coordinate) in the last bin, but make a new bin
F = 0.05
dx = (F+max(x)-min(x))/bx
dy = (F+max(y)-min(y))/by


def binx(c):
    u = np.floor((c-min(c))/dx)
    # Subtracting the min first shifts everything to 0 or greater. Dividing by bin size gives a quotient and a
    # remainder, and the floor function removes the remainder to give a bin number in x, starting at 0
    return u


def biny(c):
    u = np.floor((c-min(c))/dy)
    # Works the same  way as binx(), but with y
    return u


# now bin the positions
rx = binx(x)
ry = biny(y)

# particle IDs in each bin
particles = np.zeros((by, bx), dtype=list)
for i in range(0, by):
    for j in range(0, bx):
        particles[i, j] = []
        # this makes a list for every possible bin position.

for i in range(0, rx.size):
    particles[int(ry[i]), int(rx[i])].append(i)
    # particle ID numbers are appended to the lists created above.

assert particles.shape == (by, bx)
# that is, the particles array should have the same dimensions as number of bins in each direction

print('particles')
print(particles)
# Number of particles in each square: just count them!
N = np.zeros((by, bx))
for i in range(0, by):
    for j in range(0, bx):
        N[i, j] = len(particles[i, j])
assert N.shape == particles.shape
assert sum(sum(N)) == len(x)
# Number array should have the same dimensions as particles array, and the sum of all
# the number array should equal the number of particles we have in the first place

print('Number grid')
print(N)

# momentum information arrays (each element is for one square of the grid)
prx = np.zeros((by, bx))
pry = np.zeros((by, bx))
for i in range(0, by):
    for j in range(0, bx):
        A = particles[i, j]
        for k in A:
            prx[i, j] += px[k]
            pry[i, j] += py[k]
            # momentum is just dumped together all in one
assert prx.shape == particles.shape
assert pry.shape == prx.shape

print('x momentum grid')
print(prx)
print('y momentum grid')
print(pry)

# Mass in each bin (add up the masses!)
m = np.zeros((by, bx))
for i in range(0, by):
    for j in range(0, bx):
        A = particles[i, j]
        for k in A:
            m[i, j] += particle_mass[int(t[k])]
assert m.shape == pry.shape

print('mass grid')
print(m)


def velocity(c, d):
    # c is a component of momentum
    # d is the mass
    # using ii and jj as index variables to differentiate from prior uses.
    # Not sure if that is strictly speaking necessary but it seems to work.
    v = np.zeros((by, bx))
    for ii in range(0, by):
        for jj in range(0, bx):
            if d[ii, jj] != 0:
                v[ii, jj] = c[ii, jj]/d[ii, jj]
            else:
                v[ii, jj] = 0
    return v


# temperature of each bin
T = np.zeros((by, bx))

# cross section for each bin
sigma = np.zeros((by, bx))

vrx = velocity(prx, m)
vry = velocity(pry, m)

# Here is the new part to save the file:
np.savez('particleinfo', N=N, m=m, vrx=vrx, vry=vry, T=T, sigma=sigma)

sns.heatmap(vrx, annot=False)
plt.show()
