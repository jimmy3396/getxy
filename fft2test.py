# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 19:45:18 2024

@author: 82109
"""

from diffractio import um, mm, degrees, np, plt
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY

z = 250*mm
wavelength = 0.5 * um
xin = np.linspace(-1*mm, 1*mm, 1024)
yin = np.linspace(-1*mm, 1*mm, 1024)

xout = np.linspace(-10*mm, 10*mm, 1024)
yout = np.linspace(-10*mm, 10*mm, 1024)

# using gaussian beam for light source
u0 = Scalar_source_XY(x=xin, y=yin, wavelength=wavelength)
u0.gauss_beam(A=1, r0=(0*um,0*um), z0=0, w0=(200*um, 200*um), phi=0*degrees, theta=0*degrees)


# lens, used to shift focus in z direction
t0 = Scalar_mask_XY(x=xin, y=yin, wavelength=wavelength)
t0.lens(r0=(0*um, 0*um), radius=(2000*um, 2000*um), focal=z, angle=0*degrees)


# linear phase mask, used to shift focus in xy plane
t1 = Scalar_mask_XY(x=xin, y=yin, wavelength=wavelength)
t1.blazed_grating(period=25*um, phase_max=2*np.pi, x0=0, angle=100*degrees)

t2=t0*t1

u1 = t2*u0
#u2 = u1.fft(remove0 = False, new_field = True)
u2 = u1.CZT(z, xout, yout)
u2.draw(kind = 'intensity', logarithm=0.1) # plot intensity as logplot