import numpy as np
import cv2
import os
from slmsuite.holography.algorithms import Hologram
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography import toolbox

# Make the desired image: a random pixel targeted in a 32x32 grid it can be changed to (posx, posy)
target_size = (32, 32)
target = np.zeros(target_size)
target[5, 17] = 1 # Target position (posx, posy)

# Initialize the hologram and plot the target
# Note: For now, we'll assume the SLM and target are the same size (since they're a Fourier pair)
slm_size = target_size


# Assume a 532 nm red laser
wav_um = 0.532
slm = SLM(slm_size[0], slm_size[1], dx_um=10, dy_um=10, wav_um=wav_um)
camera = Camera(target_size[0], target_size[1])

# Set a Gaussian amplitude profile on SLM with radius = 100 in units of x/lam
slm.set_measured_amplitude_analytic(100)

# Redo the same GS calculations
hologram = Hologram(target, slm_shape=slm.shape, amp=slm.measured_amplitude)
#zoombox = hologram.plot_farfield(source=hologram.target, cbar=True) # See the output of target farfield distribution

# The setup (a FourierSLM setup with a camera placed in the Fourier plane of an SLM) holds the camera and SLM.
setup = FourierSLM(camera, slm)
hologram.cameraslm = setup

# Run 5 iterations of GS.
hologram.optimize(method='GS', maxiter=5) # GS : Gerchberg-Saxton algorithm, maxiter can be changed to 20 (FPS will be slower)

# Look at the associated near- and far- fields
#hologram.plot_nearfield(cbar=True)
#hologram.plot_farfield(limits=zoombox, cbar=True, title='FF Amp');

hologram.plot_nearfield(padded=True,cbar=True)
hologram.plot_farfield(cbar=True)


# Use a random logo
path = os.path.join(os.getcwd(), 'C:/Users/dlgot/Source/Repos/getxy/SLM_example.png')
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = cv2.bitwise_not(img) # Invert
assert img is not None, "Could not find this test image!"

# Resize (zero pad) for GS.
shape = (480,640)
target = toolbox.pad(img, shape)

holo = Hologram(target)
zoom = holo.plot_farfield(holo.target)

holo.optimize(method="WGS-Kim", maxiter=10) # WGS-Kim : Weighted Gerchberg-Saxton algorithm, maxiter can be changed to 20 (FPS will be slower) computational power is important
holo.plot_farfield(limits=zoom,title='WGS Image Reconstruction')
holo.plot_farfield(holo.target - holo.amp_ff, limits=zoom,title='WGS Image Error')