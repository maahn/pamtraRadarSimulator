# -*- coding: utf-8 -
# (c) M. Maahn, 2017

from __future__ import division, absolute_import, print_function

import numpy as np

from . import decorators
from . import pamtraRadarSimulatorLib



__version__ = '0.1'
c = 299792458.


@decorators.NDto2DtoND(referenceIn=0,noOfInDimsToKeep=0,convertInputs=[0,1,2],convertOutputs=[0],verbosity=0)
def calcSpectralBraodening(
    edr,
    wind_uv,
    height,
    beamwidth_deg,
    integration_time,
    frequency,
    kolmogorov = 0.5,
    verbosity = 0
    ):

  """
  Estimate the spectral broadening due to turbulence and horizontal wind. 

  Parameters
  ----------

    edr : array_like or float
    wind_uv : array_like or float
    height : array_like or float
    beamwidth_deg :float
    integration_time : float
    frequency : float
    kolmogorov (default 0.5)
    verbosity (default 0)


  Returns
  -------

  specbroad : array_like or float
 
  """


  pamtraRadarSimulatorLib.report_module.verbose = verbosity

  wavelength = c / (frequency*1e9)  

  assert len(edr.shape) == 1, 'edr has to be 1D'
  assert len(wind_uv.shape) == 1, 'wind_uv has to be 1D'
  assert len(height.shape) == 1, 'height has to be 1D'

  error,specbroad = pamtraRadarSimulatorLib.radar_spectral_broadening(
    edr,
    wind_uv,
    height,
    beamwidth_deg,
    integration_time,
    wavelength,
    kolmogorov
    )

  if error>0:
    raise RuntimeError('Error in Fortran routine estimate_spectralbroadening')

  return specbroad

@decorators.NDto2DtoND(referenceIn=0,noOfInDimsToKeep=2,convertInputs=range(10),convertOutputs=[0],verbosity=0)
def simulateRadarSpectrum(
  diameter_spec,
  back_spec,
  mass,
  rho_particle,
  area,
  pia,
  hgt,
  temp,
  press,
  wind_w,
  frequency,
  spectral_broadening,
  vel_size_mod ='heymsfield10_particles',
  radar_max_v =7.885,
  radar_min_v =-7.885,
  radar_aliasing_nyquist_interv = 1,
  radar_nfft = 256,
  radar_airmotion = False,
  radar_airmotion_model = "step", #"constant","linear","step"
  radar_airmotion_vmin = -4.0,
  radar_airmotion_vmax = +4.0,
  radar_airmotion_linear_steps = 30,
  radar_airmotion_step_vmin = 0.5,
  radar_pnoise1000 = -30,
  radar_k2 =0.93, # dielectric constant |K|² (always for liquid water by convention) for the radar equation
  radar_no_ave = 150,
  seed  = 0,
  verbosity = 0
  ):

  """
    Parameters
    ----------
    diameter_spec,
    back_spec,
    temp,
    press,
    wind_w,
    frequency,
    rho_particle,
    vel_size_mod,
    mass,
    area,
    radar_max_v =7.885,
    radar_min_v =-7.885,
    radar_aliasing_nyquist_interv = 1,
    radar_nfft = 256,
    radar_airmotion = False,
    radar_airmotion_model = "step", #"constant","linear","step"
    radar_airmotion_vmin = -4.0,
    radar_airmotion_vmax = +4.0,
    radar_airmotion_linear_steps = 30,
    radar_airmotion_step_vmin = 0.5,
    radar_k2 =0.93, # dielectric constant |K|² (always for liquid water by convention) for the radar equation

  """
  pamtraRadarSimulatorLib.report_module.verbose = verbosity

  wavelength = c / (frequency*1e9)  

  radar_nfft_aliased = radar_nfft *(1+2*radar_aliasing_nyquist_interv)

  if seed == 0:
    seed = np.random.randint(2**16)

  nHeights = diameter_spec.shape[0]
  nHydro = diameter_spec.shape[1]
  particle_spec = np.zeros((nHeights,nHydro,radar_nfft_aliased))

  for hh in range(nHydro):
    error,particle_spec[:,hh,:] = pamtraRadarSimulatorLib.radar_spectrum.get_radar_spectrum(
      diameter_spec[:,hh,:],
      back_spec[:,hh,:],
      temp,
      press,
      wind_w,
      wavelength,
      rho_particle[:,hh,:],
      vel_size_mod,
      mass[:,hh,:],
      area[:,hh,:],
      radar_max_v,
      radar_min_v,
      radar_aliasing_nyquist_interv,
      radar_nfft,
      radar_nfft_aliased,
      radar_airmotion,
      radar_airmotion_model,
      radar_airmotion_vmin,
      radar_airmotion_vmax,
      radar_airmotion_linear_steps,
      radar_airmotion_step_vmin,
      radar_k2,
      )
    if error>0:
      raise RuntimeError('Error in Fortran routine radar_spectrum')

  #merge all the hydrometeors
  merged_particle_spec = np.sum(particle_spec,axis=1)

  #estimate noise from value at 1 km:
  radar_pnoise = 10**(0.1*radar_pnoise1000) * (hgt/1000.)**2


  error,radar_spectrum = pamtraRadarSimulatorLib.radar_simulator.simulate_radar(wavelength,
    merged_particle_spec,
    pia,
    spectral_broadening,
    radar_pnoise,
    radar_max_v,
    radar_min_v,
    radar_nfft,
    radar_no_ave,
    radar_aliasing_nyquist_interv,
    radar_k2,
    seed,
    )
  if error>0:
    raise RuntimeError('Error in Fortran routine radar_simulator')

  return radar_spectrum