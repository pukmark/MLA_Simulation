import numpy as np


class EnvironmentType:
    def __init__(self, Nmc: int = 1):
        self.Nmc = Nmc
        self.EnvParams = EnvParamsType()
        self.EnvParams.Initialize()



class EnvParamsType:
    def __init__(self):
        self.Temperature = 298.0 # Kelvin
        self.Pressure = 101325.0 # Pa
        self.Density = 1.225 # kg/m^3
        self.Gravity = 9.81 # m/s^2
        self.RainRate = 0.0 # mm/hr
        self.WindSpeed = 0.0
        self.WindDirection = 0.0
        self.Humidity = 0.0 # %
        self.SolarFlux = 0.0 # W/m^2
        self.TempSigma = 5.0 # Kelvin

    def Initialize(self):
        self.Temperature = self.Temperature * np.random.randn(1)*self.TempSigma

    def Update(self, Time):
        self.Temperature = self.Temperature

        

