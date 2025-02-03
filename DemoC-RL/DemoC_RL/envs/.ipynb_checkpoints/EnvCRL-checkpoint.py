# DEMO
#     Uso de Reinforcement Learning (RL) para el control de movimiento de un manipulador
#     robotico en CoppeliaSIm
#     Tarea: Alcanzar el objetivo (esfera) con el efector final del robot.
#     Autor: Roger Sarango
#     Fecha: 04/11/2024

import os
import sys
import cv2 as cv
import gymnasium
import numpy as np
import pandas as pd
import sim
import time
import random
import math
import random
from matplotlib import pyplot as plt 
from scipy.interpolate import PchipInterpolator, CubicSpline
import torch as th   
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium import spaces, error, utils
from gymnasium.utils import seeding
from typing import Optional
#----------------------------------------------------------------------------------------------------------------------------
class TareaRL(gymnasium.Env):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps":4}
    def __init__(self):      
        # CONEXION AL ENTORNO
        sim.simxFinish(-1) # Cerrar todas las conexiones abiertas
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 3000, 5)
        if self.clientID!=-1:
            print('Conectado al servidor API remoto')
            sim.simxSynchronous(self.clientID,True)
        else:
            print('Conexion no exitosa')
            sys.exit('No se puede conectar')
        # VARIABLES

        self.tip = 3
        self.default_pos = np.array([0.59, -0.27, 0.20])   # T1 entrenamiento
        self.reward = 0.0   # Recompensa
        self.n_actions = 3  # Salida de espacio de accion
        self.step_counter = 0   # Contador de episodio 
        # PUNTOS INICIALES/FINALES DE TRAYECTORIAS

        # MANEJADORES DE ESCENA
        _, self.target = sim.simxGetObjectHandle(self.clientID, 'Target', sim.simx_opmode_blocking)   # EfectorFinal
        _, self.tcp = sim.simxGetObjectHandle(self.clientID, 'Tip', sim.simx_opmode_blocking)   # Punta TCP
        _, self.objetivo = sim.simxGetObjectHandle(self.clientID, 'Esfera', sim.simx_opmode_blocking)   # Objeto
        # ESPACIO DE OBSERVACION        
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype='float32')
        #spaces.Box(shape=(self.tip,), low=1, high=3, dtype = 'float32')   # observacion 
        # ESPACIO DE ACCION
        self.action_space = spaces.Box(-1., 1, shape = (self.n_actions,), dtype = 'float32') 
#----------------------------------------------------------------------------------------------------------------------------
#                                                  METODOS GYMNASIUM
#----------------------------------------------------------------------------------------------------------------------------
    def step(self, action):
        print('--------------------------------------------------------------------')
        self.step_counter += 1
        print('STEP: ', self.step_counter)
        self.truncated, self.terminated = False, False 
        print('Acciones: ', action)
        op = self.get_orientacion(self.target)   # Orientacion del efector final
        # EJECUTAR ACCION  
        self.control_ef(action)  # Establecer una accion del agente
        # OBTENER INFORMACION ADICIONAL
        self.info = self._get_info()
        self.obs = self._get_obs()   # Imagenes
        print(f'Distancia: {self.med_dist():.3f}',' cm')  
        # RECOMPENSAS y PENALIZACIONES
        rt = self.compute_reward(action)
        self.reward = rt
        print(f'--- RECOMPENSA TOTAL: {self.reward:.3f} --- ')  
        print(self.get_posicion(self.objetivo))  # posicion de la esfera
        # REINICIOS
        if self.med_dist() <= 10:   # Tarea completa
            self.terminated = True
            print(' --- ESTADO TERMINADO / TAREA COMPLETA --- ')
         
        return self.obs, self.reward, self.terminated, self.truncated, self.info
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)
        
        pos_ef = self.default_pos.copy()
        set_posIni = self.set_posicion(pos_ef)
        self.set_orientacion([0.0, 0.0, 0.0])
        observation = self._get_obs()   # Establecer la observacion
        info = self._get_info()
        return observation, info

    def render(self, mode="rgb_array"):

        return 
    
    def close(self): # no cambiar
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        sim.simxGetPingTime(self.clientID)
        sim.simxFinish(self.clientID)   # Cerrar conexion a CoppeliaSim
#--------------------------------------------------------------------------------------------------------------------------
#                                                FUNCIONES CONTROL/CONFIGURACION
#--------------------------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        """
        La observacion sera la posicion actual del efector final
        """
        p = self.get_posicion(self.target)
        return p
        
    def get_posicion(self, handle):
        """ 
            Obtener la posicion de un objeto
            Retorno: 
                Array que contiene (X,Y,Z) del objecto.
        """
        _, position = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_blocking)
        _, quaternion = sim.simxGetObjectQuaternion(self.clientID, handle, -1, sim.simx_opmode_blocking)
        return np.array(position, dtype=np.float32)   #np.r_[position, quaternion]
    
    def get_orientacion(self, handle):
        _, orientacion = sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_oneshot)
        og = [180.0 * a / math.pi for a in orientacion]   # de radianes a grados
        return np.array(og, dtype=np.float32)   # angulos Euler
    
    def set_orientacion(self, orAngles):
        or_Angles = [math.pi * a / 180.0 for a in orAngles]   # de grados a radianes
        sim.simxSetObjectOrientation(self.clientID, self.target, -1, or_Angles, sim.simx_opmode_blocking)
    
    def set_posicion(self, pos):
        """
            Establecer la posicion (X, Y, Z) de un objeto en la escena. Manejador Target.
        """
        sim.simxSetObjectPosition(self.clientID, self.target, -1, pos, sim.simx_opmode_blocking)
    
    def med_dist(self):
        """
            Medir la distancia entre el Efector final y la esfera
            Retorno:
                Distancia[metros]
        """
        
        pos_ef = self.get_posicion(self.target)  # posicion actual del efector final
        pos_ob = self.get_posicion(self.objetivo)  # posicion de la esfera
        
        # Calcula la distancia euclidiana en 3D
        dist_ab = math.sqrt(
            (pos_ef[0] - pos_ob[0])**2 +
            (pos_ef[1] - pos_ob[1])**2 +
            (pos_ef[2] - pos_ob[2])**2
        ) * 100   # en cm
        
        return dist_ab         
        
    def control_ef(self, accion):
        """
            Establecer una accion de movimiento del efector final del robot.
            Movimientos en plano XY y orientacion en Z.
            Comentar set_orientacion() si se va usar para trayectorias rectas. 
        """
        
        pos_act = self.get_posicion(self.target)  # posicion actual del efector
        or_act = self.get_orientacion(self.target)  # orientacion actual del efector
        
        #new_posicion = [pos_act[i] + (accion[i]/200) for i in range(2)]   # Calculo de movimiento XYZ
        new_posicion = [pos_act[0] + (accion[0]/200), pos_act[1] + (accion[1]/200), pos_act[2] + (accion[2]/200)]   # valor en 200
        #accionz = np.interp(accion[2], rango_origen, rango_destino)   # cambiar rango de valores 
        #new_orientacion = [or_act[0], or_act[1], or_act[2] + (accion[2])]   # Calculo de orientacion Z

        self.set_posicion(new_posicion)  # Establecer posicion X-Y-Z
        #self.set_orientacion(new_orientacion)   # Establecer orientacion en Z
        
    def _get_info(self): # verificar
        """
            Medir distancia por el efector final.
        """
        dist_i = self.med_dist()
        return {
            'info' : dist_i
        }  
       
    
    def compute_reward(self, accion):
        """
            Funcion de recompensa para alcanzar con el EF un objetivo.
            Entrada: 
                Accion.
            Retorno
                TotalReward
        """
        # Adquirir informacion de las posicion en X, Y, Z del efector final y de la esfera
        pos_ef = self.get_posicion(self.target)  # posicion actual del efector final
        pos_ob = self.get_posicion(self.objetivo)  # posicion de la esfera
        
        # Calcula la distancia euclidiana en 3D
        distancia = math.sqrt(
            (pos_ef[0] - pos_ob[0])**2 +
            (pos_ef[1] - pos_ob[1])**2 +
            (pos_ef[2] - pos_ob[2])**2
        )
        

        # Definir la recompensa como inversamente proporcional a la distancia
        self.recompensa_total = -distancia * 10 # Penalización por distancia
        
        return self.recompensa_total