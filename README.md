<h1> DEMO DRL: Alcanzar un objetivo </h1>

<h2> Descrpcion </h2>

Demo: Sistema de control robotico basado en DRL para un manipulador robotico. El robot debe alcanzar con su efector final una esfera dentro de la escena de simulacion en CoppeliaSim.
<h2> Software: </h2>

- CoppeliaSim 4.7 64 bits
- Gymnasium 0.29
- Numpy 1.25
- Matplotlib 3.7.1
- Python 3.11
- Stable Baselines3 2.0
- Anaconda 2.5.0 - Jupyter 6.5.4
<h2> Contenido del repositorio </h2>

- Escena de simulacion del manipulador robotico en CoppeliaSim (Entorno_MRR_DRL_IK.ttt).
- Archivos .ipynb para el Entrenamiento e Inferencia.
<h2> Procedure </h2>

- Instalar Anaconda, los archivos para entrenamiento e inferencia se ejecutan en los cuadernos de Jupyter.
- Instalar y abrir CoppeliaSim.
- Abrir la escena de simulacion y ejecutarla antes de iniciar con el entrenamiento.
- Abrir el archivo para el entrenamiento del agente DRL, en la celda 5 se define y entrena el agente con algoritmo A2C (configurar los parametros del agente). E iniciara el proceso de entrenamiento (por defecto estara en 1k steps)
- Una vez culminado el proceso de entrenamiento se guardara en la carpeta "logs" un archivo con nombre "best_model", este archivo es el modelo drl entrenado que sera usado en la inferencia.
- Abrir el archivo de inferencia y cargar el modelo entrenado. Aqui se mostrara el proceso obtenido durante el entrenamiento.
<h2> Funcionamiento </h2>

La escena de simulacion consta de un manipulador robotico y una esfera. El efector final del robot debe alcanzar la esfera dentro de la zona de trabajo predeterminada. el robot esta configura en modo IK, de modo que se controlara el movimiento desde el EF. La posicion inicial del robot esta predeterminada en la escena, si el target del robot esta fuera de alcnanza mostrara una advertencia de "Falla IK" y debera reiniciar la simulacion y por ende el entrenamiento o inferencia. EL reinicio del sistema drl se basa cuando el EF alcance la esfera, la posicion del robot vuelve a su posicion predeterminada, por ende se dara completa esta tarea (episodio) y se volvera a intentar alcanzar el objetivo nuevamente.
El nucleo principal de este sistema esta basado en el archivo EnvCRL, aqui se agregan funciones de comportamiento del robot y construccion del entorno para el agente. S i existen modificaciones que desea realizar, puede hacerlo en este archivo.
