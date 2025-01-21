from setuptools import setup

    # Este entorno requiere instalar el simulador CoppeliaSim
    # para poder utilizar en simulacion la escena del demo de
    # control de un manipulador robotico.   

setup(
    name="DemoC_RL",
    version="0.0.1",
    install_requires=["gymnasium"],
)