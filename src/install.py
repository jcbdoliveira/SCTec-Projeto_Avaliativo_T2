import subprocess
import sys

def instalar_bibliotecas():
    # Lista de bibliotecas a serem instaladas
    bibliotecas = [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn", 
        "re",
        "scipy",       
        "seaborn"
    ]

    for biblioteca in bibliotecas:
        try:
            __import__(biblioteca)
            print(f"A biblioteca '{biblioteca}' já está instalada.")
        except ImportError:
            print(f"A biblioteca '{biblioteca}' não está instalada. Instalando...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", biblioteca])
            print(f"A biblioteca '{biblioteca}' foi instalada com sucesso.")