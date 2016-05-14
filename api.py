# coding: iso-8859-1
from _api import *

# Taille du terrain (longueur et largeur).
TAILLE_TERRAIN = 39

# Longueur de chacune de vos deux bases.
LONGUEUR_BASE = 13

# Nombre de tours à jouer avant la fin de la partie.
NB_TOURS = 100

# Nombre de points d'action par tour.
NB_POINTS_ACTION = 4

# Nombre de points d'action que coûte la construction d'un tuyau.
COUT_CONSTRUCTION = 1

# Nombre de points d'action que coûte l'amélioration d'un tuyau.
COUT_AMELIORATION = 1

# Nombre de points d'action que coûte la destruction d'un tuyau.
COUT_DESTRUCTION = 3

# Nombre de points d'action que coûte la destruction d'un Super Tuyau™.
COUT_DESTRUCTION_SUPER_TUYAU = 4

# Charge en plasma nécessaire pour la destruction d'un tuyau ou d'un Super Tuyau™.
CHARGE_DESTRUCTION = 2.0

# Nombre de points d'action que coûte le déblayage d'une case de débris.
COUT_DEBLAYAGE = 2

# Nombre de points d'action que coûte le déplacement d'une unité de puissance d'aspiration de la base (la première modification de chaque tour est offerte).
COUT_MODIFICATION_ASPIRATION = 1

# Limite de puissance d'aspiration sur une case de base.
LIMITE_ASPIRATION = 5

# Vitesse du plasma dans un tuyau normal, en nombre de cases par tour.
VITESSE_TUYAU = 1

# Multiplicateur de la vitesse du plasma dans un Super Tuyau™.
MULTIPLICATEUR_VITESSE_SUPER_TUYAU = 2


from enum import IntEnum

# Erreurs possibles
class erreur(IntEnum):
    OK = 0  # <- L'action a été exécutée avec succès.
    PA_INSUFFISANTS = 1  # <- Vous ne possédez pas assez de points d'action pour cette action.
    AUCUN_TUYAU = 2  # <- Il n'y a pas de tuyau à la position spécifiée.
    POSITION_INVALIDE = 3  # <- La position spécifiée est hors de la carte.
    PUISSANCE_INSUFFISANTE = 4  # <- Vous ne possédez pas assez de puissance d'asipration sur cette partie de la base.
    DEPLACEMENT_INVALIDE = 5  # <- Vous ne pouvez pas déplacer de la puissance d'aspiration d'une case à elle-même.
    PAS_DANS_BASE = 6  # <- Cette case n'appartient pas à votre base.
    AMELIORATION_IMPOSSIBLE = 7  # <- Il y a déjà un Super Tuyau™ sur cette case.
    CONSTRUCTION_IMPOSSIBLE = 8  # <- Il est impossible de construire un tuyau à la position indiquée.
    DESTRUCTION_IMPOSSIBLE = 9  # <- Il n'y a pas de tuyau à la position spécifiée.
    PAS_DE_PULSAR = 10  # <- Il n'y a pas de pulsar à la position spécifiée.
    PAS_DE_DEBRIS = 11  # <- Il n'y a pas de débris à la position spécifiée.
    CHARGE_INSUFFISANTE = 12  # <- Vous ne possédez pas assez de plasma pour lancer une destruction.
    LIMITE_ASPIRATION_ATTEINTE = 13  # <- Vous avez atteint la limite d'aspiration sur cette case.


# Types de cases
class case_type(IntEnum):
    VIDE = 0  # <- Case vide
    TUYAU = 1  # <- Case contenant un tuyau
    SUPER_TUYAU = 2  # <- Case contenant un Super Tuyau™
    DEBRIS = 3  # <- Case contenant des débris à déblayer
    PULSAR = 4  # <- Case contenant un pulsar
    BASE = 5  # <- Case appartenant à une base d'un des joueurs
    INTERDIT = 6  # <- Case où aucune action n'est possible


from collections import namedtuple

# Position sur la carte, donnée par deux coordonnées.

# Représente un pulsar existant.
pulsar_info = namedtuple("pulsar_info",
    'periode ' # <- Période de pulsation du pulsar
    'puissance ' # <- Quantité de plasma émise par chaque pulsation dans chaque direction
    'pulsations_restantes ' # <- Nombre de pulsations restantes
    'pulsations_totales ' # <- Nombre total de pulsations au début de la partie
)


