/// This file has been generated, if you wish to
/// modify it in a permanent way, please refer
/// to the script file : gen/generator_python.rb

#ifndef INTERFACE_HH_
# define INTERFACE_HH_

# include <Python.h>
# include <vector>
# include <string>

/// Erreurs possibles
typedef enum erreur {
  OK, /* <- L'action a été exécutée avec succès. */
  PA_INSUFFISANTS, /* <- Vous ne possédez pas assez de points d'action pour cette action. */
  AUCUN_TUYAU, /* <- Il n'y a pas de tuyau à la position spécifiée. */
  POSITION_INVALIDE, /* <- La position spécifiée est hors de la carte. */
  PUISSANCE_INSUFFISANTE, /* <- Vous ne possédez pas assez de puissance d'asipration sur cette partie de la base. */
  DEPLACEMENT_INVALIDE, /* <- Vous ne pouvez pas déplacer de la puissance d'aspiration d'une case à elle-même. */
  PAS_DANS_BASE, /* <- Cette case n'appartient pas à votre base. */
  AMELIORATION_IMPOSSIBLE, /* <- Il y a déjà un Super Tuyau™ sur cette case. */
  CONSTRUCTION_IMPOSSIBLE, /* <- Il est impossible de construire un tuyau à la position indiquée. */
  DESTRUCTION_IMPOSSIBLE, /* <- Il n'y a pas de tuyau à la position spécifiée. */
  PAS_DE_PULSAR, /* <- Il n'y a pas de pulsar à la position spécifiée. */
  PAS_DE_DEBRIS, /* <- Il n'y a pas de débris à la position spécifiée. */
  CHARGE_INSUFFISANTE, /* <- Vous ne possédez pas assez de plasma pour lancer une destruction. */
  LIMITE_ASPIRATION_ATTEINTE, /* <- Vous avez atteint la limite d'aspiration sur cette case. */
} erreur;


/// Types de cases
typedef enum case_type {
  VIDE, /* <- Case vide */
  TUYAU, /* <- Case contenant un tuyau */
  SUPER_TUYAU, /* <- Case contenant un Super Tuyau™ */
  DEBRIS, /* <- Case contenant des débris à déblayer */
  PULSAR, /* <- Case contenant un pulsar */
  BASE, /* <- Case appartenant à une base d'un des joueurs */
  INTERDIT, /* <- Case où aucune action n'est possible */
} case_type;


/// Position sur la carte, donnée par deux coordonnées.
typedef struct position {
  int x;  /* <- Coordonnée en X */
  int y;  /* <- Coordonnée en Y */
} position;


/// Représente un pulsar existant.
typedef struct pulsar_info {
  int periode;  /* <- Période de pulsation du pulsar */
  double puissance;  /* <- Quantité de plasma émise par chaque pulsation dans chaque direction */
  int pulsations_restantes;  /* <- Nombre de pulsations restantes */
  int pulsations_totales;  /* <- Nombre total de pulsations au début de la partie */
} pulsar_info;



extern "C" {

/// Construit un tuyau sur une case donnée.
erreur api_construire(position pos);

/// Améliore un tuyau en Super Tuyau™.
erreur api_ameliorer(position pos);

/// Détruit un tuyau sur une case donnée.
erreur api_detruire(position pos);

/// Déplace une unité de puissance d'aspiration d'une case de votre base vers une autre.
erreur api_deplacer_aspiration(position source, position destination);

/// Déblaye une case de débris.
erreur api_deblayer(position pos);

/// Renvoie le type d'une case donnée.
case_type api_type_case(position pos);

/// Renvoie la liste de tous les pulsars présents.
std::vector<position> api_liste_pulsars();

/// Renvoie la liste des cases contenant du plasma.
std::vector<position> api_liste_plasmas();

/// Renvoie la liste des cases contenant un tuyau ou Super Tuyau™.
std::vector<position> api_liste_tuyaux();

/// Renvoie la liste des cases contenant un Super Tuyau™.
std::vector<position> api_liste_super_tuyaux();

/// Renvoie la liste des cases contenant des débris.
std::vector<position> api_liste_debris();

/// Renvoie vrai si et seulement si la case contient un pulsar.
bool api_est_pulsar(position pos);

/// Renvoie vrai si et seulement si la case contient un tuyau ou un Super Tuyau™.
bool api_est_tuyau(position pos);

/// Renvoie vrai si et seulement si la case contient un simple tuyau.
bool api_est_simple_tuyau(position pos);

/// Renvoie vrai si et seulement si la case contient un Super Tuyau™.
bool api_est_super_tuyau(position pos);

/// Renvoie vrai si et seulement si la case contient un débris.
bool api_est_debris(position pos);

/// Renvoie vrai si et seulement s'il est possible de construire sur cette case.
bool api_est_libre(position pos);

/// Renvoie les informations propres au pulsar à la position donnée.
pulsar_info api_info_pulsar(position pos);

/// Renvoie la quantité de plasma sur une case donnée.
double api_charges_presentes(position pos);

/// Renvoie le numéro du joueur ayant dernièrement construit ou amélioré un tuyau sur une case.
int api_constructeur_tuyau(position pos);

/// Renvoie le numéro du propriétaire d'une case de base.
int api_proprietaire_base(position pos);

/// Renvoie la liste des cases composant votre base.
std::vector<position> api_ma_base();

/// Renvoie la liste des cases composant la base de votre ennemi.
std::vector<position> api_base_ennemie();

/// Renvoie la puissance d'aspiration d'une case de base donnée. Renvoie -1 si la position n'est pas celle d'une base.
int api_puissance_aspiration(position pos);

/// Renvoie pour une case donnée la liste des cases vers lesquelles le plasma se propagera. Renvoie la liste vide si la case n'est pas une case contenant un tuyau ou si elle n'est reliée à aucune base.
std::vector<position> api_directions_plasma(position pos);

/// Renvoie la valeur du coût de la prochaine modification de vos puissances d'aspiration.
int api_cout_prochaine_modification_aspiration();

/// Renvoie la liste des tuyaux construits par votre adversaire au dernier tour.
std::vector<position> api_hist_tuyaux_construits();

/// Renvoie la liste des tuyaux détruits par votre adversaire au dernier tour.
std::vector<position> api_hist_tuyaux_detruits();

/// Renvoie la liste des tuyaux améliorés par votre adversaire au dernier tour.
std::vector<position> api_hist_tuyaux_ameliores();

/// Renvoie la liste des débris déblayés par votre adversaire au dernier tour.
std::vector<position> api_hist_debris_deblayes();

/// Renvoie la liste des cases de base de votre adversaire qui ont reçu un point d'aspiration (une même case peut apparaître plusieurs fois).
std::vector<position> api_hist_points_aspiration_ajoutes();

/// Renvoie la liste des cases de base de votre adversaire qui ont perdu un point d'aspiration (une même case peut apparaître plusieurs fois).
std::vector<position> api_hist_points_aspiration_retires();

/// Renvoie votre numéro de joueur.
int api_moi();

/// Renvoie le numéro de votre adversaire.
int api_adversaire();

/// Indique votre nombre de points d'actions restants pour ce tour-ci.
int api_points_action();

/// Renvoie le score du joueur désigné par le numéro ``id_joueur``. Renvoie -1 si ``id_joueur`` est invalide.
int api_score(int id_joueur);

/// Renvoie le numéro du tour actuel.
int api_tour_actuel();

/// Annule la dernière action. Renvoie ``false`` quand il n'y a pas d'action à annuler ce tour-ci.
bool api_annuler();

/// Affiche le contenu d'une valeur de type erreur
void api_afficher_erreur(erreur v);

/// Affiche le contenu d'une valeur de type case_type
void api_afficher_case_type(case_type v);

/// Affiche le contenu d'une valeur de type position
void api_afficher_position(position v);

/// Affiche le contenu d'une valeur de type pulsar_info
void api_afficher_pulsar_info(pulsar_info v);

}

#endif // !INTERFACE_HH_
