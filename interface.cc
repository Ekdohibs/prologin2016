/// This file has been generated, if you wish to
/// modify it in a permanent way, please refer
/// to the script file : gen/generator_python.rb

#include "interface.hh"

static PyObject* c_module;
static PyObject* py_module;
static PyObject* champ_module;

static void _init_python();

template <typename Lang, typename Cxx>
Lang cxx2lang(Cxx in)
{
  return in.__if_that_triggers_an_error_there_is_a_problem;
}

template <>
PyObject* cxx2lang<PyObject*, int>(int in)
{
  return PyLong_FromLong(in);
}


template <>
PyObject* cxx2lang<PyObject*, double>(double in)
{
  return PyFloat_FromDouble(in);
}


template <>
PyObject* cxx2lang<PyObject*, std::string>(std::string in)
{
  return PyUnicode_FromString(in.c_str());
}


template <>
PyObject* cxx2lang<PyObject*, bool>(bool in)
{
  return PyBool_FromLong(in);
}

template <typename Cxx>
PyObject* cxx2lang_array(const std::vector<Cxx>& in)
{
  size_t size = in.size();
  PyObject* out = PyList_New(size);

  for (unsigned int i = 0; i < size; ++i)
    PyList_SET_ITEM(out, i, (cxx2lang<PyObject*, Cxx>(in[i])));

  return out;
}

template <typename Lang, typename Cxx>
Cxx lang2cxx(Lang in)
{
  return in.__if_that_triggers_an_error_there_is_a_problem;
}

template <>
int lang2cxx<PyObject*, int>(PyObject* in)
{
  long out = PyLong_AsLong(in);
  if (PyErr_Occurred())
    throw 42;
  return out;
}

template <>
double lang2cxx<PyObject*, double>(PyObject* in)
{
  double out = PyFloat_AsDouble(in);
  if (PyErr_Occurred())
    throw 42;
  return out;
}

template <>
bool lang2cxx<PyObject*, bool>(PyObject* in)
{
  return (bool)lang2cxx<PyObject*, int>(in);
}

template <>
std::string lang2cxx<PyObject*, std::string>(PyObject* in)
{
  char * out = PyUnicode_AsUTF8(in);
  if (PyErr_Occurred())
    {
      throw 42;
    }
  return out;
}

template <typename Cxx>
std::vector<Cxx> lang2cxx_array(PyObject* in)
{
  if (!PyList_Check(in))
  {
    PyErr_SetString(PyExc_TypeError, "a list is required");
    throw 42;
  }

  std::vector<Cxx> out;
  unsigned int size = PyList_Size(in);

  for (unsigned int i = 0; i < size; ++i)
    out.push_back(lang2cxx<PyObject*, Cxx>(PyList_GET_ITEM(in, i)));

  return out;
}
/// Erreurs possibles
template <>
PyObject* cxx2lang<PyObject*, erreur>(erreur in)
{
  PyObject* name = PyUnicode_FromString("erreur");
  PyObject* enm = PyObject_GetAttr(py_module, name);
  if (enm == NULL) throw 42;
  PyObject* arglist = Py_BuildValue("(i)", (int) in);
  PyObject* ret = PyObject_CallObject(enm, arglist);
  Py_DECREF(name);
  Py_DECREF(arglist);
  Py_DECREF(enm);
  return ret;
}

template <>
erreur lang2cxx<PyObject*, erreur>(PyObject* in)
{
  return (erreur)lang2cxx<PyObject*, int>(in);
}

/// Types de cases
template <>
PyObject* cxx2lang<PyObject*, case_type>(case_type in)
{
  PyObject* name = PyUnicode_FromString("case_type");
  PyObject* enm = PyObject_GetAttr(py_module, name);
  if (enm == NULL) throw 42;
  PyObject* arglist = Py_BuildValue("(i)", (int) in);
  PyObject* ret = PyObject_CallObject(enm, arglist);
  Py_DECREF(name);
  Py_DECREF(arglist);
  Py_DECREF(enm);
  return ret;
}

template <>
case_type lang2cxx<PyObject*, case_type>(PyObject* in)
{
  return (case_type)lang2cxx<PyObject*, int>(in);
}

/// Position sur la carte, donnée par deux coordonnées.
template <>
PyObject* cxx2lang<PyObject*, position>(position in)
{
  PyObject* tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(tuple, 0, (cxx2lang<PyObject*, int>(in.x)));
  PyTuple_SET_ITEM(tuple, 1, (cxx2lang<PyObject*, int>(in.y)));
  return tuple;
}

template <>
position lang2cxx<PyObject*, position>(PyObject* in)
{
  position out;
  PyObject* i;
  i = PyTuple_GetItem(in, 0);
  if (i == NULL) throw 42;
  out.x = lang2cxx<PyObject*, int>(i);
  i = PyTuple_GetItem(in, 1);
  if (i == NULL) throw 42;
  out.y = lang2cxx<PyObject*, int>(i);
  return out;
}

/// Représente un pulsar existant.
template <>
PyObject* cxx2lang<PyObject*, pulsar_info>(pulsar_info in)
{
  PyObject* tuple = PyTuple_New(4);
  PyTuple_SET_ITEM(tuple, 0, (cxx2lang<PyObject*, int>(in.periode)));
  PyTuple_SET_ITEM(tuple, 1, (cxx2lang<PyObject*, double>(in.puissance)));
  PyTuple_SET_ITEM(tuple, 2, (cxx2lang<PyObject*, int>(in.pulsations_restantes)));
  PyTuple_SET_ITEM(tuple, 3, (cxx2lang<PyObject*, int>(in.pulsations_totales)));
  PyObject* name = PyUnicode_FromString("pulsar_info");
  PyObject* cstr = PyObject_GetAttr(py_module, name);
  Py_DECREF(name);
  if (cstr == NULL) throw 42;
  PyObject* ret = PyObject_CallObject(cstr, tuple);
  Py_DECREF(cstr);
  Py_DECREF(tuple);
  if (ret == NULL) throw 42;
  return ret;
}

template <>
pulsar_info lang2cxx<PyObject*, pulsar_info>(PyObject* in)
{
  pulsar_info out;
  PyObject* i;
  i = cxx2lang<PyObject*, int>(0);
  i = PyObject_GetItem(in, i);
  if (i == NULL) throw 42;
  out.periode = lang2cxx<PyObject*, int>(i);
  Py_DECREF(i);
  i = cxx2lang<PyObject*, int>(1);
  i = PyObject_GetItem(in, i);
  if (i == NULL) throw 42;
  out.puissance = lang2cxx<PyObject*, double>(i);
  Py_DECREF(i);
  i = cxx2lang<PyObject*, int>(2);
  i = PyObject_GetItem(in, i);
  if (i == NULL) throw 42;
  out.pulsations_restantes = lang2cxx<PyObject*, int>(i);
  Py_DECREF(i);
  i = cxx2lang<PyObject*, int>(3);
  i = PyObject_GetItem(in, i);
  if (i == NULL) throw 42;
  out.pulsations_totales = lang2cxx<PyObject*, int>(i);
  Py_DECREF(i);
  return out;
}

/// Construit un tuyau sur une case donnée.
static PyObject* p_construire(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, erreur>(api_construire(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Améliore un tuyau en Super Tuyau™.
static PyObject* p_ameliorer(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, erreur>(api_ameliorer(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Détruit un tuyau sur une case donnée.
static PyObject* p_detruire(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, erreur>(api_detruire(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Déplace une unité de puissance d'aspiration d'une case de votre base vers une autre.
static PyObject* p_deplacer_aspiration(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
PyObject* a1;
  if (!PyArg_ParseTuple(args, "OO", &a0, &a1)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, erreur>(api_deplacer_aspiration(lang2cxx<PyObject*, position>(a0), lang2cxx<PyObject*, position>(a1)));
  } catch (...) { return NULL; }
}

/// Déblaye une case de débris.
static PyObject* p_deblayer(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, erreur>(api_deblayer(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie le type d'une case donnée.
static PyObject* p_type_case(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, case_type>(api_type_case(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie la liste de tous les pulsars présents.
static PyObject* p_liste_pulsars(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_liste_pulsars());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases contenant du plasma.
static PyObject* p_liste_plasmas(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_liste_plasmas());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases contenant un tuyau ou Super Tuyau™.
static PyObject* p_liste_tuyaux(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_liste_tuyaux());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases contenant un Super Tuyau™.
static PyObject* p_liste_super_tuyaux(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_liste_super_tuyaux());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases contenant des débris.
static PyObject* p_liste_debris(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_liste_debris());
  } catch (...) { return NULL; }
}

/// Renvoie vrai si et seulement si la case contient un pulsar.
static PyObject* p_est_pulsar(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, bool>(api_est_pulsar(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie vrai si et seulement si la case contient un tuyau ou un Super Tuyau™.
static PyObject* p_est_tuyau(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, bool>(api_est_tuyau(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie vrai si et seulement si la case contient un simple tuyau.
static PyObject* p_est_simple_tuyau(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, bool>(api_est_simple_tuyau(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie vrai si et seulement si la case contient un Super Tuyau™.
static PyObject* p_est_super_tuyau(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, bool>(api_est_super_tuyau(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie vrai si et seulement si la case contient un débris.
static PyObject* p_est_debris(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, bool>(api_est_debris(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie vrai si et seulement s'il est possible de construire sur cette case.
static PyObject* p_est_libre(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, bool>(api_est_libre(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie les informations propres au pulsar à la position donnée.
static PyObject* p_info_pulsar(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, pulsar_info>(api_info_pulsar(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie la quantité de plasma sur une case donnée.
static PyObject* p_charges_presentes(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, double>(api_charges_presentes(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie le numéro du joueur ayant dernièrement construit ou amélioré un tuyau sur une case.
static PyObject* p_constructeur_tuyau(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_constructeur_tuyau(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie le numéro du propriétaire d'une case de base.
static PyObject* p_proprietaire_base(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_proprietaire_base(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases composant votre base.
static PyObject* p_ma_base(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_ma_base());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases composant la base de votre ennemi.
static PyObject* p_base_ennemie(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_base_ennemie());
  } catch (...) { return NULL; }
}

/// Renvoie la puissance d'aspiration d'une case de base donnée. Renvoie -1 si la position n'est pas celle d'une base.
static PyObject* p_puissance_aspiration(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_puissance_aspiration(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie pour une case donnée la liste des cases vers lesquelles le plasma se propagera. Renvoie la liste vide si la case n'est pas une case contenant un tuyau ou si elle n'est reliée à aucune base.
static PyObject* p_directions_plasma(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang_array(api_directions_plasma(lang2cxx<PyObject*, position>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie la valeur du coût de la prochaine modification de vos puissances d'aspiration.
static PyObject* p_cout_prochaine_modification_aspiration(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_cout_prochaine_modification_aspiration());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des tuyaux construits par votre adversaire au dernier tour.
static PyObject* p_hist_tuyaux_construits(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_hist_tuyaux_construits());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des tuyaux détruits par votre adversaire au dernier tour.
static PyObject* p_hist_tuyaux_detruits(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_hist_tuyaux_detruits());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des tuyaux améliorés par votre adversaire au dernier tour.
static PyObject* p_hist_tuyaux_ameliores(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_hist_tuyaux_ameliores());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des débris déblayés par votre adversaire au dernier tour.
static PyObject* p_hist_debris_deblayes(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_hist_debris_deblayes());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases de base de votre adversaire qui ont reçu un point d'aspiration (une même case peut apparaître plusieurs fois).
static PyObject* p_hist_points_aspiration_ajoutes(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_hist_points_aspiration_ajoutes());
  } catch (...) { return NULL; }
}

/// Renvoie la liste des cases de base de votre adversaire qui ont perdu un point d'aspiration (une même case peut apparaître plusieurs fois).
static PyObject* p_hist_points_aspiration_retires(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang_array(api_hist_points_aspiration_retires());
  } catch (...) { return NULL; }
}

/// Renvoie votre numéro de joueur.
static PyObject* p_moi(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_moi());
  } catch (...) { return NULL; }
}

/// Renvoie le numéro de votre adversaire.
static PyObject* p_adversaire(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_adversaire());
  } catch (...) { return NULL; }
}

/// Indique votre nombre de points d'actions restants pour ce tour-ci.
static PyObject* p_points_action(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_points_action());
  } catch (...) { return NULL; }
}

/// Renvoie le score du joueur désigné par le numéro ``id_joueur``. Renvoie -1 si ``id_joueur`` est invalide.
static PyObject* p_score(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_score(lang2cxx<PyObject*, int>(a0)));
  } catch (...) { return NULL; }
}

/// Renvoie le numéro du tour actuel.
static PyObject* p_tour_actuel(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, int>(api_tour_actuel());
  } catch (...) { return NULL; }
}

/// Annule la dernière action. Renvoie ``false`` quand il n'y a pas d'action à annuler ce tour-ci.
static PyObject* p_annuler(PyObject* self, PyObject* args)
{
  (void)self;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
    try {
return cxx2lang<PyObject*, bool>(api_annuler());
  } catch (...) { return NULL; }
}

/// Affiche le contenu d'une valeur de type erreur
static PyObject* p_afficher_erreur(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
api_afficher_erreur(lang2cxx<PyObject*, erreur>(a0));
  Py_INCREF(Py_None);
  return Py_None;
  } catch (...) { return NULL; }
}

/// Affiche le contenu d'une valeur de type case_type
static PyObject* p_afficher_case_type(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
api_afficher_case_type(lang2cxx<PyObject*, case_type>(a0));
  Py_INCREF(Py_None);
  return Py_None;
  } catch (...) { return NULL; }
}

/// Affiche le contenu d'une valeur de type position
static PyObject* p_afficher_position(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
api_afficher_position(lang2cxx<PyObject*, position>(a0));
  Py_INCREF(Py_None);
  return Py_None;
  } catch (...) { return NULL; }
}

/// Affiche le contenu d'une valeur de type pulsar_info
static PyObject* p_afficher_pulsar_info(PyObject* self, PyObject* args)
{
  (void)self;
PyObject* a0;
  if (!PyArg_ParseTuple(args, "O", &a0)) {
    return NULL;
  }
    try {
api_afficher_pulsar_info(lang2cxx<PyObject*, pulsar_info>(a0));
  Py_INCREF(Py_None);
  return Py_None;
  } catch (...) { return NULL; }
}


/*
** Api functions to register.
*/
static PyMethodDef api_callback[] = {
  {"construire", p_construire, METH_VARARGS, "construire"},  {"ameliorer", p_ameliorer, METH_VARARGS, "ameliorer"},  {"detruire", p_detruire, METH_VARARGS, "detruire"},  {"deplacer_aspiration", p_deplacer_aspiration, METH_VARARGS, "deplacer_aspiration"},  {"deblayer", p_deblayer, METH_VARARGS, "deblayer"},  {"type_case", p_type_case, METH_VARARGS, "type_case"},  {"liste_pulsars", p_liste_pulsars, METH_VARARGS, "liste_pulsars"},  {"liste_plasmas", p_liste_plasmas, METH_VARARGS, "liste_plasmas"},  {"liste_tuyaux", p_liste_tuyaux, METH_VARARGS, "liste_tuyaux"},  {"liste_super_tuyaux", p_liste_super_tuyaux, METH_VARARGS, "liste_super_tuyaux"},  {"liste_debris", p_liste_debris, METH_VARARGS, "liste_debris"},  {"est_pulsar", p_est_pulsar, METH_VARARGS, "est_pulsar"},  {"est_tuyau", p_est_tuyau, METH_VARARGS, "est_tuyau"},  {"est_simple_tuyau", p_est_simple_tuyau, METH_VARARGS, "est_simple_tuyau"},  {"est_super_tuyau", p_est_super_tuyau, METH_VARARGS, "est_super_tuyau"},  {"est_debris", p_est_debris, METH_VARARGS, "est_debris"},  {"est_libre", p_est_libre, METH_VARARGS, "est_libre"},  {"info_pulsar", p_info_pulsar, METH_VARARGS, "info_pulsar"},  {"charges_presentes", p_charges_presentes, METH_VARARGS, "charges_presentes"},  {"constructeur_tuyau", p_constructeur_tuyau, METH_VARARGS, "constructeur_tuyau"},  {"proprietaire_base", p_proprietaire_base, METH_VARARGS, "proprietaire_base"},  {"ma_base", p_ma_base, METH_VARARGS, "ma_base"},  {"base_ennemie", p_base_ennemie, METH_VARARGS, "base_ennemie"},  {"puissance_aspiration", p_puissance_aspiration, METH_VARARGS, "puissance_aspiration"},  {"directions_plasma", p_directions_plasma, METH_VARARGS, "directions_plasma"},  {"cout_prochaine_modification_aspiration", p_cout_prochaine_modification_aspiration, METH_VARARGS, "cout_prochaine_modification_aspiration"},  {"hist_tuyaux_construits", p_hist_tuyaux_construits, METH_VARARGS, "hist_tuyaux_construits"},  {"hist_tuyaux_detruits", p_hist_tuyaux_detruits, METH_VARARGS, "hist_tuyaux_detruits"},  {"hist_tuyaux_ameliores", p_hist_tuyaux_ameliores, METH_VARARGS, "hist_tuyaux_ameliores"},  {"hist_debris_deblayes", p_hist_debris_deblayes, METH_VARARGS, "hist_debris_deblayes"},  {"hist_points_aspiration_ajoutes", p_hist_points_aspiration_ajoutes, METH_VARARGS, "hist_points_aspiration_ajoutes"},  {"hist_points_aspiration_retires", p_hist_points_aspiration_retires, METH_VARARGS, "hist_points_aspiration_retires"},  {"moi", p_moi, METH_VARARGS, "moi"},  {"adversaire", p_adversaire, METH_VARARGS, "adversaire"},  {"points_action", p_points_action, METH_VARARGS, "points_action"},  {"score", p_score, METH_VARARGS, "score"},  {"tour_actuel", p_tour_actuel, METH_VARARGS, "tour_actuel"},  {"annuler", p_annuler, METH_VARARGS, "annuler"},  {"afficher_erreur", p_afficher_erreur, METH_VARARGS, "afficher_erreur"},  {"afficher_case_type", p_afficher_case_type, METH_VARARGS, "afficher_case_type"},  {"afficher_position", p_afficher_position, METH_VARARGS, "afficher_position"},  {"afficher_pulsar_info", p_afficher_pulsar_info, METH_VARARGS, "afficher_pulsar_info"},  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__api()
{
  static struct PyModuleDef apimoduledef = {
      PyModuleDef_HEAD_INIT,
      "_api",
      "API module",
      -1,
      api_callback,
      NULL,
      NULL,
      NULL,
      NULL,
  };
  return PyModule_Create(&apimoduledef);
}


/*
** Load a Python module
*/

static PyObject* _import_module(const char* m)
{
  PyObject* name = PyUnicode_FromString(m);
  PyObject* module = PyImport_Import(name);
  Py_DECREF(name);
  if (module == NULL)
    if (PyErr_Occurred())
    {
      PyErr_Print();
      abort();
    }
  return module;
}

/*
** Inititialize python, register API functions,
** and load .py file
*/
static void _init_python()
{
  static wchar_t empty_string[] = L"";
  static wchar_t *argv[] = { (wchar_t *) &empty_string, NULL };

  const char* champion_path;

  champion_path = getenv("CHAMPION_PATH");
  if (champion_path == NULL)
    champion_path = ".";

  setenv("PYTHONPATH", champion_path, 1);

  static wchar_t program_name[] = L"stechec";
  Py_SetProgramName(program_name);

  PyImport_AppendInittab("_api", PyInit__api);
  Py_Initialize();
  PySys_SetArgvEx(1, argv, 0);

  champ_module = _import_module("prologin");
  py_module = _import_module("api");
}

/*
** Run a python function.
*/
static PyObject* _call_python_function(const char* name)
{
  static bool initialized = false;

  if (!initialized)
  {
    initialized = true;
    _init_python();
  }

  PyObject *arglist, *func;
  PyObject *result = NULL;

  func = PyObject_GetAttrString(champ_module, (char*)name);
  if (func && PyCallable_Check(func))
  {
    arglist = Py_BuildValue("()");
    result = PyEval_CallObject(func, arglist);
    Py_XDECREF(arglist);
    Py_DECREF(func);
  }
  if (result == NULL && PyErr_Occurred())
    PyErr_Print();

  return result;
}

/*
** Functions called from stechec to C.
*/
extern "C" void partie_init()
{
  PyObject* _retval = _call_python_function("partie_init");
  if (!_retval && PyErr_Occurred()) { PyErr_Print(); abort(); }
  try {
  Py_XDECREF(_retval);
  } catch (...) { PyErr_Print(); abort(); }
}

extern "C" void jouer_tour()
{
  PyObject* _retval = _call_python_function("jouer_tour");
  if (!_retval && PyErr_Occurred()) { PyErr_Print(); abort(); }
  try {
  Py_XDECREF(_retval);
  } catch (...) { PyErr_Print(); abort(); }
}

extern "C" void partie_fin()
{
  PyObject* _retval = _call_python_function("partie_fin");
  if (!_retval && PyErr_Occurred()) { PyErr_Print(); abort(); }
  try {
  Py_XDECREF(_retval);
  } catch (...) { PyErr_Print(); abort(); }
}

