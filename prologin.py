from api import *
from heapq import heappush, heappop, heapify
from time import time

carte = [[None] * TAILLE_TERRAIN for i in range(TAILLE_TERRAIN)]
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def timed(f):
    def wrap(*a, **kw):
        t0 = time()
        r = f(*a, **kw)
        took = time() - t0
        print("[%d] %s took %fs" % (mon_id, f.__name__, took))
        return r
    wrap.__name__ = f.__name__
    return wrap

def padd(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def valide(p):
    return 0 <= p[0] < TAILLE_TERRAIN and 0 <= p[1] < TAILLE_TERRAIN

@timed
def read_carte():
    carte = make_matrix()
    for i in range(TAILLE_TERRAIN):
        for j in range(TAILLE_TERRAIN):
            carte[i][j] = type_case((i, j))
    return carte

def is_tuyau(p, carte):
    return carte[p[0]][p[1]] in [case_type.TUYAU, case_type.SUPER_TUYAU]

def all_positions():
    return [(x, y) for x in range(TAILLE_TERRAIN) for y in range(TAILLE_TERRAIN)]

def all_pulsars(carte):
    return [(x, y) for (x, y) in all_positions() if carte[x][y] == case_type.PULSAR]

def make_matrix(elem = None):
    return [[elem] * TAILLE_TERRAIN for _ in range(TAILLE_TERRAIN)]

def adj(p):
    l = [padd(p, dir) for dir in DIRS]
    return [x for x in l if valide(x)]

def argmin(l, f = lambda x: x):
    best = f(l[0])
    ibest = 0
    for i in range(1, len(l)):
        r = f(l[i])
        if r < best:
            best = r
            ibest = i
    return ibest


dist_tuyaux = None

@timed
def distance_tuyaux(carte):
    distance = make_matrix()
    tas = []

    for p in all_positions():
        if carte[p[0]][p[1]] == case_type.BASE:
            tas.append((-puissance_aspiration(p), p))
    heapify(tas)

    while len(tas) > 0:
        d, p = heappop(tas)
        x, y = p
        if distance[x][y] != None: continue
        distance[x][y] = d
        for newp in adj(p):
            if is_tuyau(newp, carte):
                heappush(tas, (d + 1, newp))

    return distance

rev_tuyaux = None

@timed
def tuyaux_revenu(carte, dist_tuyaux):
    pos = [p for p in all_positions() if dist_tuyaux[p[0]][p[1]] != None]
    dsts = [(dist_tuyaux[p[0]][p[1]], p) for p in pos]
    dsts.sort()
    revenus = [make_matrix(0.) for _ in range(2)]

    for d, p in dsts:
        x, y = p
        if carte[x][y] == case_type.BASE:
            revenus[proprietaire_base(p) % 2][x][y] = 1.
            continue

        voisins = [(nx, ny) for (nx, ny) in adj(p) if dist_tuyaux[nx][ny] == d - 1]

        assert(len(voisins) > 0)

        for nx, ny in voisins:
            for a in range(2):
                revenus[a][x][y] += revenus[a][nx][ny]
        for a in range(2):
            revenus[a][x][y] /= len(voisins)

    return revenus

@timed
def joue(carte, dist_tuyaux, rev_tuyaux):
    dsts = [(dist_tuyaux[x][y] / 2., (x, y)) for (x, y) in all_positions() \
            if rev_tuyaux[moi() % 2][x][y] > 0]
    orig = set(a[1] for a in dsts)
    r = make_matrix()
    heapify(dsts)

    while len(dsts) > 0:
        d, p = heappop(dsts)
        x, y = p
        if r[x][y] != None: continue
        r[x][y] = d

        if p in orig:
            for (nx, ny) in adj(p):
                if carte[nx][ny] == case_type.VIDE:
                    heappush(dsts, (d + 1, (nx, ny)))
        elif carte[x][y] == case_type.VIDE:
            for newp in adj(p):
                heappush(dsts, (d + 1, newp))
            
    
    pss = [pos for pos in all_pulsars(carte) if r[pos[0]][pos[1]] != None \
           and info_pulsar(pos).pulsations_restantes > 0 \
    ]
    pss += [(x, y) for (x, y) in all_positions() if \
            r[x][y] != None and dist_tuyaux[x][y] != None and \
            rev_tuyaux[moi() % 2][x][y] == 0 and \
            rev_tuyaux[adversaire() % 2][x][y] > 0 and \
            dist_tuyaux[x][y] >= 2 * r[x][y]]
    pss += [(x, y) for (x, y) in all_positions() if \
            dist_tuyaux[x][y] == None and is_tuyau((x, y), carte)
    ]

    pss = list(set(p for p in pss if \
                   any(padd(p, dir) not in orig for dir in DIRS)))

    # TODO: do something
    if pss == []: return

    closest = pss[argmin(pss, lambda p: r[p[0]][p[1]])]
    x, y = p = closest
    while True:
        for newp in adj(p):
            nx, ny = newp
            if newp in orig and carte[x][y] == case_type.VIDE:
                construire(p)
                return
            if r[nx][ny] == r[x][y] - 1 and carte[nx][ny] == case_type.VIDE:
                x, y = p = newp
                break
        else:
            assert False

def augmente_aspiration():
    bases = ma_base()
    is_used = [False] * len(bases)
    for i, pos in enumerate(bases):
        for pp in adj(pos):
            if est_tuyau(pp):
                is_used[i] = True
                break
    for i in range(len(bases)):
        if not is_used[i]: continue
        if puissance_aspiration(bases[i]) == LIMITE_ASPIRATION:
            continue
        for j in range(len(bases)):
            if is_used[j]: continue
            if puissance_aspiration(bases[j]) == 0: continue
            deplacer_aspiration(bases[j], bases[i])
            
mon_id = None
# Fonction appelée au début de la partie.
def partie_init():
    global mon_id
    mon_id = moi()

# Fonction appelée à chaque tour.
def jouer_tour():
    
    # Recontruire les tuyaux détruits par l'adversaire
    for p in hist_tuyaux_detruits():
        deplayer(p)
        contruire(p)
    
    for i in range(4):
        carte = read_carte()
        dist_tuyaux = distance_tuyaux(carte)
        rev_tuyaux = tuyaux_revenu(carte, dist_tuyaux)
        joue(carte, dist_tuyaux, rev_tuyaux)

    augmente_aspiration()
        
# Fonction appelée à la fin de la partie.
def partie_fin():
    pass

