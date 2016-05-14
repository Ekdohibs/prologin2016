from api import *
from heapq import heappush, heappop, heapify


carte = [[None] * TAILLE_TERRAIN for i in range(TAILLE_TERRAIN)]
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def padd(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def valide(p):
    return 0 <= p[0] < TAILLE_TERRAIN and 0 <= p[1] < TAILLE_TERRAIN

def read_carte():
    for i in range(TAILLE_TERRAIN):
        for j in range(TAILLE_TERRAIN):
            carte[i][j] = type_case((i, j))

dist_tuyaux = None
def distance_tuyaux():
    global dist_tuyaux
    distance = [[None] * TAILLE_TERRAIN for i in range(TAILLE_TERRAIN)]
    tas = []
    for x in range(TAILLE_TERRAIN):
        for y in range(TAILLE_TERRAIN):
            if carte[x][y] == case_type.BASE:
                tas.append((-puissance_aspiration((x, y)), (x, y)))
    heapify(tas)
    while len(tas) > 0:
        d, p = heappop(tas)
        x, y = p
        if distance[x][y] != None: continue
        distance[x][y] = d
        for dir in DIRS:
            nx, ny = newp = padd(p, dir)
            if valide(newp) and carte[nx][ny] in [case_type.TUYAU, case_type.SUPER_TUYAU]:
                heappush(tas, (d + 1, newp))
    dist_tuyaux = distance
    return distance

rev_tuyaux = None
def tuyaux_revenu():
    global rev_tuyaux
    pos = [(x, y) for x in range(TAILLE_TERRAIN) for y in range(TAILLE_TERRAIN) if dist_tuyaux[x][y] != None]
    dsts = [(dist_tuyaux[x][y], (x, y)) for (x, y) in pos]
    dsts.sort()
    revenus = [[[0.] * TAILLE_TERRAIN for i in range(TAILLE_TERRAIN)] for _ in range(2)]
    for d, p in dsts:
        x, y = p
        if carte[x][y] == case_type.BASE:
            revenus[proprietaire_base((x, y)) % 2][x][y] = 1.
            continue
        voisins = []
        for dir in DIRS:
            nx, ny = newp = padd(p, dir)
            if valide(newp) and dist_tuyaux[nx][ny] == d - 1:
                voisins.append((nx, ny))
        assert(len(voisins) > 0)
        for nx, ny in voisins:
            for a in range(2):
                revenus[a][x][y] += revenus[a][nx][ny]
        for a in range(2):
            revenus[a][x][y] /= len(voisins)
    rev_tuyaux = revenus
    return revenus

def argmin(l, f=lambda x: x):
    best = f(l[0])
    ibest = 0
    for i in range(1, len(l)):
        r = f(l[i])
        if r < best:
            best = r
            ibest = i
    return ibest

def joue():
    dsts = [(dist_tuyaux[x][y] / 2., (x, y)) for x in range(TAILLE_TERRAIN) for y in range(TAILLE_TERRAIN) if rev_tuyaux[moi() % 2][x][y] > 0]
    orig = set(a[1] for a in dsts)
    r = [[None] * TAILLE_TERRAIN for i in range(TAILLE_TERRAIN)]
    heapify(dsts)
    while len(dsts) > 0:
        d, p = heappop(dsts)
        x, y = p
        if r[x][y] != None: continue
        r[x][y] = d
        if carte[x][y] == case_type.VIDE or p in orig:
            for dir in DIRS:
                nx, ny = newp = padd(p, dir)
                if valide(newp):
                    heappush(dsts, (d + 1, newp))
    pss = [pos for pos in liste_pulsars() if r[pos[0]][pos[1]] != None]
    pss += [(x, y) for x in range(TAILLE_TERRAIN) for y in range(TAILLE_TERRAIN) if r[x][y] != None and dist_tuyaux[x][y] != None and rev_tuyaux[moi() % 2][x][y] == 0 and rev_tuyaux[adversaire() % 2][x][y] > 0 and dist_tuyaux[x][y] >= 2 * r[x][y]]
    pss = list(set(p for p in pss if all(padd(p, dir) not in orig for dir in DIRS)))
    if pss == []: return
    closest_pulsar = pss[argmin(pss, lambda p: r[p[0]][p[1]])]
    x, y = p = closest_pulsar
    #while r[x][y] > 1:
    while True:
        for dir in DIRS:
            nx, ny = newp = padd(p, dir)
            if newp in orig:
                #print(moi(), "CONSTRUIRE", p)
                construire(p)
                return
            if valide(newp) and r[nx][ny] == r[x][y] - 1 and carte[nx][ny] == case_type.VIDE:
                x, y = p = newp
                break
    construire(p)
                
    

# Fonction appelée au début de la partie.
def partie_init():
    pass

# Fonction appelée à chaque tour.
def jouer_tour():
    for p in hist_tuyaux_detruits():
        deplayer(p)
        contruire(p)
    for i in range(4):
        read_carte()
        distance_tuyaux()
        tuyaux_revenu()
        joue()

# Fonction appelée à la fin de la partie.
def partie_fin():
    pass
