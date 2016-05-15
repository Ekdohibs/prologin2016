from api import *
from heapq import heappush, heappop, heapify
from time import time, sleep
from copy import deepcopy

carte = [[None] * TAILLE_TERRAIN for i in range(TAILLE_TERRAIN)]
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def log(*a, **kw):
    print("[%d]" % mon_id, end = " ")
    print(*a, **kw)

timed_dict = {}
    
def timed(f):
    name = f.__name__
    def wrap(*a, **kw):
        t0 = time()
        r = f(*a, **kw)
        took = time() - t0
        (total_time, calls) = timed_dict.get(name, (0., 0))
        timed_dict[name] = (total_time + took, calls + 1)
        return r
    wrap.__name__ = f.__name__
    return wrap

def timed_debut_tour():
    timed_dict.clear()

def timed_show_log():
    for name in timed_dict:
        tt, calls = timed_dict[name]
        log("%s took %fs (%d calls)" % (name, tt, calls))

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

@timed
def read_carte_plasma():
    carte = make_matrix(0.)
    for i in range(TAILLE_TERRAIN):
        for j in range(TAILLE_TERRAIN):
            carte[i][j] = charges_presentes((i, j))
    return carte


def is_tuyau(p, carte):
    return carte[p[0]][p[1]] in [case_type.TUYAU, case_type.SUPER_TUYAU]

def all_positions():
    return [(x, y) for x in range(TAILLE_TERRAIN) for y in range(TAILLE_TERRAIN)]

def all_pulsars(carte):
    return [(x, y) for (x, y) in all_positions() if carte[x][y] == case_type.PULSAR]

def all_tuyaux(carte):
    return [pos for pos in all_positions() if is_tuyau(pos, carte)]

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
def tuyaux_time_revenu(carte, dist_tuyaux):
    pos = [p for p in all_positions() if dist_tuyaux[p[0]][p[1]] != None]
    dsts = [(dist_tuyaux[p[0]][p[1]], p) for p in pos]
    dsts.sort()
    revenus = [make_matrix(0.) for _ in range(2)]
    times = make_matrix(0.)

    for d, p in dsts:
        x, y = p
        if carte[x][y] == case_type.BASE:
            revenus[proprietaire_base(p) % 2][x][y] = 1.
            times[x][y] = 0
            continue

        voisins = [(nx, ny) for (nx, ny) in adj(p) if dist_tuyaux[nx][ny] == d - 1]

        assert(len(voisins) > 0)

        for nx, ny in voisins:
            for a in range(2):
                revenus[a][x][y] += revenus[a][nx][ny]
            times[x][y] += times[nx][ny] / len(voisins)
        for a in range(2):
            revenus[a][x][y] /= len(voisins)

    return revenus, times


@timed
def revenu_moyen(carte, rev_tuyaux, carte_plasma, ttimes, plasma_value = 0):
    l = []
    ntrs = NB_TOURS - tour_actuel() + 1
    for a in range(2):
        value = 0.
        for pos in all_pulsars(carte):
            inf = info_pulsar(pos)
            if inf.pulsations_restantes == 0: continue
            r = inf.puissance * inf.pulsations_restantes
            for (x, y) in adj(pos):
                value += r * rev_tuyaux[a][x][y]
                #if ttimes[x][y] <= ntrs:
                #    value += r * rev_tuyaux[a][x][y] * (ntrs - ttimes[x][y])
        if plasma_value > 0:
            for x, y in all_tuyaux(carte):
                #if ttimes[x][y] <= ntrs:
                #    value += rev_tuyaux[a][x][y] * carte_plasma[x][y] * (ntrs - ttimes[x][y])
                value += plasma_value * carte_plasma[x][y] * rev_tuyaux[a][x][y]
        
        l.append(value)
    return l

DOUBLE_SIZE = True
SHORTER_TRADEOFF = 1/2.
EXPAND_TRADEOFF = 1. + 1./1024
POWER_TRADEOFF = 8

@timed
def joue(carte, dist_tuyaux, rev_tuyaux, carte_plasma):
    flow = tuyaux_flow(carte, dist_tuyaux, carte_plasma)
    
    dsts = [(dist_tuyaux[x][y] + 5, (x, y), (x, y)) for (x, y) in all_positions() \
            if rev_tuyaux[moi() % 2][x][y] > 0]
    orig = set(a[1] for a in dsts)
    r = make_matrix()
    org = make_matrix()
    
    heapify(dsts)

    while len(dsts) > 0:
        d, p, op = heappop(dsts)
        x, y = p
        if r[x][y] != None: continue
        r[x][y] = d
        org[x][y] = op

        if p in orig:
            for (nx, ny) in adj(p):
                if carte[nx][ny] == case_type.VIDE:
                    heappush(dsts, (d + EXPAND_TRADEOFF, (nx, ny), op))
        elif carte[x][y] == case_type.VIDE:
            for newp in adj(p):
                heappush(dsts, (d + EXPAND_TRADEOFF, newp, op))
            
    pss = []
    wpss = {}
    for pos in all_pulsars(carte):
        if r[pos[0]][pos[1]] != None:
            u = info_pulsar(pos)
            if u.pulsations_restantes > 0:
                pss.append(pos)
                wpss[pos] = u.puissance / u.periode

    for x, y in all_positions():
        if r[x][y] != None and dist_tuyaux[x][y] != None and \
           dist_tuyaux[x][y] + 5 > r[x][y] + 1e-3 and \
           flow[x][y] > 0:
              pss.append((x, y))
              wpss[(x, y)] = flow[x][y]
        elif dist_tuyaux[x][y] == None and is_tuyau((x, y), carte) \
             and r[x][y] != None:
            pss.append((x, y))
            wpss[(x, y)] = 0

    pss = list(set(p for p in pss if \
                   any(padd(p, dir) not in orig for dir in DIRS)))

    # TODO: do something
    if pss == []: return

    def h(p):
        ox, oy = org[p[0]][p[1]]
        return r[p[0]][p[1]] - r[ox][oy] * SHORTER_TRADEOFF - \
            wpss[p] * POWER_TRADEOFF

    
    best = pss[argmin(pss, h)]
    x, y = p = best
    while True:
        for newp in adj(p):
            nx, ny = newp
            if newp in orig and carte[x][y] == case_type.VIDE:
                du, dv = nx - x, ny - y
                construire(p)
                if DOUBLE_SIZE:
                    construire((x + dv, y - du))
                return
            if r[nx][ny] == r[x][y] - EXPAND_TRADEOFF and carte[nx][ny] == case_type.VIDE:
                x, y = p = newp
                break
        else:
            assert False

@timed            
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

@timed
def tuyaux_flow(carte, dist_tuyaux, carte_plasma):
    pos = [p for p in all_positions() if dist_tuyaux[p[0]][p[1]] != None]
    dsts = [(dist_tuyaux[p[0]][p[1]], p) for p in pos]
    dsts.sort(reverse = True)
    flow = make_matrix(0.)

    for pos in all_pulsars(carte):
        inf = info_pulsar(pos)
        if inf.pulsations_restantes == 0: continue
        r = inf.puissance * inf.pulsations_restantes
        for (x, y) in adj(pos):
            flow[x][y] += r
    for x, y in all_tuyaux(carte):
        flow[x][y] += carte_plasma[x][y]

    for d, p in dsts:
        x, y = p
        if carte[x][y] == case_type.BASE: continue

        voisins = [(nx, ny) for (nx, ny) in adj(p) if dist_tuyaux[nx][ny] == d - 1]

        assert(len(voisins) > 0)

        for nx, ny in voisins:
            flow[nx][ny] += flow[x][y] / len(voisins)

    return flow

@timed
def flow_directions(carte, dist_tuyaux):
    drs = make_matrix()

    for x, y in all_positions():
        d = dist_tuyaux[x][y]
        if d == None: continue
        drs[x][y] = [(nx, ny) for (nx, ny) in adj((x, y)) if dist_tuyaux[nx][ny] == d - 1]

    return drs


MAX_AT_TRIES = 50
MAX_AT_TIME = 0.3 # In seconds
AT_THRESH = 100.
AT_DELAY_TRADEOFF = 5
            
@timed
def attaque(carte, dist_tuyaux, rev_tuyaux, carte_plasma, t_times):
    flow = tuyaux_flow(carte, dist_tuyaux, carte_plasma)
    fld = flow_directions(carte, dist_tuyaux)
    rv = revenu_moyen(carte, rev_tuyaux, carte_plasma, t_times)
    rvdiff = rv[moi() % 2] - rv[adversaire() % 2]
    crvdiff = rvdiff
    best = None
    #at = all_tuyaux(carte)
    at = [pos for pos in all_positions() if carte[pos[0]][pos[1]] == case_type.TUYAU]
    at = [p for p in at if rev_tuyaux[adversaire() % 2][p[0]][p[1]] > 0]
    def wh(p):
        return -(min(p[0] - 1, TAILLE_TERRAIN - 2 - p[0], \
                     p[1] - 1, TAILLE_TERRAIN - 2 - p[1])) or -100
    at.sort(key = lambda p: (flow[p[0]][p[1]], wh(p)), reverse = True)
    at = at[:MAX_AT_TRIES]
    t0 = time()
    iterations = 0
    for p in at:
        if time() - t0 > MAX_AT_TIME:
            log("Early exit of attaque after %d iterations (%fs elapsed)" % \
                (iterations, time() - t0))
            break
        iterations += 1
        ncarte = deepcopy(carte)
        ncarte[p[0]][p[1]] = case_type.DEBRIS
        dt = distance_tuyaux(ncarte)
        rvt, tt = tuyaux_time_revenu(ncarte, dt)
        rm = revenu_moyen(ncarte, rvt, carte_plasma, tt)
        fldd = flow_directions(ncarte, dt)
        delayed = 0
        for ty in all_tuyaux(ncarte):
            if fldd[ty[0]][ty[1]] == None: continue
            w = len(fldd[ty[0]][ty[1]])
            for ntx, nty in fldd[ty[0]][ty[1]]:
                if fld[ntx][nty] != None:
                    ww = len(fld[ntx][nty])
                    for nnt in fld[ntx][nty]:
                        if ty == nnt:
                            delayed += carte_plasma[ty[0]][ty[1]] * \
                                       rev_tuyaux[adversaire() % 2][ty[0]][ty[1]] / (w * ww)
        
        rmdiff = rm[moi() % 2] - rm[adversaire() % 2]
        rmdiff += AT_DELAY_TRADEOFF * delayed

        if rmdiff > crvdiff:
            crvdiff = rmdiff
            best = p
    if crvdiff > rv[moi() % 2] / AT_THRESH and crvdiff >= CHARGE_DESTRUCTION and best != None:
        detruire(best)

MAX_RENF_TRIES = 20
MAX_RENF_TIME = 0.1 # In seconds
        
@timed
def renforce_tout(carte, dist_tuyaux, rev_tuyaux, carte_plasma, t_times):
    if points_action() == 0:
        return True
    flow = tuyaux_flow(carte, dist_tuyaux, carte_plasma)
    rv = revenu_moyen(carte, rev_tuyaux, carte_plasma, t_times)
    rvdiff = rv[moi() % 2] - rv[adversaire() % 2]
    crvdiff = rvdiff
    worst = None
    at = all_tuyaux(carte)
    at = [p for p in at if rev_tuyaux[moi() % 2][p[0]][p[1]] > 0]
    def wh(p):
        return -(min(p[0] - 1, TAILLE_TERRAIN - 2 - p[0], \
                     p[1] - 1, TAILLE_TERRAIN - 2 - p[1]))
    at.sort(key = lambda p: (flow[p[0]][p[1]], wh(p)), reverse = True)
    at = at[:MAX_RENF_TRIES]
    t0 = time()
    iterations = 0
    for p in at:
        if time() - t0 > MAX_RENF_TIME:
            log("Early exit of renforce_tout after %d iterations (%fs elapsed)" % \
                (iterations, time() - t0))
            break
        
        iterations += 1
        ncarte = deepcopy(carte)
        ncarte[p[0]][p[1]] = case_type.DEBRIS
        dt = distance_tuyaux(ncarte)
        rvt, tt = tuyaux_time_revenu(ncarte, dt)
        rm = revenu_moyen(ncarte, rvt, carte_plasma, tt)
        rmdiff = rm[moi() % 2] - rm[adversaire() % 2]
        if rmdiff < crvdiff:
            crvdiff = rmdiff
            worst = p
    if worst != None:
        renforce(worst)
        return False
    else:
        return True

        
mon_id = None
# Fonction appelée au début de la partie.
def partie_init():
    global mon_id
    mon_id = moi()

def renforce(p):
    l = [(x, y) for x in range(p[0] - 1, p[0] + 2) \
         for y in range(p[1] - 1, p[1] + 2) if valide((x, y)) and \
         est_libre((x, y))]
    if l == []:
        return
    i = argmin(l, lambda pp: -sum(est_tuyau(r) for r in adj(pp)))
    log("Renforce", l[i])
    construire(l[i])

MAX_WD_TRIES = 30
MAX_WD_TIME = 0.2 # In seconds
WD_TRADEOFF = 10.
WD_PLASMA = 1

@timed
def was_destroyed(dpos):
    if points_action() == 0:
        return
    carte = read_carte()
    ncarte = deepcopy(carte)
    ncarte[dpos[0]][dpos[1]] = case_type.TUYAU
    carte_plasma = read_carte_plasma()
    dist_tuyaux = distance_tuyaux(ncarte)
    rev_tuyaux, t_times = tuyaux_time_revenu(ncarte, dist_tuyaux)

    flow = tuyaux_flow(ncarte, dist_tuyaux, carte_plasma)
    rv = revenu_moyen(ncarte, rev_tuyaux, carte_plasma, t_times, WD_PLASMA)

    rvdiff = rv[moi() % 2] - rv[adversaire() % 2]
    crvdiff = rvdiff

    at = all_tuyaux(carte)
    at = [p for p in at if rev_tuyaux[moi() % 2][p[0]][p[1]] > 0]
    def wh(p):
        return -(min(p[0] - 1, TAILLE_TERRAIN - 2 - p[0], \
                     p[1] - 1, TAILLE_TERRAIN - 2 - p[1]))
    at.sort(key = lambda p: (flow[p[0]][p[1]], wh(p)), reverse = True)
    at = at[:MAX_RENF_TRIES]
    t0 = time()
    iterations = 0
    for p in at:
        if time() - t0 > MAX_WD_TIME:
            log("Early exit of was_destroyed after %d iterations (%fs elapsed)" % \
                (iterations, time() - t0))
            break
        
        iterations += 1
        ncarte = deepcopy(carte)
        ncarte[p[0]][p[1]] = case_type.DEBRIS
        dt = distance_tuyaux(ncarte)
        rvt, tt = tuyaux_time_revenu(ncarte, dt)
        rm = revenu_moyen(ncarte, rvt, carte_plasma, tt, WD_PLASMA)
        rmdiff = rm[moi() % 2] - rm[adversaire() % 2]
        if rmdiff < crvdiff:
            crvdiff = rmdiff

    print(rv, rvdiff, crvdiff)
    ddiff = rvdiff - crvdiff
    if ddiff < rv[moi() % 2] / WD_TRADEOFF:
        return

    deblayer(dpos)
    construire(dpos)
    renforce(dpos)

    
# Fonction appelée à chaque tour.
def jouer_tour():
    log("Tour %d" % tour_actuel())
    timed_debut_tour()
    
    # Recontruire les tuyaux détruits par l'adversaire
    for p in hist_tuyaux_detruits():
        was_destroyed(p)

    carte_plasma = read_carte_plasma()

    if points_action() >= COUT_DESTRUCTION and CHARGE_DESTRUCTION <= score(moi()):
        carte = read_carte()
        dist_tuyaux = distance_tuyaux(carte)
        rev_tuyaux, t_times = tuyaux_time_revenu(carte, dist_tuyaux)
        attaque(carte, dist_tuyaux, rev_tuyaux, carte_plasma, t_times)
        
    for i in range(4):
        if points_action() == 0: break
        carte = read_carte()
        carte_plasma = read_carte_plasma()
        dist_tuyaux = distance_tuyaux(carte)
        rev_tuyaux = tuyaux_revenu(carte, dist_tuyaux)
        joue(carte, dist_tuyaux, rev_tuyaux, carte_plasma)

    for i in range(4):
        if points_action() == 0: break
        carte = read_carte()
        dist_tuyaux = distance_tuyaux(carte)
        rev_tuyaux, t_times = tuyaux_time_revenu(carte, dist_tuyaux)
        if renforce_tout(carte, dist_tuyaux, rev_tuyaux, carte_plasma, t_times):
            break

    augmente_aspiration()

    timed_show_log()
        
# Fonction appelée à la fin de la partie.
def partie_fin():
    pass

