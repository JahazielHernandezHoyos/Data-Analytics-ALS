import copy
import numpy as np

def heuristica01(jugada):
    patterns = ["XXX", "OOO"]
    h1 = 0
    for p in patterns:
        if any(jugada[i:i+3] == p for i in range(0, 9, 3)) or \
           any(jugada[i::3] == p for i in range(3)) or \
           jugada[0]+jugada[4]+jugada[8] == p or \
           jugada[2]+jugada[4]+jugada[6] == p:
            h1 = 1000 if p == "XXX" else -1000
            break
    else:
        h1 = 1 if jugada.count('_') == 0 else 0

    return h1

def generar_hijos(padre, minmax):
    return [padre[:i] + ('X' if minmax == "max" else 'O') + padre[i+1:] for i, c in enumerate(padre) if c == "_"]

def poda_alfa_beta(padre, alfa, beta, minmax):
    heuristica = heuristica01(padre)

    if abs(heuristica) > 0 or '_' not in padre:
        return alfa, beta

    hijos = generar_hijos(padre, minmax)
    heuristicas = [heuristica01(hijo) for hijo in hijos]

    for heur, hijo in sorted(zip(heuristicas, hijos), key=lambda x: abs(x[0]), reverse=True):
        alfat, betat = poda_alfa_beta(hijo, alfa, beta, "min" if minmax == "max" else "max")
        alfa, beta = (max(alfa, betat), beta) if minmax == "max" else (alfa, min(beta, alfat))

        if alfa >= beta:
            break

    return alfa, beta

def main():
    casos = [
        "_________",
        "X___O____",
        "XOX_O____",
        "XOXOO__X_",
        "_X_______",
        "____X____",
    ]

    with open("resultados_v03.txt", "w") as f:
        for caso in casos:
            alfa, beta = poda_alfa_beta(caso, -100000000, 100000000, "max" if caso.count('_') % 2 == 0 else "min")
            print(f"Board state: {caso}", file=f)
            print(f"Alfa: {alfa}", file=f)
            print(f"Beta: {beta}", file=f)
            if (alfa > 1) and (beta == 100000000):
                print("X puede ganar", file=f)
            elif (alfa < 0) and (beta == 100000000):
                print("O puede ganar", file=f)
            elif (alfa == -100000000) and (beta > 1):
                print("X puede ganar", file=f)
            elif (alfa == -100000000) and (beta < 0):
                print("O puede ganar", file=f)
            else:
                print("Pueden quedar empatados", file=f)
                print("", file=f)

if __name__ == "__main__":
    main()

