import math

# Pakiet: kmeans (nazwa symboliczna – nie ma pakietów w Pythonie w ten sam sposób)

# Stałe dane wejściowe
DATASET = [
    [1.0, 1.0], [5.0, 5.0], [8.0, 1.0],
    [0.9, 1.1], [5.1, 4.9], [8.1, 1.1],
    [1.1, 0.9], [4.9, 5.1], [7.9, 0.9],
    [1.2, 1.0], [5.2, 5.0], [8.2, 1.0]
]

# Stałe do testów
K_VALUES = [2, 3, 4]
MAX_ITERS = 100
TOLERANCE = 1e-4


def main():
    for k in K_VALUES:
        labels = k_means(DATASET, k, MAX_ITERS, TOLERANCE)
        self_check(labels, k)
    print("Gratulacje – wszystkie testy zakończyły się sukcesem!")


def k_means(data, k, max_iter, tolerance):
    """
    Zadanie: ***zaimplementuj*** algorytm k-średnich.

    Wymagania:
      1. Inicjalizuj centroidy pierwszymi `k` punktami z wejścia.
      2. Przypisz każdy punkt do najbliższego centroidu (odległość euklidesowa).
      3. Oblicz nowe centroidy jako średnią arytmetyczną punktów w klastrze.
      4. Powtarzaj kroki 2-3, aż klastry przestaną się zmieniać, centroidy
         przesuną się mniej niż `tolerance`, lub osiągnięto `maxIter` iteracji.
      5. Zwróć tablicę etykiet (wartości 0…k-1) – labels[i] to numer
         klastra, do którego został przypisany data[i].

    Możesz – ale nie musisz – dopisać metody pomocnicze w tej klasie.
    Nie modyfikuj sygnatury ani nazwy tej metody. Możesz korzystać z publicznej metody dist().
    """
    # ---------------------------------------------------------------------------------
    # TODO: Twoja implementacja tutaj
    # ---------------------------------------------------------------------------------
    return None


def dist(p, q):
    """Odległość euklidesowa w 2D."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return math.sqrt(dx * dx + dy * dy)


def self_check(labels, k):
    assert labels is not None, f"Metoda k_means zwróciła None (k={k})"
    assert len(labels) == len(DATASET), f"Nieprawidłowa długość tablicy etykiet (k={k})"

    if k == 2:
        expected = [
            [1.0500, 1.0000],
            [6.5500, 3.0000]
        ]
    elif k == 3:
        expected = [
            [1.0500, 1.0000],
            [5.0500, 5.0000],
            [8.0500, 1.0000]
        ]
    elif k == 4:
        expected = [
            [1.1000, 0.9667],
            [5.0500, 5.0000],
            [8.0500, 1.0000],
            [0.9000, 1.1000]
        ]
    else:
        raise ValueError(f"Brak zdefiniowanego testu dla k={k}")

    found = centroids_from_labels(labels, k)

    if not matches(expected, found):
        raise ValueError(f"Wynik algorytmu odbiega od oczekiwanego – sprawdź implementację (k={k}).")


def centroids_from_labels(labels, k):
    centroids = [[0.0, 0.0] for _ in range(k)]
    counts = [0] * k
    for i in range(len(DATASET)):
        c = labels[i]
        if c < 0 or c >= k:
            raise ValueError(f"Etykieta spoza zakresu 0…{k - 1}")
        counts[c] += 1
        centroids[c][0] += DATASET[i][0]
        centroids[c][1] += DATASET[i][1]

    for c in range(k):
        if counts[c] == 0:
            raise ValueError(f"Pusty klaster (k={k}, klaster={c}) – upewnij się, że inicjalizacja jest poprawna.")
        centroids[c][0] /= counts[c]
        centroids[c][1] /= counts[c]
    return centroids


def matches(exp, found):
    if len(exp) != len(found):
        return False
    used = [False] * len(found)
    for e in exp:
        ok = False
        for i in range(len(found)):
            if not used[i] and dist(e, found[i]) < 0.01:
                used[i] = True
                ok = True
                break
        if not ok:
            return False
    return True


if __name__ == "__main__":
    main()
