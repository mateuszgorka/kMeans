import math
import pandas as pd

# Pakiet: kmeans (nazwa symboliczna – nie ma pakietów w Pythonie w ten sam sposób)

# Wczytanie danych z pliku iris.data
def load_data(filename):
    """Wczytuje dane z pliku iris.data, uwzględniając etykiety klas."""
    data = pd.read_csv(filename, header=None)
    return data.iloc[:, :-1].values, data.iloc[:, -1].values  # Pobieramy cechy + etykiety klas

# Stałe do testów
MAX_ITERS = 100
TOLERANCE = 1e-4


def main():
    filename = "data/iris.data"
    data, labels_orig = load_data(filename)  # Pobranie danych i etykiet

    k = int(input("Podaj liczbę klastrów (k): "))  # Pobranie wartości k od użytkownika
    labels, centroids = k_means(data, k, MAX_ITERS, TOLERANCE)

    print("\nSkład grup:")
    for i in range(k):
        cluster_points = [(data[j], labels_orig[j]) for j in range(len(data)) if labels[j] == i]
        print(f"\nKlaster {i} ({len(cluster_points)} elementów):")
        for point, name in cluster_points:
            print(f"  {point} -> {name}")


def k_means(data, k, max_iter, tolerance):
    """
    Zadanie: ***zaimplementuj*** algorytm k-średnich.

    Wymagania:
      1. Inicjalizuj centroidy pierwszymi `k` punktami z wejścia.
      2. Przypisz każdy punkt do najbliższego centroidu (odległość euklidesowa).
      3. Oblicz nowe centroidy jako średnią arytmetyczną punktów w klastrze.
      4. Powtarzaj kroki 2-3, aż klastry przestaną się zmieniać, centroidy
         przesuną się mniej niż `tolerance`, lub osiągnięto `maxIter` iteracji.
      5. Zwróć tablicę etykiet (wartości 0…k-1) - labels[i] to numer
         klastra, do którego został przypisany data[i].
    """

    centroids = data[:k]  
    labels = [0] * len(data)

    for iter in range(max_iter):
        clusters = [[] for _ in range(k)] 
        new_labels = []  
        
        for p in data:
            distance = [dist(p, centroid) for centroid in centroids]  
            closest_ones = distance.index(min(distance))
            clusters[closest_ones].append(p)  
            new_labels.append(closest_ones)  

       
        centroid_changed = []
        for cl in clusters:
            if len(cl) == 0:
                raise ValueError("Pusty klaster")

            centroid_changed.append([sum(dim) / len(cl) for dim in zip(*cl)])

        
        change_iteration = [dist(centroids[i], centroid_changed[i]) for i in range(k)]
        
        centroids = centroid_changed
        labels = new_labels

        total_distance = sum(dist(data[j], centroids[labels[j]]) for j in range(len(data)))

        print(f"Iteracja {iter + 1}: {total_distance:.2f}")  

        if max(change_iteration) < tolerance:  
            break

    return labels, centroids


def dist(p, q):
    """Odległość euklidesowa w N-wymiarach."""
    return math.sqrt(sum((px - qx) ** 2 for px, qx in zip(p, q)))


if __name__ == "__main__":
    main()
