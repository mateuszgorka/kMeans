import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def load_data(filename):
    data = pd.read_csv(filename, header=None)
    return data.iloc[:, :-1].values, data.iloc[:, -1].values  # --> tuuuutaj cechy + etykiety klas!


MAX_ITERS = 100
TOLERANCE = 1e-4


def main():
    filename = "data/iris.data"
    data, labels_orig = load_data(filename)

    k = int(input("Podaj k: "))
    labels, centroids = k_means(data, k, MAX_ITERS, TOLERANCE)

    print("\nSkład grup:")
    for i in range(k):
        cluster_points = [(data[j], labels_orig[j]) for j in range(len(data)) if labels[j] == i]
        print(f"\nKlaster {i} ({len(cluster_points)} elementów):")
        for point, name in cluster_points:
            print(f"  {point} -> {name}")


    visualize(data, labels, centroids)


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

    centroids = data[:k]  # ->>> Wybieramy pierwsze k punktów jako poczatkowe centroidy
    labels = [0] * len(data)

    for iter in range(max_iter):
        clusters = [[] for _ in range(k)]       # pusciutka lista dla kazdego z k klastrow
        new_labels = []                         # a tutaj numerow klastrów
        
        for p in data:
            distance = [dist(p, centroid) for centroid in centroids]   # czyli wywolujemy distance od danegoo (p) punktu i centroidu
            closest_ones = distance.index(min(distance))
            clusters[closest_ones].append(p)                           # dodaaaaaaaajemy punkty do klastra z najblizszym indexem
            new_labels.append(closest_ones)                            # a tu mega luz bo same indexy chcemy


        # teraz bedziemy obliczac nowe centroidy
        # srednia arytmetyczna punktow w klastrze
        # ale nie mozemy tego zrobic w prosty hardcodeowy sposob

       
        centroid_changed = []
        for cl in clusters:
            if len(cl) == 0:
                raise ValueError("Pusty klaster")

            centroid_changed.append([sum(dim) / len(cl) for dim in zip(*cl)])

            # zip dziala w taki sposob ze gdy mamy
            # cl = [(1, 2), (3, 4), (5, 6)] to automatyczie dostajemy podzbiory (1,2,3)(2,4,6)
            # dla dwoch prosto mo mozna zrobuc meanX i meanY ale dla wielu blizej nieokreslonych najlepsza opcja
            # w tym momencie pi prostu sumujemy wartosci ktore wyciagamy z arraya zip(ktore wyciaga cl czyli te wektory)


        
        change_iteration = [dist(centroids[i], centroid_changed[i]) for i in range(k)]  ### lista z dystansami od pierwotnego centroidu do nowego centroidu [|\]
                                                                                        # czyli mamy nie wiem (1,3,6) i pierwotne (1,3)
        # Aktualizacje
        # ------------

        centroids = centroid_changed
        labels = new_labels

        # -----------

        total_distance = sum(dist(data[j], centroids[labels[j]]) for j in range(len(data)))   #### caaalkowity dystans do przypisanych dla nich centroidow

        print(f"Iteracja {iter + 1}: {total_distance:.2f}")  

        if max(change_iteration) < tolerance:  
            break

    return labels, centroids


def dist(p, q):
    ### odleglosc euklidesowa

    return math.sqrt(sum((px - qx) ** 2 for px, qx in zip(p, q)))


def visualize(data, labels, centroids, it=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colormaps = ['b', 'g', 'r']

    for i in range(len(centroids)):
        clusters = [data[j] for j in range(len(data)) if labels[j] == i]
        clusters = np.array(clusters)


        ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], color=colormaps[i % len(colormaps)],
                   label=f'Klaster {i}')


    centroids = np.array(centroids)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], color='black', marker='X', s=100, label='Centroidy')


    ax.set_xlabel('Cecha 1')
    ax.set_ylabel('Cecha 2')
    ax.set_zlabel('Cecha 3')


    ax.set_title('Wizualizacja centroidów i przypisanych punktów')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
