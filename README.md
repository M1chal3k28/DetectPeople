# People Detection API

Ten projekt to API oparte na Flask, które wykorzystuje model YOLOv8 do wykrywania osób w obrazie z kamery.

## Wymagania

Utworz w folderze repo python venv

```bash
py -m venv .
cd Scripts/
activate
```

Przed uruchomieniem projektu upewnij się, że masz zainstalowane potrzebne biblioteki:

```bash
pip3 install -r requirements.txt
```

Dodatkowo, do przyspieszenia obliczeń zalecane jest posiadanie GPU oraz zainstalowanego CUDA.

## Uruchomienie

Skrypt można uruchomić w dwóch trybach:

### Tryb debugowania (dla testów)

```bash
python script.py --deploy 0
```

Dostępne endpointy:
- `GET /detect` - zwraca JSON z informacją, czy osoba jest wykryta
- `GET /detect_image` - zwraca obraz z oznaczoną wykrytą osobą (tylko w trybie debugowania)

### Tryb produkcyjny

```bash
python script.py --deploy 1
```

### Flaga restart

- Automatyczne restartowanie API podczas problemów z kamerką
    ```bash
    python script.py --restart 1
    ```


Serwer działa na `127.0.0.1:8080` przy użyciu Waitress.

## Opis funkcji

- **`detect_person()`** - Sprawdza, czy w obrazie z kamery znajdują sie ludzie, zwraca także ich ilość.
- **`detect_person_image()`** - Dodatkowo rysuje ramki wokół wykrytych osób.
- **`getImg()`** - Pobiera obraz z kamery.

## Logi

- API loguje na poziomie:
    - Debug
    - Error
    - Warning
    - Info
- Do pliku **`logs`**, który stworzy się przy pierwszym uruchomieniu

## Uwagi

- Skrypt używa najmniejszego modelu YOLOv8 (`yolov8n.pt`) dla szybszych obliczeń.
- Kamera jest ładowana z domyślnego urządzenia (`cv2.VideoCapture(0)`). Jeśli masz wiele kamer, może być konieczna zmiana indeksu.
- Endpoint `/detect_image` działa tylko w trybie debugowania.

## Licencja

Projekt na licencji MIT.

