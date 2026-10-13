# Tracking de personas en una tienda con YOLO y ByteTrack

Reto de tracking: sobre un video de la entrada de un local, el programa detecta y sigue personas con
YOLO + ByteTrack y dibuja tres cosas:

1. **Estela de movimiento:** el recorrido del centro de cada persona, que se desvanece con el tiempo.
2. **Mapa de calor:** acumula las posiciones de todas las personas y muestra las zonas de mayor tránsito.
3. **Contador de entradas y salidas:** una línea virtual cuenta quién la cruza. Cada ID se cuenta una sola vez.

## Resultado

- Video procesado: [salidas/store_resultado.mp4](salidas/store_resultado.mp4)
- Mapa de calor final: [salidas/heatmap_final.png](salidas/heatmap_final.png)

## Instalación (WSL / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Uso

```bash
python tracking_tienda.py                      # video completo (unos 8 minutos en CPU)
python tracking_tienda.py --max-frames 600     # prueba rápida
python tracking_tienda.py --linea-y 450        # altura de la línea de entrada
```

Opciones: `--video` (por defecto `videos/store.mp4`), `--salida`, `--linea-y`, `--max-frames`.

## Cómo funciona

- **Detección y seguimiento:** `model.track(..., classes=[0], tracker="bytetrack.yaml")` devuelve cada persona con un ID estable entre cuadros.
- **Filtro de falsos positivos:** un ID cuenta como persona solo después de moverse 60 píxeles. Así se descartan objetos fijos que YOLO confunde con personas (por ejemplo un cartel).
- **Estela:** se guardan los últimos 45 cuadros de cada ID y los trazos más viejos se dibujan más transparentes.
- **Mapa de calor:** cada centro suma 1 en una grilla 4 veces más pequeña que el video. Se difumina, se pasa a escala logarítmica (para que una zona muy transitada no oculte el resto del recorrido), se colorea y se superpone.
- **Contador:** la línea tiene una zona muerta de 12 píxeles. Un cruce de arriba hacia abajo es una entrada y de abajo hacia arriba es una salida.
- **Compresión:** al terminar, el video se re-codifica a H.264 para que pese unos 30 MB.
