"""Reto de tracking: estela de movimiento, mapa de calor y contador de entradas.

Usa YOLO + ByteTrack sobre un video de una entrada y dibuja:
  1. Estela (trayectoria) de cada persona, que se desvanece con el tiempo
  2. Mapa de calor acumulado de las posiciones
  3. Contador de personas que cruzan una linea virtual (cada ID se cuenta una vez)

Uso (en WSL, con el entorno activado):
    python tracking_tienda.py
    python tracking_tienda.py --max-frames 600     # prueba rapida
"""
import argparse
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

MODELO = "yolo26n.pt"    # YOLO pre-entrenado (clase 0 = persona)
CONF_MIN = 0.3           # confianza minima de una deteccion
FRAMES_ESTELA = 45       # cuanto dura una estela (frames) antes de desvanecerse
ESCALA_CALOR = 4         # la grilla del mapa de calor es 4 veces mas chica que el video
ALFA_CALOR = 0.65        # opacidad maxima del mapa de calor
MARGEN_LINEA = 12        # pixeles de zona muerta alrededor de la linea (evita rebotes)
MIN_DESPLAZAMIENTO = 60  # pixeles que un ID debe moverse para considerarlo persona real
CADA_N_CALOR = 10        # cada cuantos frames se recalcula la imagen del mapa de calor

AMARILLO, BLANCO = (0, 220, 255), (255, 255, 255)


def leer_argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--video", default="videos/store.mp4")
    p.add_argument("--salida", default="salidas/store_resultado.mp4")
    p.add_argument("--linea-y", type=int, default=450, help="altura (y) de la linea de entrada")
    p.add_argument("--max-frames", type=int, default=0, help="0 = todo el video")
    return p.parse_args()


def comprimir(origen, destino):
    """Re-codifica a H.264 para que el video pese mucho menos (limite de GitHub: 100 MB)."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", origen, "-c:v", "libx264",
                    "-crf", "28", "-preset", "fast", "-pix_fmt", "yuv420p", destino], check=True)


def color_de(id_):
    """Un color distinto y estable para cada ID."""
    hsv = np.uint8([[[(id_ * 47) % 180, 255, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def dibujar_estelas(frame, estelas, n):
    """Dibuja la trayectoria de cada ID; los puntos mas viejos se ven mas transparentes."""
    capa = np.zeros_like(frame)                       # colores de los trazos
    alfa = np.zeros(frame.shape[:2], np.uint8)        # opacidad de cada pixel de trazo
    for id_ in list(estelas):
        puntos = estelas[id_]
        while puntos and n - puntos[0][2] > FRAMES_ESTELA:
            puntos.popleft()                          # olvida los puntos muy viejos
        if not puntos:
            del estelas[id_]
            continue
        color = color_de(id_)
        lista = list(puntos)
        for (xa, ya, _), (xb, yb, nb) in zip(lista, lista[1:]):
            fuerza = 1 - (n - nb) / FRAMES_ESTELA     # 1 = reciente, 0 = a punto de desaparecer
            cv2.line(capa, (xa, ya), (xb, yb), color, 3)
            cv2.line(alfa, (xa, ya), (xb, yb), int(255 * fuerza), 3)
    mascara = alfa > 0
    a = (alfa[mascara].astype(np.float32) / 255)[:, None]
    frame[mascara] = (frame[mascara] * (1 - a) + capa[mascara] * a).astype(np.uint8)


def sumar_calor(calor, cx, cy):
    """Suma 1 en la celda de la grilla donde esta el centro de la persona."""
    fila = min(cy // ESCALA_CALOR, calor.shape[0] - 1)
    col = min(cx // ESCALA_CALOR, calor.shape[1] - 1)
    calor[fila, col] += 1


def crear_capa_calor(calor, tamano):
    """Convierte la grilla acumulada en (imagen de color, opacidad) del tamano del video."""
    suave = cv2.GaussianBlur(calor, (0, 0), 3.5)      # difumina las celdas
    positivos = suave[suave > 0]
    tope = np.percentile(positivos, 99) if positivos.size else 1.0
    # escala logaritmica: una zona muy caliente no apaga al resto del recorrido
    norm = np.clip(np.log1p(suave) / np.log1p(max(tope, 1e-6)), 0, 1)
    norm = cv2.resize(norm, tamano, interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alfa = (norm * ALFA_CALOR)[..., None]             # zonas frias casi transparentes
    return color.astype(np.float32), alfa.astype(np.float32)


def aplicar_calor(frame, color, alfa):
    frame[:] = (frame * (1 - alfa) + color * alfa).astype(np.uint8)


def zona(cy, linea_y):
    """'arriba' o 'abajo' de la linea; None si esta dentro de la zona muerta."""
    if cy < linea_y - MARGEN_LINEA:
        return "arriba"
    if cy > linea_y + MARGEN_LINEA:
        return "abajo"
    return None


def main():
    args = leer_argumentos()
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(args.video)
    ancho, alto = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    if args.max_frames:
        total = min(total, args.max_frames)

    modelo = YOLO(MODELO)
    temporal = str(Path(args.salida).with_suffix(".tmp.mp4"))
    escritor = cv2.VideoWriter(temporal, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))

    estelas = defaultdict(deque)                       # id -> puntos (x, y, n_frame)
    calor = np.zeros((alto // ESCALA_CALOR, ancho // ESCALA_CALOR), np.float32)
    capa_calor = None                                  # (color, opacidad) en cache
    zona_previa = {}                                   # id -> ultima zona definida
    entradas, salidas = set(), set()                   # IDs ya contados (una sola vez)
    origen, confirmados = {}, set()                    # filtro de falsos positivos fijos
    ids_vistos = set()
    fondo = None

    resultados = modelo.track(source=args.video, stream=True, persist=True, classes=[0],
                              conf=CONF_MIN, tracker="bytetrack.yaml", verbose=False)
    for n, r in enumerate(resultados):
        if args.max_frames and n >= args.max_frames:
            break
        frame = r.orig_img.copy()
        if fondo is None:
            fondo = frame.copy()

        if r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            cajas = r.boxes.xyxy.cpu().numpy()
            for id_, (x1, y1, x2, y2) in zip(ids, cajas):
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)   # centro de la caja
                if id_ not in confirmados:
                    # un objeto que nunca se mueve (ej. un cartel) no es una persona
                    ox, oy = origen.setdefault(id_, (cx, cy))
                    if np.hypot(cx - ox, cy - oy) < MIN_DESPLAZAMIENTO:
                        continue
                    confirmados.add(id_)
                ids_vistos.add(id_)
                estelas[id_].append((cx, cy, n))                  # 1) estela
                sumar_calor(calor, cx, cy)                        # 2) mapa de calor

                z = zona(cy, args.linea_y)                        # 3) contador
                if z:
                    previa = zona_previa.get(id_)
                    if previa == "arriba" and z == "abajo":
                        if id_ not in entradas:
                            print(f"[frame {n}] ENTRADA id={id_}", flush=True)
                        entradas.add(id_)
                    elif previa == "abajo" and z == "arriba":
                        if id_ not in salidas:
                            print(f"[frame {n}] SALIDA  id={id_}", flush=True)
                        salidas.add(id_)
                    zona_previa[id_] = z
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_de(id_), 2)
                cv2.putText(frame, f"ID {id_}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_de(id_), 2)

        if capa_calor is None or n % CADA_N_CALOR == 0:
            capa_calor = crear_capa_calor(calor, (ancho, alto))
        aplicar_calor(frame, *capa_calor)
        dibujar_estelas(frame, estelas, n)

        cv2.line(frame, (0, args.linea_y), (ancho, args.linea_y), AMARILLO, 2)
        cv2.rectangle(frame, (10, alto - 70), (520, alto - 10), (0, 0, 0), -1)   # fondo del marcador
        cv2.putText(frame, f"Entradas: {len(entradas)}   Salidas: {len(salidas)}", (25, alto - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, BLANCO, 2)
        escritor.write(frame)
        if n % 300 == 0:
            print(f"frame {n}/{total}  entradas={len(entradas)} salidas={len(salidas)}", flush=True)

    escritor.release()
    np.save(Path(args.salida).with_name("calor.npy"), calor)
    comprimir(temporal, args.salida)
    Path(temporal).unlink()
    if fondo is not None:
        final = fondo.copy()
        aplicar_calor(final, *crear_capa_calor(calor, (ancho, alto)))
        cv2.imwrite(str(Path(args.salida).with_name("heatmap_final.png")), final)
    print(f"Listo. Entradas: {len(entradas)} | Salidas: {len(salidas)} | IDs distintos: {len(ids_vistos)}")
    print(f"Video: {args.salida}")


if __name__ == "__main__":
    main()
