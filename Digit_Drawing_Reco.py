# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:34:43 2026

@author: quent
"""
from Model.model import CNN
import tkinter as tk

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageDraw

MEAN, STD = 0.1307, 0.3081
TAILLE, EPAISSEUR = 280, 22


def preparer(image):
    # Dessin 280x280 (blanc sur noir) -> tenseur (1, 1, 28, 28) au format MNIST
    bbox = image.getbbox()
    if bbox is None:
        return None
    chiffre = image.crop(bbox)

    w, h = chiffre.size
    echelle = 20 / max(w, h)
    taille = (max(1, round(w * echelle)), max(1, round(h * echelle)))
    chiffre = chiffre.resize(taille, Image.LANCZOS)

    img28 = Image.new("L", (28, 28), 0)
    img28.paste(chiffre, ((28 - taille[0]) // 2, (28 - taille[1]) // 2))

    a = np.asarray(img28, dtype=np.float32)
    ys, xs = np.indices(a.shape)
    cy = (ys * a).sum() / a.sum()
    cx = (xs * a).sum() / a.sum()
    img28 = ImageChops.offset(img28, round(13.5 - cx), round(13.5 - cy))

    x = np.asarray(img28, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


class App:
    def __init__(self, racine):
        racine.title("Chiffre manuscrit")
        self.canvas = tk.Canvas(racine, width=TAILLE, height=TAILLE, bg="black", cursor="cross")
        self.canvas.pack()
        self.label = tk.Label(racine, text="Dessine un chiffre", font=("Arial", 16))
        self.label.pack(pady=5)
        tk.Button(racine, text="Effacer", command=self.effacer).pack(pady=5)

        self.image = Image.new("L", (TAILLE, TAILLE), 0)   # copie du dessin, lisible en pixels
        self.dessin = ImageDraw.Draw(self.image)
        self.dernier = None

        self.canvas.bind("<ButtonPress-1>", self.debut)
        self.canvas.bind("<B1-Motion>", self.trace)
        self.canvas.bind("<ButtonRelease-1>", self.fin)

    def point(self, x, y):
        r = EPAISSEUR / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.dessin.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def debut(self, event):
        self.dernier = (event.x, event.y)
        self.point(event.x, event.y)

    def trace(self, event):
        x0, y0 = self.dernier
        x1, y1 = event.x, event.y
        self.canvas.create_line(x0, y0, x1, y1, width=EPAISSEUR, fill="white", capstyle=tk.ROUND)
        self.dessin.line((x0, y0, x1, y1), fill=255, width=EPAISSEUR)
        self.point(x1, y1)
        self.dernier = (x1, y1)

    def fin(self, event):
        self.dernier = None
        self.predire()

    def predire(self):
        x = preparer(self.image)
        if x is None:
            return
        with torch.no_grad():
            probas = F.softmax(model(x), dim=1)[0]
        confiance, chiffre = probas.max(0)
        self.label.config(text=f"Prédiction : {chiffre.item()}  ({confiance.item():.0%})")

    def effacer(self):
        self.canvas.delete("all")
        self.dessin.rectangle((0, 0, TAILLE, TAILLE), fill=0)
        self.label.config(text="Dessine un chiffre")


if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("Model/mnist_cnn.pt"))
    model.eval()

    racine = tk.Tk()
    App(racine)
    racine.mainloop()
