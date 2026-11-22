# Roadmap BEAT-Engine-Fork (Stand 22.11.2026, Vorschlag von Claude zur Prüfung)

Ziel des Besitzers: Line-Array- und Hornsimulationen bis 20 kHz mit großen Netzen (z. B. zwei Hörner, 300 mm
Höhe bei λ/12 und 16 kHz; 40 Treiber in einem 1300-mm-Schlitz), auf Mac (Metal) und gemieteter CUDA-GPU, mit
nachweisbarer Genauigkeit statt Raten. Details und Messwerte: `docs/FMM_PLAN.md` auf `hmatrix-dev`.

Zuständigkeit: **C** = Claude (hat als Einziger Zugriff auf den Mac des Besitzers), **G** = ChatGPT (Linux/CPU),
**B** = Besitzer (Entscheidungen, GPU-Kosten, Merge). Aufgaben für ChatGPT tragen das Label `aufgabe-chatgpt`.

## Stand

| Bereich | Stand |
|---|---|
| Speicher-Fork (Metal, CUDA, Split-Verfeinerung) | fertig; Float32-Standardpfad 10–20 kHz Float64-genau (h343: ≤ 0,08 dB auch in Nullstellen) |
| Solver-Fehlerschätzung | PR #2, dritte Review-Runde |
| H-Matrix-Prototyp (`hmatrix-dev`) | PR #3, dritte Review-Runde; Speicher 28 % / 16 % / 9 % des dichten Systems bei 11k / 22k / 43k, Build 30–130 s auf dem Mac |

## Schritte

| # | Schritt | Wer | Abnahmekriterium | Aufwand | hängt ab von |
|---|---|---|---|---|---|
| 1 | PR #2 und #3 abschließen | C, G | Approve von G, Merge durch B | ~0,5 Tag | – |
| 2 | Fehlerschätzung je Ausgabe in dB (Polaren, Membrangeschwindigkeit, Spulenstrom), Hinweis „Float64 nachrechnen“ je Frequenz | C | an h343 (Mac): geschätzte dB-Unsicherheit ≥ wahrer Fehler vs. Float64 in ≥ 95 % der Punkte innerhalb 20 dB des Maximums, keine falsche Freigabe in den bekannten Fehlfällen | 1–2 Tage | 1 |
| 3 | Verfeinerung über die Schätzung steuern (#6) | C | Dump-Konfiguration (`DENSE_SPLIT=off`): Fehler < 1e-4 statt 13 %; Standardpfad: Laufzeit +≤ 5 % | 0,5–1 Tag | 1 |
| 4 | Indikator für den Float32-Assemblierungsfehler (#4) | G (Prototyp CPU), C (Mac-Validierung) | 2-kHz-Fixture und h343: Indikator innerhalb Faktor 3 des wahren Fehlers, Kosten ≤ 20 % eines Frequenzpunkts | 2–3 Tage | 1 |
| 5 | Unabhängige Referenztests für die H-Matrix (Varianten der Fixture, Symmetrie, Interface-Fälle) | G | Tests im Repo, laufen auf CPU in < 10 min, decken jeden Fix der Reviews ab | 1–2 Tage | 1 |
| 6 | H-Matrix-Pfad im Motor hinter einem Schalter (keine dichte BEM-Assemblierung) | C | h343 10/20 kHz: Ausgaben vs. dichter Float64-Pfad ≤ 0,1 dB innerhalb 20 dB; Speicher ≤ 35 % des dichten Pfads | 1–2 Wochen | 1, 2 |
| 7 | GPU-Beschleunigung von H·x und Rekompression; Feldauswertung aus dem H-Pfad | C | 43k-Netz 20 kHz: Gesamtzeit pro Frequenz ≤ dichter Pfad bei gleicher Genauigkeit | ~1 Woche | 6 |
| 8 | Großes Modell (zwei Hörner, sobald STEP vorliegt): Netz, Kosten- und Zeitabschätzung | C, B | Bericht vor jeder GPU-Miete | 1 Tag | 6 |
| 9 | VACS-Export der Ergebnisse | G | vom Besitzer in VACS geöffnet und geprüft | 2–3 Tage | – |

Gesamt bis Schritt 7: etwa 3–4 Wochen. Schritte 4, 5 und 9 laufen parallel bei G.

## Fragen an ChatGPT

1. Ist die Reihenfolge sinnvoll, insbesondere 2 und 3 vor 6?
2. Sind die Abnahmekriterien messbar und streng genug? Fehlen Fälle (z. B. Resonanzen, schlechte Kondition)?
3. Welche der Aufgaben 4, 5, 9 übernimmst du zuerst?
