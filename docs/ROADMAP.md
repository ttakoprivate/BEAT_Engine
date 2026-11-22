# Roadmap BEAT-Engine-Fork (Stand 22.11.2026, Vorschlag von Claude zur Prüfung)

Ziel des Besitzers: Line-Array- und Hornsimulationen bis 20 kHz mit großen Netzen (z. B. zwei Hörner, 300 mm
Höhe bei λ/12 und 16 kHz; 40 Treiber in einem 1300-mm-Schlitz), auf Mac (Metal) und gemieteter CUDA-GPU, mit
nachweisbarer Genauigkeit statt Raten. Details und Messwerte: `docs/FMM_PLAN.md` auf `hmatrix-dev`.

Zuständigkeit: **C** = Claude (hat als Einziger Zugriff auf den Mac des Besitzers), **G** = ChatGPT (Linux/CPU),
**B** = Besitzer (Entscheidungen, GPU-Kosten, Merge). Aufgaben für ChatGPT tragen das Label `chatgpt` (wie in `docs/COLLABORATION.md`).

## Stand

| Bereich | Stand |
|---|---|
| Speicher-Fork (Metal, CUDA, Split-Verfeinerung) | fertig; Float32-Standardpfad 10–20 kHz, h343: Pegelabweichung (nur Betrag) ≤ 0,08 dB gegenüber Float64, auch in Nullstellen; der maximale komplexe relative Fehler je Ausgabe wird mit Schritt 2 nachgereicht |
| Solverdiagnose (Fehlerschätzung des Lösers) | PR #2, Approve von G für `8722d12` |
| H-Matrix-Prototyp (`hmatrix-dev`) | PR #3, dritte Review-Runde; Speicher 28 % / 16 % / 9 % des dichten Systems bei 11k / 22k / 43k, Build 30–130 s auf dem Mac |

## Schritte

| # | Schritt | Wer | Abnahmekriterium | Aufwand | hängt ab von |
|---|---|---|---|---|---|
| 1 | PR #2 (Solverdiagnose) und #3 abschließen | C, G | Approve von G, Merge durch B. PR #2 ist nur eine **Solverdiagnose**: Fehler beim Aufbau der Matrix (Float32-Assemblierung) sieht sie prinzipbedingt nicht. Eine Aussage zur Gesamtgenauigkeit erst mit Schritt 4 und einem Test gegen eine unabhängig assemblierte Float64-Dichtreferenz | ~0,5 Tag | – |
| 2 | Fehlerschätzung je Ausgabe in dB (Polaren, Membrangeschwindigkeit, Spulenstrom), Hinweis „Float64 nachrechnen“ je Frequenz | C | an h343 (Mac): geschätzte dB-Unsicherheit ≥ wahrer Fehler vs. Float64 in ≥ 95 % der Punkte innerhalb 20 dB des Maximums, keine falsche Freigabe in den bekannten Fehlfällen; zusätzlich je Ausgabe der maximale komplexe relative Fehler; die größten Abweichungen und die Zahl falscher Freigaben bei Resonanzen und Auslöschungen werden getrennt ausgewiesen | 1–2 Tage | 1 |
| 3 | Verfeinerung über die Schätzung steuern (#6) | C | Dump-Konfiguration (`BLAB_COUPLED_DENSE_SPLIT=0`): Fehler < 1e-4 statt 13 %; Standardpfad: Laufzeit +≤ 5 % | 0,5–1 Tag | 1 |
| 4 | Indikator für den Float32-Assemblierungsfehler (#4) | G (Prototyp CPU), C (Mac-Validierung) | 2-kHz-Fixture und h343: Indikator innerhalb Faktor 3 des wahren Fehlers, Kosten ≤ 20 % eines Frequenzpunkts | 2–3 Tage | 1 |
| 5 | Unabhängige Referenztests für die H-Matrix (Varianten der Fixture, Symmetrie, Interface-Fälle) | G | Tests im Repo in eigenen Dateien von G (nicht in Claudes `tests/fmm_*`, Zuständigkeit nach `docs/COLLABORATION.md`; Zielbranch entscheidet B in #11), laufen auf CPU in < 10 min, decken jeden Fix der Reviews ab, einschließlich gekoppelter Druck-/Flussausgaben bei Interferenz und einer Frequenz nahe einer Resonanz | 1–2 Tage | 1 |
| 6 | H-Matrix-Pfad im Motor hinter einem Schalter (keine dichte BEM-Assemblierung) | C | h343 10/20 kHz, je Ausgabe und Anregung: maximaler komplexer relativer Fehler gegenüber dem dichten Float64-Pfad (Schwelle legt B nach Schritt 2 fest), zusätzlich ≤ 0,1 dB innerhalb 20 dB; Speicher ≤ 35 % des dichten Pfads. Integration in `memory-fork` erst nach bestandenen Tests aus Schritt 5 und mit Schritten 2 und 3 | 1–2 Wochen | 1 (Implementierung parallel zu 2, 3) |
| 7 | GPU-Beschleunigung von H·x und Rekompression; Feldauswertung aus dem H-Pfad | C | 43k-Netz 20 kHz: Gesamtzeit pro Frequenz ≤ dichter Pfad bei gleicher Genauigkeit, gemessen nach `docs/Benchmarking.md` (≥ 3 abwechselnde Läufe, Median und Spannweite, Abschnittszeiten, Spitzenspeicher, Genauigkeit gegen Referenz); keine Aussage innerhalb der Lauf-zu-Lauf-Streuung | ~1 Woche | 6 |
| 8 | Großes Modell (zwei Hörner, sobald STEP vorliegt): Netz, Kosten- und Zeitabschätzung | C, B | Bericht vor jeder GPU-Miete | 1 Tag | 6 |
| 9 | VACS-Export der Ergebnisse | G | feste Ergebnis-Fixture: exportierte Werte und Metadaten (Frequenzachse, Einheiten, Reihenfolge) mit festgelegten Toleranzen gegen die Engine-Ausgabe verglichen (Unit-Test); danach vom Besitzer in VACS geöffnet und bestätigt | 2–3 Tage | – |

Gesamt bis Schritt 7: etwa 3–4 Wochen. Schritte 4, 5 und 9 laufen parallel bei G.

**Genauigkeitsmaß für alle numerischen Abnahmen:** maximaler komplexer relativer Fehler je physikalischer Ausgabe
und Anregung gegenüber der unabhängig assemblierten Float64-Dichtlösung. Pegelangaben in dB sind nur eine Ergänzung
(+1 Pa und −1 Pa haben 0 dB Pegeldifferenz, aber 200 % komplexen Fehler). Ist die Referenz exakt null, wird der
absolute Fehler mit Einheit angegeben.

**Offen für B:** Reihenfolge (Vorschlag G: 2 und 3 vor der automatischen Freigabe, H-Matrix-Implementierung parallel);
Zuständigkeit und Zielbranch der Tests aus Schritt 5 (#11); VACS-Format (#12).

## Fragen an ChatGPT

1. Ist die Reihenfolge sinnvoll, insbesondere 2 und 3 vor 6?
2. Sind die Abnahmekriterien messbar und streng genug? Fehlen Fälle (z. B. Resonanzen, schlechte Kondition)?
3. Welche der Aufgaben 4, 5, 9 übernimmst du zuerst?
