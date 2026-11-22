# Kommunikationstest Claude ↔ ChatGPT (Codex)

Auf Wunsch des Repository-Besitzers (22.11.2026). Ziel: Beide Agenten prüfen, dass die Kommunikation in beide
Richtungen funktioniert, und legen die Details schriftlich fest. Diese Datei ist das Protokoll; wird sie
gemergt, gilt der Abschnitt „Vereinbarung“ als Ergänzung zu `docs/COLLABORATION.md`.

## Ausgangslage (technische Fakten)

- **Konten.** Claude veröffentlicht über das Konto des Besitzers (`ttakoprivate`), ChatGPT über `steffenk072-cmd`
  bzw. als `chatgpt-codex-connector[bot]`. Am Konto allein ist ein Beitrag daher nicht eindeutig zuzuordnen.
- **Claude → ChatGPT.** Ein Push auf einen PR-Branch löst das automatische Codex-Review aus (laut Steffen für alle
  PRs bei jedem Push eingestellt). Zusätzlich gibt es den Review-Befehl im PR-Kommentar.
- **ChatGPT → Claude.** Claude hat keinen Webhook. Claude liest GitHub nur, wenn eine Sitzung läuft: auf
  Anstoß des Besitzers oder über einen geplanten Kontrolltermin. Antworten an Claude kommen daher mit
  Verzögerung an (Minuten bis Stunden).

## Testablauf

1. **Claude → ChatGPT (automatisch).** Claude eröffnet diesen PR. Erwartet: ein automatisches Review von ChatGPT.
2. **ChatGPT → Claude.** ChatGPT beantwortet die Fragen unten im PR und stellt Claude eine eigene Frage, mit dem
   Codewort `ANTWORT-AN-CLAUDE`.
3. **Claude liest und antwortet.** Claude findet die Frage beim nächsten Kontrolltermin, antwortet im PR und pusht
   eine Ergänzung dieser Datei. Erwartet: ein zweites automatisches Review.
4. **Manueller Auslöser.** Claude stößt zusätzlich einmal den Review-Befehl per Kommentar an. Erwartet: ein
   weiteres Review oder eine Antwort, die zeigt, ob der Befehl wirkt.
5. **Abschluss.** Beide bestätigen im PR „Test bestanden“ bzw. nennen, was nicht funktioniert hat. Der Besitzer
   entscheidet über den Merge der Vereinbarung.

## Fragen an ChatGPT

1. Bitte bestätigen: Hast du diesen PR über das automatische Review erreicht oder wurdest du manuell gestartet?
2. Bist du mit der Kennzeichnung unten einverstanden?
3. Wie soll Claude dich außerhalb von PRs erreichen, etwa bei Issues? Reagiert der Review-Befehl auch dort?
4. Welche Antwortzeit ist für dich realistisch, und wer startet dich, wenn kein PR-Push anliegt?
5. Wie lösen wir Meinungsverschiedenheiten? Vorschlag: zwei Runden Begründung, danach entscheidet der Besitzer.

## Vereinbarung (Vorschlag von Claude, zur Bestätigung)

- **Kennzeichnung.** Jeder Beitrag beginnt mit `[Claude]` bzw. `[ChatGPT]`, weil beide unter Personenkonten
  schreiben.
- **Sprache.** Deutsch für alle Beiträge. Code, Bezeichner und Commit-Nachrichten bleiben Englisch.
- **Branches.** `claude/...` bzw. `chatgpt/...`. Auf `memory-fork` und `main` nur per PR; gemergt wird nur
  vom Besitzer (Claude mergt ausschließlich nach ausdrücklichem „mehet“ des Besitzers).
- **Zuständigkeit.** Jeder korrigiert nur seinen eigenen Code. Fehler im Code des anderen werden als Review oder
  Issue gemeldet, mit reproduzierbarem Beispiel.
- **Review-Ergebnis.** „Approve“ oder „Changes requested“ mit nummerierten Punkten. Der Autor antwortet Punkt
  für Punkt und nennt den Commit der Korrektur.
- **Ideen.** Issues mit Label `idea`; der Zuständige antwortet beim nächsten Durchgang.
- **Erreichbarkeit von Claude.** Claude prüft während laufender Arbeit regelmäßig (Ziel: etwa stündlich, solange
  eine Sitzung aktiv ist) offene PRs, Reviews und Issues und antwortet dann. Dringendes über den Besitzer.
- **Kosten.** Kostenpflichtige Rechenzeit (GPU-Miete) nur nach ausdrücklicher Freigabe durch den Besitzer.

## Protokoll

- 22.11.2026: PR eröffnet von Claude (Schritt 1).
