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
- **Branches.** `claude/...` bzw. `chatgpt/...`. Änderungen an `memory-fork` nur per PR. `main` bleibt
  unveränderter Upstream-Spiegel, ohne Commits und ohne PRs (wie in `docs/COLLABORATION.md`).
- **Merge.** Der Besitzer entscheidet. Entscheidung des Besitzers (22.11.2026): Claude führt den Merge aus,
  aber nur nach seiner ausdrücklichen Freigabe („mehet“) für genau diesen PR. ChatGPT merged nicht. Diese
  Ausnahme ergänzt die Regel „kein Agent merged“ in `docs/COLLABORATION.md`.
- **Zuständigkeit.** Jeder korrigiert nur seinen eigenen Code. Fehler im Code des anderen werden als Review oder
  Issue gemeldet, mit reproduzierbarem Beispiel.
- **Review-Ergebnis.** „Approve“, „Changes requested“ oder „Comment“ (nicht blockierende Hinweise), wie in
  `docs/COLLABORATION.md`, mit nummerierten Punkten. Der Autor antwortet Punkt
  für Punkt und nennt den Commit der Korrektur.
- **Ideen.** Issues mit Label `idea`; der Zuständige antwortet beim nächsten Durchgang.
- **Erreichbarkeit von Claude.** Claude prüft während laufender Arbeit regelmäßig (Ziel: etwa stündlich, solange
  eine Sitzung aktiv ist) offene PRs, Reviews und Issues und antwortet dann. Dringendes über den Besitzer.
- **Kosten.** Kostenpflichtige Rechenzeit (GPU-Miete) nur nach ausdrücklicher Freigabe durch den Besitzer.

## Protokoll

- 22.11.2026, 10:28 (MEZ): PR eröffnet von Claude (Schritt 1).
- 10:28–10:30: automatisches Codex-Review (`chatgpt-codex-connector[bot]`) nach ~2,5 Minuten, Auslöser
  „PR opened“. Es prüft den Inhalt wie Code (3 berechtigte Hinweise, hier eingearbeitet), beantwortet aber die
  Fragen nicht: Der automatische Review-Bot ist ein Prüfer, kein Gesprächspartner. Schritt 2 ist damit offen.
- 10:47: Claude arbeitet die 3 Hinweise ein (dieser Push = Test „zweites automatisches Review“) und testet,
  ob eine Erwähnung des Codex-Bots mit einer Frage eine inhaltliche Antwort auslöst.
- 10:47–10:49: Push → zweites automatisches Review nach ~2 Minuten, ohne Befund. **Claude → ChatGPT
  funktioniert automatisch.**
- 10:47: Erwähnung `@codex` mit inhaltlicher Bitte aus Claudes Kommentar (Konto des Besitzers) → Antwort des Bots
  „To use Codex here, create a Codex account and connect to github“. Erwähnungen wirken nur für ein GitHub-Konto,
  das mit einem ChatGPT/Codex-Konto verbunden ist; das Konto des Besitzers ist es nicht. Die inhaltlichen
  Antworten (Schritt 2) kommen deshalb nur, wenn Steffen ChatGPT startet.
- Offen: Schritt 2 (Antwort von ChatGPT auf die Fragen 1–5 und eine Frage an Claude).
