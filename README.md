# Caratterizzazione dei sistemi innovativi

## Obiettivi
- Trovare una misura dell'innovazione per un ecosistema di startup (città).
- Predire quali città avranno piú innovazione.

## Dati
- Informazioni sulle aziende: data di fondazione, localizzazione della sede, settori di mercato, ...
- Informazioni sugli impieghi dei lavoratori: id lavoratore, ruolo ricoperto, azienda, date d'inizio e fine (ma manca circa 1/3 delle date d'inizio e la maggior parte delle date di fine), ...

## Lavori in corso
- Come cambia nel tempo il n. di startup e di lavoratori? Sia la funzione sia la cumulativa
- Quanti lavoratori ci sono e quanti contratti in funzione del tempo?
- Grafico a torta sui settori
- Quante aziende italiane? E nelle varie regioni?
- Traiettoria delle persone: quanti salti (mobilità)?
- Ruoli: troppi executive. Estrarre 'founder' da 'executive'
- z-score
- Profilo per aziende e poi media
- Binning negli istogrammi dei ruoli
- Airbnb: dalla nascita, com'è evoluto il profilo dell'azienda?

## Idee
- Fare l'analisi solo su startup. Dovremmo avere una definizione di "startup", tipo azienda che ha ricevuto fondi fino a X, con Y dipendenti, etc.
- Per confrontare i risultati con quelli di Moreno, usare dati fino a dicembre 2015
- Potrebbe essere il caso di eliminare aziende in base alla data di fondazione?
- Pulire i dati: limitare con una data iniziale, sistemare i ruoli (per es., estrarre 'founder' da 'executive')
- Valutare se usare altre misure di distanza
- Provare HDBSCAN

### Per dopo
- Ripetere l'analisi usando i settori di mercato invece dei ruoli
- Fare un clustering a partire dalla rete (imprese collegate da lavoratori). Confrontare i risultati
- Profilo d'ogni città con evoluzione temporale (ogni 5 anni?). Sono stabili nel tempo? Geograficamente c'è stata un'evoluzione? La Silicon Valley è ancora il centro del mondo? Imprese che si spopolano?
- Come definire il successo per un gruppo d'imprese?
- In tempi piú recenti il successo è distribuito piú uniformemente che all'inizio?
- Come valutare l'innovazione?
