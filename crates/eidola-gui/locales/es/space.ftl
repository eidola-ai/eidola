# The conversation window (src/space_view/mod.rs), Spanish.

space-error-archived = Esta conversación está archivada, así que no puede recibir nuevas respuestas.
space-error-not-joined = Tus agentes abrieron esta conversación entre ellos. Puedes leerla entera; participar significa unirte a ella, algo que esta versión aún no puede hacer.

space-footnote-delegation-concluded = llegó a su fin
space-footnote-delegation-paused = alcanzó su límite de respuestas ({ $depth } de { $limit })
space-footnote-delegation-budget = agotó sus { $limit } turnos
space-footnote-delegation-failed = se detuvo: { $reason ->
        [upstream] no se pudo contactar con el modelo
        [funding] no se pudo pagar el turno
        [configuration] algo en su configuración
       *[other] no se pudo completar un turno
    }
