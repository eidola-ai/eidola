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

space-regenerating = Regenerando…
space-error-response-truncated = El modelo agotó todo su margen de longitud razonando y nunca empezó una respuesta. No se cambió nada: inténtalo de nuevo o haz una pregunta más breve.
space-error-regeneration-in-flight = Esta respuesta ya se está regenerando.
space-answer-cut-off = Esta respuesta alcanzó su límite de longitud y se corta a mitad de idea.
