# The conversation window (src/space_view/mod.rs), Spanish.

space-error-archived = Esta conversación está archivada, así que no puede recibir nuevas respuestas.
space-error-not-joined = Tus agentes abrieron esta conversación entre ellos. Puedes leerla entera; para cambiar algo o volver a pedirlo aquí, participa primero publicando un mensaje.

space-footnote-delegation-concluded = llegó a su fin
space-footnote-delegation-concluded-truncated = llegó a su fin, sobre una respuesta cortada en su límite de longitud
space-footnote-delegation-paused = alcanzó su límite de respuestas ({ $depth } de { $limit })
space-footnote-delegation-paused-truncated = alcanzó su límite de respuestas ({ $depth } de { $limit }), sobre una respuesta cortada en su límite de longitud
space-footnote-delegation-budget = agotó sus { $limit } turnos
space-footnote-delegation-budget-truncated = agotó sus { $limit } turnos, sobre una respuesta cortada en su límite de longitud
space-footnote-delegation-failed = se detuvo: { $reason ->
        [upstream] no se pudo contactar con el modelo
        [funding] no se pudo pagar el turno
        [configuration] algo en su configuración
       *[other] no se pudo completar un turno
    }

space-regenerating = Regenerando…
space-error-model-unavailable = { $model } ya no está disponible en su backend, y ningún otro modelo responderá en su lugar. No se envió nada ni se cobró nada: elige otro modelo e inténtalo de nuevo.
space-error-hold-exceeds-denomination = Esta respuesta podría costar más de lo que una sola solicitud puede retener, así que no se envió nada ni se cobró nada. Añadir crédito no lo cambiará: prueba con una conversación más corta o con otro modelo.
space-regenerating-elsewhere = Esta respuesta ya se está regenerando.
space-answer-cut-off = Esta respuesta se detuvo en su límite de longitud.
space-continue-answer = Continuar
space-continue-answer-label = Continuar esta respuesta desde donde se detuvo
space-continuing = Continuando…
