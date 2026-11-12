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
space-regenerating-elsewhere = Esta respuesta ya se está regenerando.
space-answer-cut-off = Esta respuesta se detuvo en su límite de longitud.
space-continue-answer = Continuar
space-continue-answer-label = Continuar esta respuesta desde donde se detuvo
space-continuing = Continuando…
space-error-issuer-key-off-calendar = El servidor de Eidola ofreció una clave de pago que no sigue el calendario público de claves, así que esta aplicación no la usó. No se envió nada ni se cobró nada, y los créditos que ya tienes siguen funcionando.
space-error-issuer-key-second-for-epoch = El servidor de Eidola ofreció esta semana una clave de pago distinta de la que esta aplicación ya tiene. Un servidor honesto usa una sola clave por semana, así que esta aplicación no aceptará créditos nuevos con ella hasta la clave de la próxima semana. No se envió nada ni se cobró nada, y los créditos que ya tienes siguen funcionando.
space-error-issuer-key-reused = El servidor de Eidola ofreció la misma clave de pago para dos semanas distintas, algo que un servidor honesto nunca hace, así que esta aplicación no la usó. No se envió nada ni se cobró nada, y los créditos que ya tienes siguen funcionando.
space-error-uncached-retry-unfunded = El motor de Eidola ya no tenía preparado el contexto anterior de esta conversación, y los créditos que tiene esta aplicación no alcanzan para enviarla de nuevo completa. No se convirtió nada más de tu cuenta. Vuelve a intentarlo para enviarla como una solicitud nueva.
