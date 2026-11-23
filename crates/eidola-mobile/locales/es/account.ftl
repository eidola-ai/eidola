# Account (Spanish). See the English source for what each message is.
account-title = Cuenta
account-none = No hay ninguna cuenta en este dispositivo. Una cuenta guarda el crédito que compras; las conversaciones se pagan con créditos anónimos sacados de ella.
account-set-up = Configurar una cuenta…
account-section-credentials = Credenciales
account-show-secret = Mostrar el secreto de la cuenta
account-hide-secret = Ocultar el secreto de la cuenta
account-secret-note = Guárdalos en un lugar seguro. Si se pierde el secreto, no se puede recuperar, y el crédito de la cuenta no se podrá usar.
account-section-balance = Saldo
account-balance = Disponible
account-credits =
    { $count ->
        [one] { $credits } crédito
       *[other] { $credits } créditos
    }
account-pool =
    { $count ->
        [one] { $credits } crédito
       *[other] { $credits } créditos
    }
account-pool-expiring =
    { $count ->
        [one] { $credits } crédito, { $when }
       *[other] { $credits } créditos, { $when }
    }
account-expired = caducado
account-expires-today = caduca hoy
account-expires-tomorrow = caduca mañana
account-expires-days =
    { $count ->
        [one] caduca en { $count } día
       *[other] caduca en { $count } días
    }
account-expires-weeks =
    { $count ->
        [one] caduca en { $count } semana
       *[other] caduca en { $count } semanas
    }
account-expires-months =
    { $count ->
        [one] caduca en { $count } mes
       *[other] caduca en { $count } meses
    }
account-expires-years =
    { $count ->
        [one] caduca en { $count } año
       *[other] caduca en { $count } años
    }
account-balance-loading = Cargando…
account-balance-failed = No se pudo leer el saldo: { $reason }
account-balance-stale = No se pudo volver a leer el saldo; se muestra el último. { $reason }
account-refresh = Actualizar
account-retry = Reintentar
account-add-credit = Añadir crédito
account-section-subscription = Suscripción
account-subscription-checking = Comprobando tu suscripción…
account-subscription-failed = No se pudo comprobar tu suscripción.
account-subscription-stale = No se pudo volver a comprobar tu suscripción; se muestra la última respuesta.
account-subscription-active = Tu suscripción está activa.
account-subscription-trialing = Tu suscripción está en su periodo de prueba.
account-subscription-past-due = Tu suscripción está activa, pero un pago no se completó. Actualiza tu método de pago para conservarla.
account-subscription-other = Tu suscripción está vigente, con el estado «{ $status }».
account-subscription-in-force = Tu suscripción está vigente.
account-subscription-inactive = Ahora mismo no tienes ninguna suscripción.
account-subscription-none = No tienes ninguna suscripción en esta cuenta.
account-manage = Gestionar la suscripción
account-billing = Facturación y recibos
account-check-again = Volver a comprobar
account-opening = Abriendo…
account-stale-mint = La cuenta cambió mientras se preparaba, así que no se abrió nada. Inténtalo de nuevo.
account-door-failed = No se pudo abrir: { $reason }
account-section-reset = Restablecer
account-reset = Restablecer la cuenta…
account-reset-warning = Esto olvida las claves de la cuenta en este dispositivo. El saldo restante queda inaccesible; el registro local de lo gastado se queda en este dispositivo.
account-reset-confirm = Restablecer la cuenta
account-reset-keep = Conservar la cuenta
account-op-failed = No se pudo cambiar la cuenta: { $reason }
wallet-title = Monedero
wallet-note = Las credenciales de gasto anónimas se generan solas a partir de tu saldo; el servidor no puede vincularlas con tu cuenta.
wallet-empty = Aún no hay credenciales; aparecen cuando empiezas a conversar.
wallet-active = activa · generación { $generation }
wallet-spending =
    { $count ->
        [one] en curso: { $credits } crédito retenido
       *[other] en curso: { $credits } créditos retenidos
    }
wallet-spent =
    { $count ->
        [one] gastada: { $credits } crédito cobrado
       *[other] gastada: { $credits } créditos cobrados
    }
wallet-expired = caducada
wallet-failed = No se pudo leer el monedero: { $reason }
account-id = ID de la cuenta
account-secret = Secreto de la cuenta
