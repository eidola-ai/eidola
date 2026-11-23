# Purchase (Spanish). See the English source for what each message is.
purchase-title = Añadir crédito
purchase-intro = Elige un plan o compra crédito una sola vez. Los créditos de suscripción duran su periodo de facturación; las compras únicas duran un año. Los créditos no usados y no caducados son reembolsables si los pides.
purchase-checkout-note = El pago se abre en tu navegador, fuera de esta aplicación, y el crédito llega a esta cuenta. Vuelve aquí cuando termines.
purchase-subscribed-note = Esta cuenta ya tiene una suscripción; gestiónala desde Cuenta. Aquí aún puedes añadir crédito único.
purchase-loading = Cargando planes…
purchase-failed = No se pudieron cargar los planes: { $reason }
purchase-none = Ahora mismo no hay planes disponibles.
purchase-none-topups = Ahora mismo no hay recargas puntuales disponibles.
purchase-opened = El pago está abierto en tu navegador. Tu saldo se vuelve a leer cuando regresas.

# The plan rows: the desktop app's plans.ftl translations.
plans-list = Planes disponibles
plans-opening-checkout = Abriendo el pago…
plans-free = Gratis
plans-price-cadence =
    { $count ->
        [1]
            { $interval ->
                [day] { $amount }/día
                [week] { $amount }/semana
                [month] { $amount }/mes
                [year] { $amount }/año
               *[other] { $amount }/{ $interval }
            }
       *[other]
            { $interval ->
                [day] { $amount } cada { $count } días
                [week] { $amount } cada { $count } semanas
                [month] { $amount } cada { $count } meses
                [year] { $amount } cada { $count } años
               *[other] { $amount } cada { $count } { $interval }
            }
    }
plans-credits-one-time =
    { $count ->
        [one] { $credits } crédito, caduca un año después de la compra
       *[other] { $credits } créditos, caducan un año después de la compra
    }
plans-credits-recurring =
    { $count ->
        [one] { $credits } crédito, caduca al final de cada periodo de facturación
       *[other] { $credits } créditos, caducan al final de cada periodo de facturación
    }
plans-credits-described = { $line } — { $description }
