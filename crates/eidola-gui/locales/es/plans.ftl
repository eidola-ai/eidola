# The shared plans rows (src/plans.rs), Spanish.

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
