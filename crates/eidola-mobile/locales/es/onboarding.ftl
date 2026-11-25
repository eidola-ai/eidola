# The first run on a phone (src/ui/onboarding.rs), Spanish: the
# desktop app's onboarding translations where the words are the same.

# The screen's title.
onboarding-title = Bienvenida

# The two published document titles and the id placeholder are deliberately
# absent, as on the desktop: they fall back to the English source.

onboarding-pause-body =
    ## *Detente un momento*

    Eidola *no* es lo mismo que ChatGPT, Claude o Gemini.

onboarding-control-body =
    ## *Tu* control

    Tú y solo tú tienes el control: ni nosotros, ni los operadores que gestionan el hardware:

    - **Solo tú puedes leer, conservar o perfilar tus interacciones.** Tus datos se descifran únicamente dentro de enclaves sellados y atestiguados por hardware, que no conservan nada, y la parte de Eidola que gestiona el pago está separada criptográficamente de la que atiende tus solicitudes.
    - **Solo tú puedes actualizar Eidola, tanto en tu equipo como en el servidor.** Nada cambia hasta que tu cliente verifica una nueva versión y tú decides confiar en ella.

    No confíes ciegamente en lo que decimos; compruébalo. Si no sabes cómo evaluar nuestro código y nuestra arquitectura, **pide la opinión de la persona más técnica en la que ya confíes**.

onboarding-responsibility-body =
    ## *Tu* responsabilidad

    Eidola es una herramienta que te facilita ejecutar modelos de IA sin conocimientos técnicos muy especializados ni hardware caro. Nadie está mirando por encima de tu hombro, así que es fundamental que entiendas lo siguiente:

    - Vas a ejecutar modelos de IA, que en el fondo son conjuntos de probabilidades. Los modelos que Eidola ejecuta se pueden descargar y usar libremente, con o sin Eidola.
    - Incluso los mejores modelos son falibles. Pueden ser asombrosamente útiles, pero cometen errores y pueden comportarse de forma inesperada. Es matemáticamente imposible evaluar cómo se comportará un modelo grande en todos los escenarios posibles.
    - Los modelos en sí no tienen memoria intrínseca ni capacidad de causar efectos externos; solo pueden evaluar datos y realizar las acciones que tú pongas a su alcance. Eidola facilita entender y configurar ese acceso, pero los resultados — buenos y malos — son en última instancia responsabilidad tuya.

onboarding-create-account-body =
    ## Crear una cuenta

    Lee y comprende nuestros documentos { -fixed-terms-of-service } y { -fixed-privacy-policy }.

onboarding-new-account-body =
    ## Tu nueva cuenta

    Tu nueva cuenta se ha creado:

onboarding-existing-account-body =
    ## Tu cuenta existente

    Introduce los datos de tu cuenta:

onboarding-purchase-body =
    ## Añadir crédito

    Elige un plan o compra créditos directamente. Usamos Stripe para procesar los pagos.

    Los créditos de suscripción duran su periodo de facturación; las compras puntuales duran un año. Los créditos no usados y no caducados son reembolsables si los solicitas.

onboarding-cta-understood = Lo entiendo.

onboarding-cta-new-account = Necesito una cuenta nueva.

onboarding-cta-existing-account = Ya tengo una cuenta.

onboarding-consent-agree = Acepto los documentos { -fixed-terms-of-service } y { -fixed-privacy-policy }.

onboarding-terms-loading = Comprobando los documentos actuales…

onboarding-terms-retry = Reintentar

onboarding-link-external = { $label } ↗

onboarding-document-version = { $name } (versión { $version })

onboarding-cta-create = Crear una cuenta nueva.

onboarding-cta-create-pending = Creando tu cuenta anónima…

onboarding-account-id = ID de la cuenta

onboarding-account-secret = Secreto de la cuenta

onboarding-account-secret-placeholder = secreto de la cuenta

onboarding-new-account-note = Solo sirve para añadir y consumir créditos. Si pierdes el secreto de la cuenta, no se puede recuperar y tendrás que crear una cuenta nueva.

onboarding-cta-saved = Ya lo he guardado en algún sitio.

onboarding-create-not-saved = Tu cuenta se creó, pero no se pudo guardar aquí ({ $reason }). Esta es la única copia de su secreto: copia ambos ahora y, cuando se resuelva, vincúlalos con «Ya tengo una cuenta.».

onboarding-create-recovery-file = También se guardó una copia en { $path }. Borra ese archivo cuando hayas vinculado la cuenta.

onboarding-cta-verify = Consultar el saldo de la cuenta.

onboarding-cta-verify-pending = Comprobando…

onboarding-verify-missing-credentials = Introduce tanto el ID como el secreto de la cuenta.

onboarding-verify-unverifiable = No hemos podido verificar esa cuenta. Comprueba el ID y el secreto, o crea una cuenta nueva.

onboarding-verified-balance =
    { $count ->
        [one] Esta cuenta es válida y tiene un saldo de { $credits } crédito.
       *[other] Esta cuenta es válida y tiene un saldo de { $credits } créditos.
    }

onboarding-cta-existing-purchase = Quiero comprar más créditos.

onboarding-cta-existing-done = Esto me sirve.

onboarding-purchase-checkout-note = El pago se abre en tu navegador; el crédito llega a esta cuenta.

onboarding-cta-purchase-later = Compraré créditos más adelante.

onboarding-get-started-body =
    ## Empezar

    Necesitarás créditos para ejecutar modelos.

    Tu cuenta no es más que un identificador aleatorio: comprar crédito es el único paso que toca un método de pago, e incluso entonces no podemos, por construcción, vincular tus solicitudes con él.

onboarding-link-unlinkability = Cómo tus solicitudes quedan desvinculadas de tu pago
onboarding-cta-skip-account = Continuar sin cuenta.
onboarding-cta-plans = Elegir un plan…
onboarding-create-failed = No se pudo crear la cuenta: { $reason }
onboarding-verify-failed = No se pudo comprobar esa cuenta: { $reason }

onboarding-accept-terms-body =
    ## Aceptar los términos

    Las compras en esta cuenta necesitan que acepte los { -fixed-terms-of-service } y la { -fixed-privacy-policy } vigentes. Léelos, por favor.
onboarding-cta-review-terms = Revisar y aceptar los términos…
onboarding-cta-record-agreement = Registrar mi aceptación.
onboarding-cta-record-agreement-pending = Registrando tu aceptación…
onboarding-terms-recorded = Tu aceptación ha quedado registrada.
onboarding-cta-continue = Continuar.
onboarding-accept-failed = No se pudo registrar la aceptación: { $reason }
onboarding-terms-outstanding = La aceptación de { $documents } por parte de la cuenta no está registrada, así que sus compras se rechazarán hasta que lo esté.
onboarding-terms-unknown = No se pudo confirmar si quedó registrada la aceptación de los términos por parte de la cuenta ({ $reason }), así que una compra podría rechazarse.
