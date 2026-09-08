# The onboarding window (src/onboarding/), Spanish.
#
# `-terms-of-service`, `-privacy-policy`, `onboarding-link-terms-of-service`,
# `onboarding-link-privacy-policy` and `onboarding-account-id-placeholder` are
# deliberately absent: the two published document titles stay English in every
# locale, and the id placeholder is a shape rather than words. They fall back to
# the English source.

onboarding-pause-body =
    ## *Detente un momento*

    Eidola *no* es lo mismo que ChatGPT, Claude o Gemini.

onboarding-tool-body =
    ## Eidola es *tu* herramienta

    Hace años, una aplicación llegaba a tu ordenador en un CD:

    - Su comportamiento *no podía* cambiar de forma espontánea sin que tú intervinieras.
    - Tus archivos, tus planes, tus hábitos de uso y tus ideas eran *solo tuyos*, fuera del alcance de cualquier tercero.
    - Era la **estructura** de la tecnología — y *no* las promesas de una empresa — la que garantizaba esas propiedades.

    Eidola se acerca a ese planteamiento todo lo posible, maximizando estructuralmente la soberanía del usuario incluso para cargas de trabajo que conviene ejecutar en un centro de datos.

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

onboarding-get-started-body =
    ## Empezar

    Necesitarás créditos para ejecutar modelos.

    Tu cuenta no es más que un identificador aleatorio: comprar crédito es el único paso que toca un método de pago, e incluso entonces [no podemos, por construcción, vincular tus solicitudes con él]({ $unlinkability }).

onboarding-create-account-body =
    ## Crear una cuenta

    Lee y comprende nuestros documentos { -terms-of-service } y { -privacy-policy }.

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

onboarding-back = Ir a la diapositiva anterior

onboarding-cta-pause = De acuerdo, tienes mi atención.
onboarding-cta-understood = Lo entiendo.
onboarding-cta-new-account = Necesito una cuenta nueva.
onboarding-cta-existing-account = Ya tengo una cuenta.
onboarding-cta-skip-account = Continuar sin cuenta: solo modelos en el dispositivo.

onboarding-consent-agree = Acepto los documentos { -terms-of-service } y { -privacy-policy }.
onboarding-terms-loading = Comprobando los documentos actuales…
onboarding-terms-retry = Reintentar

onboarding-link-external = { $label } ↗
onboarding-link-repository = El repositorio de código de Eidola
onboarding-document-version = { $name } (versión { $version })

onboarding-cta-create = Crear una cuenta nueva.
onboarding-cta-create-pending = Creando tu cuenta anónima…

onboarding-account-id = ID de la cuenta
onboarding-account-secret = Secreto de la cuenta
onboarding-account-secret-placeholder = secreto de la cuenta
onboarding-copy = Copiar
onboarding-copy-label = Copiar { $label }
onboarding-new-account-note = Solo sirve para añadir y consumir créditos. Si pierdes el secreto de la cuenta, no se puede recuperar y tendrás que crear una cuenta nueva.
onboarding-cta-saved = Ya lo he guardado en algún sitio.

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
onboarding-purchase-subscribed-note = Esta cuenta ya tiene una suscripción: gestiónala en Settings ▸ Account. Aun así, aquí puedes añadir crédito puntual.
onboarding-purchase-loading = Cargando planes…
onboarding-purchase-none-topups = Ahora mismo no hay recargas puntuales disponibles.
onboarding-purchase-none-plans = Ahora mismo no hay planes disponibles.
onboarding-checkout-stale-mint = La cuenta cambió mientras se preparaba, así que no se abrió nada. Inténtalo de nuevo.
onboarding-cta-purchase-later = Compraré créditos más adelante.
