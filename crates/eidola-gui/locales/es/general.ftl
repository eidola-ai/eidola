# Ajustes ▸ General — apariencia, tamaño del texto, inicio y línea de comandos
# (src/general.rs), español.

general-appearance = Apariencia

general-day-night = Día y noche
general-day-night-auto = Automático
general-day-night-system = Sistema
general-day-night-day = Día
general-day-night-night = Noche
general-day-night-choice = Día y noche: { $choice }
general-day-night-note = Sistema sigue a macOS. Automático sigue al sol — de día entre el amanecer y el atardecer.

general-time-of-day = Hora del día
general-on = Activado
general-off = Desactivado
general-time-of-day-choice = Hora del día: { $choice }
general-time-of-day-note = Sigue la luz del sol — fría hacia el amanecer, neutra a mediodía y a medianoche, cálida hacia el atardecer. El amanecer y el atardecer se estiman a partir de tu zona horaria.

general-light = Luz
general-light-cool = Fría
general-light-neutral = Neutra
general-light-warm = Cálida
general-light-choice = Luz: { $choice }
general-light-note = Un carácter de luz fijo para la paleta.

general-text-size = Tamaño del texto
general-text-size-smaller-name = Texto más pequeño, ahora { $percent }
general-text-size-larger-name = Texto más grande, ahora { $percent }
general-text-size-reset = Restablecer
general-text-size-reset-name = Restablecer el tamaño del texto al 100 %
general-text-size-note = Cambia el tamaño de todo el texto. También en el menú View — Actual Size, Zoom In, Zoom Out.

general-startup = Inicio
general-open-at-login = Abrir al iniciar sesión
general-login-item-switch = Abrir Eidola al iniciar sesión
general-login-item-name = Abrir Eidola al iniciar sesión
general-login-item-off = Eidola no aparece hasta que lo abres.
general-login-item-on = Eidola se abre con tu sesión.
general-login-item-needs-approval = Activado, pero macOS espera tu permiso — permite Eidola en Ajustes del Sistema → General → Ítems de inicio.
general-login-item-unsupported = No disponible — hace falta macOS 13 o posterior, y una app instalada y firmada.
general-login-item-unsupported-linux = No disponible aquí — en Linux, `eidola service install` activa el servicio de usuario.
general-login-item-too-old = Abrir al iniciar sesión requiere macOS 13 o posterior — el sistema de este Mac es más antiguo.
general-login-item-not-here = Abrir al iniciar sesión no está disponible en esta plataforma.

general-command-line = Línea de comandos
general-cli-field = Comando de Terminal
general-cli-not-installed = No instalado. Al instalarlo se pone un comando `eidola` en { $dir }, enlazado a esta copia de Eidola, para usarlo desde Terminal.
general-cli-needs-admin = macOS pedirá la contraseña de un administrador.
general-cli-installed = Instalado — `eidola` en Terminal ejecuta esta copia de Eidola, a través de { $path }.
general-cli-dangling = { $path } apunta a una copia de Eidola que ya no está ({ $target }) — probablemente la app se movió. Actualiza el enlace para usar esta copia.
general-cli-other-copy = { $path } ejecuta otra copia de Eidola, en { $target }. Actualiza el enlace para usar esta copia.
general-cli-occupied-file = { $path } ya es un archivo que Eidola no puso ahí, así que Eidola no lo sustituirá. Si ya no lo quieres, muévelo o bórralo tú y luego instala.
general-cli-occupied-directory = { $path } es una carpeta, así que Eidola no pondrá un comando ahí. Si ya no la quieres, muévela o bórrala tú y luego instala.
general-cli-occupied-link = { $path } es un enlace a { $target }, que no es Eidola, así que Eidola no lo sustituirá. Si ya no lo quieres, muévelo o bórralo tú y luego instala.
general-cli-occupied-other = { $path } ya es algo que Eidola no puso ahí, así que Eidola no lo sustituirá.
general-cli-unreadable = No se pudo comprobar { $path } — { $reason }
general-cli-not-bundled = Solo disponible cuando Eidola se ejecuta como app instalada.
general-cli-translocated = macOS está ejecutando esta copia de Eidola desde una ubicación temporal, y un enlace a ella dejaría de funcionar. Mueve Eidola a tu carpeta Aplicaciones, ábrelo desde ahí y luego instala.
general-cli-missing = Esta copia de Eidola no incluye la herramienta de línea de comandos.
general-cli-install = Instalar
general-cli-install-admin = Instalar…
general-cli-install-name = Instalar el comando eidola en { $dir }
general-cli-update = Actualizar enlace
general-cli-update-admin = Actualizar enlace…
general-cli-update-name = Hacer que { $path } apunte a esta copia de Eidola
general-cli-installing = Instalando…
general-cli-waiting = Esperando la contraseña de un administrador…
general-cli-declined = No se cambió nada. Para instalarlo tú, ejecuta esto en Terminal:
general-cli-failed = No se pudo instalar el comando — { $reason }
general-cli-manual = Para instalarlo tú, ejecuta esto en Terminal:
general-cli-copy = Copiar
general-cli-copy-name = Copiar el comando de instalación
