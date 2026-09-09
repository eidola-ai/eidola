# Ajustes ▸ Proxy — el proxy de inferencia local (src/proxy_settings.rs), español.

proxy-lead = Permite que otras herramientas de este ordenador usen los modelos a los que llegas con Eidola. Hablan la API de OpenAI; cada petición sigue pasando por Eidola — atestiguada, pagada desde tu cartera y anotada en el Registro.

proxy-serve = Atender peticiones
proxy-serve-name = Atender peticiones locales a través de Eidola

proxy-listening = Escuchando en { $address }
proxy-stopped = Sin escuchar
proxy-listen-failed = No se pudo escuchar en esa dirección — { $reason }

proxy-exposed-warning = Esta dirección es accesible desde tu red, y el proxy todavía no cifra nada. Todo lo que pase por él — tus indicaciones y las respuestas — viaja en claro.

proxy-address = Dirección
proxy-address-name = La dirección IP en la que escucha el proxy
proxy-port-name = El puerto en el que escucha el proxy
proxy-binding-change = Cambiar…
proxy-binding-save = Guardar
proxy-binding-cancel = Cancelar

proxy-backends = Proveedores
proxy-backends-note = Solo se puede llegar a lo que marques aquí. A una herramienta que nombre cualquier otra cosa se le dice que el modelo no existe.
proxy-backend-name = Ofrecer { $backend } a través del proxy
proxy-backends-empty = Todavía no hay ningún proveedor configurado.

proxy-exposure = Modelos en el dispositivo
proxy-exposure-loaded = Solo los cargados
proxy-exposure-downloaded = Todos los descargados
proxy-exposure-note = «Todos los descargados» arranca un motor en la primera petición que nombre un modelo, lo que tarda un rato y ocupa memoria. «Solo los cargados» ofrece lo que ya está en marcha.

proxy-keys = Claves de API
proxy-keys-note = Una herramienta envía su clave como token bearer. Eidola solo guarda un hash, así que una clave se muestra una vez y no se puede volver a mostrar.
proxy-keys-empty = Todavía no hay claves — nada puede llegar al proxy hasta que crees una.
proxy-key-label-placeholder = ¿Qué va a usar esta clave?
proxy-key-create = Generar una clave
proxy-key-creating = Generando…
proxy-key-show-first = Copia la clave de arriba y pulsa Listo primero.
proxy-key-revoked = revocada
proxy-key-unused = sin usar
proxy-key-used = usada
proxy-key-revoke = Revocar
proxy-key-revoke-name = Revocar la clave llamada { $label }

proxy-key-minted = Cópiala ahora. Eidola solo ha guardado su hash y no puede volver a mostrarla.
proxy-key-copy = Copiar
proxy-key-done = Hecho

proxy-failed = No se pudieron leer los ajustes del proxy.
proxy-retry = Reintentar
proxy-keys-failed = No se pudieron listar las claves de API.
proxy-backends-failed = No se pudo leer el registro de proveedores.
proxy-loading = Cargando…
proxy-stale = No se pudo actualizar — se muestra la última respuesta.
