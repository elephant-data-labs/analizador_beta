# Analizador de Beta

Proyecto Python independiente para estimar el **Beta** de una acción por
regresión OLS contra un índice de mercado, con validación cruzada
Cov/Var y diagnóstico de significancia estadística. Es un trabajo académico
de Análisis Financiero — no descarga ni procesa EEFF/XBRL: eso vive en otro
proyecto separado (`analizador_eeff_ifrs`), no se mezclan.

Caso de prueba incluido: **Aguas Andinas S.A. (AGUAS-A.SN)**, 2021–2025,
contra el **S&P/CLX IPSA**.

## Metodología

\[ R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_t \]

Retorno simple, \(R_t = P_t/P_{t-1} - 1\), sobre el precio ajustado por
dividendos (`Adj Close`). El retorno logarítmico está disponible como
alternativa (`kind="log"` en `returns.compute_returns`), pero no es el
default: no se cambia la metodología principal sin justificarlo.

El Beta se calcula de dos formas independientes y deben coincidir:

- **Regresión OLS** (`statsmodels`): pendiente de la recta.
- **Covarianza / Varianza**: \(\beta = \mathrm{Cov}(R_i,R_m)/\mathrm{Var}(R_m)\).

Todo el cálculo (retornos, regresión, p-values, R², intervalos de confianza,
gráfico, validación) es determinístico — sin IA. Ver el principio de diseño
más abajo.

## Fuente de datos — verificación previa a programar

Antes de escribir el código definitivo se verificó manualmente en el sitio
de cada proveedor (no se asumió nada):

### Acción: Yahoo Finance (`yfinance`)

- **Ticker verificado**: `AGUAS-A.SN`. El nemotécnico chileno (`AGUAS-A`) por
  sí solo *no* es el ticker de Yahoo: Yahoo requiere el sufijo `.SN` para
  instrumentos de la Bolsa de Santiago.
- **Disponibilidad histórica**: serie diaria continua desde enero de 2021
  hasta la fecha, sin huecos relevantes.
- **Precio usado**: `Adj Close` (ajustado por dividendos) como default,
  configurable a `Close` sin ajustar desde la interfaz.
- **Dividendos/splits**: se confirmaron pagos de dividendos dentro de la
  ventana 2021–2025 (`Close` y `Adj Close` difieren alrededor de esas
  fechas, como se espera). No se observaron splits.
- **Frecuencia disponible**: diaria, semanal y mensual, las tres verificadas
  en el sitio.

### Índice de mercado: S&P/CLX IPSA — pero descargado desde **Stooq**, no Yahoo

El **S&P/CLX IPSA** es el índice bursátil de referencia de la Bolsa de
Santiago (las acciones chilenas más grandes y líquidas) y es el proxy de
mercado estándar para estimar Beta de una acción chilena, tanto en la
literatura de finanzas corporativas como en la práctica local (es el mismo
criterio que usan los reportes de clasificadoras de riesgo y consultoras
locales). Por eso se eligió, y no arbitrariamente.

Sin embargo, al verificar la fuente **antes de programar** se encontró un
problema concreto: **Yahoo Finance no entrega una serie histórica utilizable
para `^IPSA`** — la página de históricos devuelve, como mucho, un único dato
suelto, no una serie diaria para ningún rango de fechas probado (2021, 2022,
o el último año). Se probó también el ticker alternativo `SPCLXIPSA.SN`, con
el mismo resultado.

Como reemplazo consistente se usa **Stooq** (`stooq.com`), que sí tiene la
serie diaria completa de `^IPSA` para toda la ventana 2021–2025 (verificado
directamente en su sitio). Stooq no entrega un "Adj Close" separado para
índices — se usa el cierre directamente, lo cual es razonable porque un
índice de precio no "paga" dividendos por sí mismo.

**Limitación documentada, no resuelta en esta versión**: el IPSA de Stooq es
un índice de *precio*, no de retorno total, mientras que el precio de la
acción sí se ajusta por dividendos (`Adj Close`). Esto introduce una
asimetría menor entre ambas series (el mercado "pierde" el dividend yield
que la acción sí refleja). No se corrige en la primera versión porque el
encargo pide explícitamente no sobre-construir antes de validar el flujo
básico; queda anotado como mejora futura.

**Nota técnica sobre Stooq — y qué pasa si bloquea la descarga**: su
endpoint de descarga exige un header `User-Agent` de navegador real (sin eso
responde `Access denied`), y en la práctica a veces devuelve una página de
verificación anti-bot (Cloudflare) en vez del CSV — algo que un
`requests.get` normal, con los headers que sea, no puede resolver por sí
solo (no ejecuta JavaScript). En pruebas repetidas el bloqueo resultó
persistente, no ocasional, así que no conviene depender de que Stooq
"vuelva a funcionar solo".

**Orden de prioridad real de la fuente del índice** (`pipeline._fetch_market_prices`):

1. **CSV manual** en `data/raw/<ticker>_manual.csv` (p. ej. `data/raw/ipsa_manual.csv`
   para `^IPSA`) — si el archivo existe, se usa directamente, sin llamar a
   Stooq ni a Yahoo para el índice. Es el mecanismo pensado para cuando Stooq
   está bloqueado: se baja el histórico una vez a mano y la app lo detecta
   solo en cada corrida siguiente (ver `data/raw/README.md`). **Ya viene
   incluido `data/raw/ipsa_manual.csv`** con el histórico diario del S&P/CLX
   IPSA 2021-01-04 a 2025-12-30 (1.243 observaciones), obtenido de
   Investing.com ante el bloqueo persistente de Stooq — así que, tal como
   está el proyecto, el caso Aguas Andinas 2021–2025 ya usa el índice real,
   no el replicado.
2. **Stooq**, la fuente configurada por defecto, si no hay CSV manual.
3. Si (2) falla y el respaldo está habilitado, `pipeline.run_beta_case`
   **arma automáticamente un índice de reemplazo** con
   `data_sources.build_replicated_index_prices`: una cartera equiponderada de
   las 27 componentes del IPSA verificadas en `catalog.py` (excluyendo la
   acción que se está analizando, para no comparar la acción contra un
   mercado que la incluye a ella misma), descargadas de Yahoo Finance — la
   misma fuente, ya probada, que la acción. Esto **no es el S&P/CLX IPSA
   oficial**: la interfaz lo deja explícito con una advertencia visible y en
   la fila "Fuente del índice" de la tabla de regresión, para que nunca
   quede ambiguo qué se usó de verdad (se puede desactivar con
   `allow_replicated_index_fallback=False` si se prefiere que falle en vez
   de usar el reemplazo).

Para otro índice o para refrescar el histórico del IPSA con fechas más
recientes, basta con reemplazar el CSV en `data/raw/` — no hace falta tocar
código.

### Por qué no se usó `^IPSA` de Yahoo ni un ETF en dólares (`ECH`)

Se evaluó `ECH` (iShares MSCI Chile ETF, cotiza en USD en EE.UU.) como
alternativa, porque sí tiene historial completo en Yahoo. Se descartó como
proxy principal: al estar denominado en USD, mezclaría el retorno accionario
chileno con el movimiento del tipo de cambio USD/CLP, lo que sesgaría el
Beta estimado con un factor de riesgo cambiario ajeno al CAPM local. Queda
como alternativa documentada, no como default.

## Frecuencia y período — cómo se documentan en la salida

La interfaz permite elegir frecuencia (diaria/semanal/mensual) y siempre
muestra explícitamente:

```text
Frecuencia: diaria/semanal/mensual
Período: fecha inicial – fecha final
```

para que nunca quede ambiguo qué se usó, incluso si se cambia desde el
default.

## Validación independiente (Beta OLS vs. Cov/Var)

Se calculan ambos y se muestran en una tabla junto con la diferencia. Si la
diferencia no es prácticamente cero (tolerancia `1e-6`, ver
`validation.DIFFERENCE_TOLERANCE`), la interfaz muestra una alerta — eso
indicaría un error de alineación de datos, no una discrepancia legítima
entre métodos (matemáticamente son la misma fórmula).

## Diagnóstico estadístico

La conclusión de significancia se decide solo por el p-value de Beta al 5%,
sin intervención de IA (`regression.significance_conclusion`).

## Días atípicos (outliers)

Pestaña aparte en la interfaz ("🔎 Días atípicos"), con su propio umbral
ajustable. Es una adaptación directa del proyecto final del Bootcamp de
Python de Carlos ("Detector de movimientos anómalos", Tkinter, Beta vs.
S&P 500 fijo) — la lógica es la misma, pero generalizada: acá el mercado de
comparación es el índice que defina `CaseParameters` para el caso (por
defecto S&P/CLX IPSA, no un mercado fijo) y la implementación queda separada
de la interfaz (`outliers.py`), igual que el resto del pipeline.

Metodología (`outliers.detect_outliers`, sin IA, determinística):

```
residuo_t = R_i,t - (alpha + beta * R_m,t)      # lo que el índice NO explica del retorno del día
z_t       = (residuo_t - media(residuo)) / desviación_estándar(residuo)
```

Un día se marca atípico si `|z_t|` supera el umbral (2,5 desviaciones
estándar por defecto, editable en la interfaz). No se vuelve a estimar
alpha/beta: se reutiliza exactamente la regresión ya calculada para el caso,
así que el resultado de outliers es siempre consistente con el Beta
mostrado en la pestaña de resultados.

La pestaña muestra: cantidad de días atípicos (y cuántos positivos/negativos
— sobre y bajo lo que predice el modelo), una tabla ordenada por `|z|`
descendente con columnas explicadas (formato porcentual, tooltips por
columna vía `st.column_config`, y una columna "Dirección" con 🔼/🔽), y el
precio de la acción (a la misma frecuencia del análisis) con esos días
marcados encima — todo con **Plotly** (gráfico interactivo: zoom y hover
con el detalle de cada punto), sin IA.

## Principio de diseño

Sin IA para: descargar datos, calcular retornos, calcular Beta, la
regresión, p-values, R², intervalos de confianza, los gráficos, la
validación, ni la detección de días atípicos. Todo eso es Python puro
(`pandas`, `numpy`, `statsmodels`, `plotly`) reproducible con las
versiones fijadas en `requirements.txt`.

Las explicaciones "en lenguaje simple" que se muestran en el Diagnóstico
estadístico (qué significa el Beta, el R², el p-value y el intervalo de
confianza) también son deterministas: `regression.beta_interpretation`,
`r_squared_interpretation`, `p_value_interpretation` y `ci_interpretation`
arman el texto con tramos de umbrales fijos en el código (una tabla de
casos, igual que `significance_conclusion`), no con un modelo de lenguaje —
el mismo principio de diseño se mantiene también para esta parte.

## Alcance de esta primera versión

```text
Precio acción
      +
Retornos acción + Retornos índice
      v
Regresión OLS
      v
Beta, Alpha, estadísticos
      v
Validación Cov/Var
      v
Gráfico  +  Días atípicos (residuos vs. z-score)
```

**No implementado a propósito todavía** (ver `src/beta_analyzer/comparables.py`,
que deja el contrato de las funciones documentado pero sin implementar):
comparables, desapalancamiento/reapalancamiento de Beta (Hamada), estructura
de capital objetivo, CAPM, costo patrimonial, WACC.

## Combo box de empresas

La interfaz trae un selector ("Empresa") con las **27 componentes actuales
del S&P/CLX IPSA** según la propia página de Yahoo Finance
(`/quote/%5EIPSA/components/`) — el índice no tiene un número fijo de 30,
varía con los rebalanceos periódicos; a la fecha de esta verificación
(31-08-2026) son 27. Cada ticker se comprobó además uno por uno, mismo
procedimiento que con Aguas Andinas: existe en Yahoo Finance, cotiza en CLP
en la Bolsa de Santiago, y tiene historial diario desde enero de 2021. Al
elegir una empresa se autocompleta el ticker. El catálogo vive en
`src/beta_analyzer/catalog.py`:

| Empresa | Ticker | Sector |
|---|---|---|
| Sociedad Química y Minera de Chile (SQM-B) | `SQM-B.SN` | Minería / litio |
| Banco de Chile | `CHILE.SN` | Banca |
| Falabella S.A. | `FALABELLA.SN` | Retail |
| Banco Santander-Chile | `BSANTANDER.SN` | Banca |
| Banco de Crédito e Inversiones (BCI) | `BCI.SN` | Banca |
| LATAM Airlines Group S.A. | `LTM.SN` | Aerolíneas |
| Enel Américas S.A. | `ENELAM.SN` | Energía eléctrica |
| Plaza S.A. (Mallplaza) | `MALLPLAZA.SN` | Retail — centros comerciales |
| Empresas Copec S.A. | `COPEC.SN` | Energía / forestal |
| Cencosud S.A. | `CENCOSUD.SN` | Retail |
| Enel Chile S.A. | `ENELCHILE.SN` | Energía eléctrica |
| Embotelladora Andina S.A. | `ANDINA-B.SN` | Bebidas |
| Parque Arauco S.A. | `PARAUCO.SN` | Retail — centros comerciales |
| Empresas CMPC S.A. | `CMPC.SN` | Forestal / celulosa |
| Colbún S.A. | `COLBUN.SN` | Energía eléctrica |
| Inversiones La Construcción S.A. (ILC) | `ILC.SN` | Holding — seguros/salud |
| Compañía Sud Americana de Vapores S.A. | `VAPORES.SN` | Naviera |
| Compañía Cervecerías Unidas S.A. (CCU) | `CCU.SN` | Bebidas |
| Engie Energía Chile S.A. | `ECL.SN` | Energía eléctrica |
| Aguas Andinas S.A. | `AGUAS-A.SN` | Utilities — agua |
| Empresa Nacional de Telecomunicaciones S.A. (Entel) | `ENTEL.SN` | Telecomunicaciones |
| Inversiones Aguas Metropolitanas S.A. (IAM) | `IAM.SN` | Utilities — agua |
| CAP S.A. | `CAP.SN` | Minería / acero |
| Ripley Corp S.A. | `RIPLEY.SN` | Retail |
| SalfaCorp S.A. | `SALFACORP.SN` | Construcción / inmobiliaria |
| Viña Concha y Toro S.A. | `CONCHATORO.SN` | Bebidas — vinos |
| Sonda S.A. | `SONDA.SN` | Tecnología / servicios TI |

Nota: `LTM.SN` (LATAM Airlines) muestra una caída fuerte de precio entre
2021 y hoy — es consistente con la reestructuración societaria de la
empresa en ese período (Chapter 11, 2020–2022), no un error de la fuente.

**Agregar una empresa al catálogo permanente**: no basta con adivinar el
ticker. Repetir la verificación (existe en Yahoo Finance, moneda CLP,
historial desde la fecha que se necesite) y agregar una `CompanyEntry` en
`catalog.py`. Hay un test (`tests/test_catalog.py`) que solo revisa
duplicados y forma del ticker, no reemplaza esa verificación manual.

## Combo box de índices

Igual que con la empresa, el índice de mercado se elige de un selector en
vez de escribir nombre + ticker + fuente a mano cada vez (declutter del
panel de parámetros). Dos presets verificados, en `catalog.KNOWN_INDICES`:

| Índice | Ticker | Fuente | Motivo |
|---|---|---|---|
| S&P/CLX IPSA | `^IPSA` | Stooq (+ CSV manual + índice replicado, ver más arriba) | Yahoo Finance no tiene serie histórica utilizable para este ticker |
| S&P 500 | `^GSPC` | Yahoo Finance | Historial diario completo y estable, misma fuente que la acción — no hace falta Stooq |

Cada preset trae su `source` (`config.IndexSource`, `"stooq"` o `"yahoo"`),
que decide en `pipeline._fetch_market_prices` cómo se descarga —
independiente del CSV manual, que manda primero sea cual sea la fuente. La
última opción del selector ("Otro índice") habilita campos manuales de
nombre, ticker y fuente, para cualquier otro índice no listado — sigue
siendo cierto que el proyecto es reutilizable sin tocar código, ver más
abajo.

### Otra empresa (fuera del catálogo, incluidas las que no son chilenas)

La última opción del selector ("Otra empresa — buscar en Yahoo Finance")
funciona igual que "Otro índice" en la columna de la derecha: aparecen al
tiro dos campos — nombre y ticker — para escribir directamente cualquier
empresa que exista en Yahoo Finance, no solo del catálogo IPSA ni solo
chilenas (por ejemplo `GOOGL` para Alphabet/Google o `AAPL` para Apple, sin
sufijo; una acción chilena fuera del catálogo suele llevar el sufijo `.SN`).
No hace falta buscar primero.

Como ayuda opcional, más abajo hay un desplegable "¿No conoce el ticker
exacto? Buscar por nombre en Yahoo Finance": se escribe el nemotécnico o el
nombre y el botón **Buscar en Yahoo Finance** consulta directamente el
buscador de Yahoo (`beta_analyzer.data_sources.search_tickers`, sobre
`yfinance.Search`). Si lo encuentra, muestra los resultados (con la Bolsa de
Santiago priorizada, sin descartar homónimos de otras bolsas — por ejemplo,
"Banco de Chile" también tiene un ADR `BCH` en NYSE); el botón **Usar este
resultado** completa los campos de nombre y ticker de arriba con el
resultado elegido. Se verificó a mano antes de programar que tanto buscar
por nemotécnico ("ENTEL", "AGUAS-A") como por nombre ("banco de chile",
"Google") funcionan contra el buscador real de Yahoo.

## Reutilizar el proyecto para otra empresa

Nada queda hardcodeado a Aguas Andinas: `src/beta_analyzer/config.py` define
`CaseParameters` (empresa, ticker, índice, ventana, frecuencia, precio) como
un solo objeto explícito, y la interfaz permite editarlo directamente. Para
otra acción chilena alcanza con cambiar el ticker (verificando primero, igual
que se hizo acá, que Yahoo lo reconozca con el sufijo `.SN` correcto) y las
fechas.

## Ejecutar

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La navegación es la misma que el proyecto EEFF hermano: botones tipo radio
horizontales (no pestañas), con **Inicio** como portada que se muestra
primero y bloquea el resto de la app hasta que se elige otra hoja. Misma
identidad visual también (logo, tarjeta de portada con autor y credencial,
estilos) — ambos proyectos se ven como un mismo laboratorio.

1. **Inicio** — portada (autor, credencial, Elephant Data Labs), qué hace la
   app, cómo usarla en 3 pasos, la nota de fuente de datos del índice
   (Stooq/CSV manual/replicado, por qué no se usa Investing.com en vivo),
   el estado del catálogo (empresas e índices disponibles) y qué hay en
   cada hoja. Para leer antes de tocar los parámetros.
2. **Analizador de Beta** — parámetros precargados con el caso Aguas
   Andinas 2021–2025; presione **CALCULAR BETA** para correr el pipeline
   completo (Beta, regresión, diagnóstico con explicación en lenguaje
   simple, gráfico interactivo, validación).
3. **Días atípicos** — reutiliza el último cálculo de la hoja anterior
   (queda en `st.session_state`, no vuelve a descargar ni a recalcular la
   regresión); muestra un aviso si todavía no se ha calculado ningún caso.

## Estructura

```text
app.py                              interfaz Streamlit (hojas tipo radio: Inicio, Analizador de Beta, Días atípicos)
src/beta_analyzer/config.py         parámetros del caso (empresa, ticker, índice + fuente, ventana, frecuencia)
src/beta_analyzer/catalog.py        catálogo de empresas chilenas e índices (IPSA/S&P 500) verificados, para los combo box
src/beta_analyzer/data_sources.py   descarga de precios (Yahoo Finance / Stooq), búsqueda de tickers, índice replicado de respaldo, CSV manual
src/beta_analyzer/returns.py        retornos y alineación acción/índice por fecha y frecuencia
src/beta_analyzer/regression.py     regresión OLS determinística (statsmodels) + interpretaciones en lenguaje simple (Beta, R², p-value, IC)
src/beta_analyzer/validation.py     validación independiente Cov/Var
src/beta_analyzer/outliers.py       días atípicos por z-score del residuo de la regresión (adaptado del Bootcamp)
src/beta_analyzer/plotting.py       gráficos interactivos (Plotly): dispersión con recta de regresión + precio con días atípicos marcados
src/beta_analyzer/pipeline.py       orquesta el flujo completo; CSV manual > Yahoo directo o Stooq (según índice) > índice replicado si Stooq falla
src/beta_analyzer/comparables.py    contrato futuro (Hamada) — sin implementar
data/raw/                           respaldo manual de CSV si falla la descarga automática
tests/                              pruebas del motor (retornos, regresión, validación, outliers)
```

## Pruebas

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest
```

Las pruebas usan datos sintéticos con semilla fija, y red simulada
(`monkeypatch`) para todo lo que llama a Yahoo Finance / Stooq — no
requieren conexión a internet. Verifican, entre otras cosas: la fórmula de
retorno simple, la alineación por fecha, que la regresión recupere un Beta
conocido dentro de un margen de error muestral razonable, que el Beta de la
regresión coincida con un cálculo manual independiente vía `numpy`, que
Beta OLS y Beta Cov/Var coincidan, que el buscador de tickers priorice
Bolsa de Santiago, y que el pipeline arme el índice replicado automáticamente
cuando Stooq falla — excluyendo siempre la acción en análisis de ese índice
de reemplazo. Además se corrió el pipeline completo de punta a punta
(`data_sources` → `returns` → `regression` → `validation` → `plotting`) con
datos sintéticos en frecuencia diaria, semanal y mensual antes de entregar
el proyecto, confirmando que las tres frecuencias producen una diferencia
Beta OLS vs. Cov/Var del orden de `1e-16` (ruido de punto flotante, no un
error real).
