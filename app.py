# -*- coding: utf-8 -*-
"""
MPD · Grupo 2 — Estudio de modelos para una pizzería local
==========================================================
Aplicación única en Streamlit que integra tres modelos conectados entre sí:

    💲 Elasticidad de precios  →  📈 Demanda (series de tiempo)  →  📦 Inventario (revisión continua)

Cadena de dependencias (todo se recalcula al mover cualquier parámetro):
  1. El escenario de precios (Δ%) y la elasticidad ε generan un multiplicador de demanda.
  2. El pronóstico semanal de series de tiempo (Random Forest con rezagos) fija el nivel
     de demanda esperado respecto al histórico (factor g).
  3. Ambos factores escalan el consumo diario de insumos, y con él SS, ROP, EOQ y costos.

Ejecución local:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# 0. CONFIGURACIÓN GENERAL Y ESTILO
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MPD · Pizzería: Elasticidad + Demanda + Inventario",
                   page_icon="🍕", layout="wide")

DATA_FILE = "pizza_sales_ordenado.csv"
PALETA = ["#C1272D", "#F4A300", "#6B8E23", "#8B4513", "#F5D061"]
COLOR_BASE, COLOR_ESC = "#7A2E0E", "#C1272D"

VARIABLES = ["precio", "semana", "mes", "t", "sin_doy", "cos_doy",
             "lag_1", "lag_2", "lag_3", "lag_4", "media_4"]
Z_TABLA = {"90 %": 1.282, "95 %": 1.645, "97.5 %": 1.960, "99 %": 2.326}
OPC_TOP5 = "Top 5 agregado"
M_RF, M_PCA = "Random Forest (propuesto)", "Random Forest + PCA"
M_NAIVE, M_MA4 = "Ingenuo (última semana)", "Media móvil 4 semanas"
M_AUTO = "Automático (menor MAPE en backtesting)"
COLUMNAS_REQ = ["order_id", "quantity", "order_date", "unit_price", "total_price",
                "pizza_size", "pizza_category", "pizza_ingredients", "pizza_name"]

st.markdown(
    """
    <style>
    .stApp { background-color: #FFF8EE; }
    h1, h2, h3 { color: #7A2E0E; font-family: 'Trebuchet MS', sans-serif; }
    h1 { border-bottom: 4px solid #C1272D; padding-bottom: 8px; }
    [data-testid="stSidebar"] { background-color: #FBE7C8; }
    [data-testid="stMetricValue"] { color: #C1272D; }
    hr { border-top: 2px dashed #F4A300; }
    </style>
    """,
    unsafe_allow_html=True,
)


def estilo(ax):
    ax.set_facecolor("#FFF8EE")
    ax.figure.set_facecolor("#FFF8EE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#7A2E0E")
    return ax


def mostrar(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


# ----------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# ----------------------------------------------------------------------------
def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    faltan = [c for c in COLUMNAS_REQ if c not in df.columns]
    if faltan:
        raise ValueError("Al CSV le faltan las columnas: " + ", ".join(faltan))
    txt = df["order_date"].astype(str).str.strip().str.replace("-", "/", regex=False)
    fechas = pd.to_datetime(txt, format="%d/%m/%Y", errors="coerce")
    if fechas.isna().mean() > 0.5:  # por si el archivo viniera como mes/día/año
        fechas = pd.to_datetime(txt, format="%m/%d/%Y", errors="coerce")
    df["order_date"] = fechas
    for c in ["quantity", "unit_price", "total_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["pizza_size", "pizza_category", "pizza_ingredients", "pizza_name"]:
        df[c] = df[c].astype(str).str.strip()
    df = df.dropna(subset=["order_date", "quantity", "unit_price", "total_price"])
    return df.reset_index(drop=True)


def _leer_csv(texto: str) -> pd.DataFrame:
    primera = texto.split("\n", 1)[0]
    sep = ";" if primera.count(";") > primera.count(",") else ","
    return _limpiar(pd.read_csv(io.StringIO(texto), sep=sep))


@st.cache_data(show_spinner="Cargando datos…")
def cargar_desde_ruta(ruta: str, firma: float) -> pd.DataFrame:
    return _leer_csv(Path(ruta).read_text(encoding="utf-8-sig", errors="replace"))


@st.cache_data(show_spinner="Cargando datos…")
def cargar_desde_bytes(contenido: bytes) -> pd.DataFrame:
    return _leer_csv(contenido.decode("utf-8-sig", errors="replace"))


def obtener_datos(subido) -> pd.DataFrame:
    try:
        if subido is not None:
            return cargar_desde_bytes(subido.getvalue())
        ruta = Path(__file__).resolve().parent / DATA_FILE
        if ruta.exists():
            return cargar_desde_ruta(str(ruta), ruta.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer el CSV: {exc}")
        st.stop()
    st.error(f"No se encontró «{DATA_FILE}» junto a app.py. Súbalo desde el panel lateral (📂 Datos).")
    st.stop()


# ----------------------------------------------------------------------------
# 2. MODELO DE ELASTICIDAD DE PRECIOS (OLS log-log + benchmarks)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def agg_variantes(_df, clave: str) -> pd.DataFrame:
    agg = (_df.groupby(["pizza_name", "pizza_category", "pizza_size"])
           .agg(precio_promedio=("unit_price", "mean"), cantidad_total=("quantity", "sum"),
                ingreso_total=("total_price", "sum")).reset_index())
    agg = agg[(agg["precio_promedio"] > 0) & (agg["cantidad_total"] > 0)].reset_index(drop=True)
    agg["ln_precio"] = np.log(agg["precio_promedio"])
    agg["ln_cantidad"] = np.log(agg["cantidad_total"])
    return agg


def disenio(sub: pd.DataFrame, tamano=True, categoria=True) -> pd.DataFrame:
    X = pd.DataFrame({"Intercepto": 1.0, "ln_precio": sub["ln_precio"].values}, index=sub.index)
    partes = [X]
    if tamano and sub["pizza_size"].nunique() > 1:
        partes.append(pd.get_dummies(sub["pizza_size"], prefix="tamaño", drop_first=True))
    if categoria and sub["pizza_category"].nunique() > 1:
        partes.append(pd.get_dummies(sub["pizza_category"], prefix="categoría", drop_first=True))
    return pd.concat(partes, axis=1).astype(float)


def ajustar_ols(X: pd.DataFrame, y: np.ndarray):
    """OLS con inferencia clásica (equivale a statsmodels.OLS). None si no es identificable."""
    Xv, y = X.values.astype(float), np.asarray(y, dtype=float)
    n, k = Xv.shape
    gl = n - k
    if gl <= 0 or np.linalg.matrix_rank(Xv) < k:
        return None
    beta = np.linalg.lstsq(Xv, y, rcond=None)[0]
    res = y - Xv @ beta
    rss, tss = float(res @ res), float(((y - y.mean()) ** 2).sum())
    cov = (rss / gl) * np.linalg.inv(Xv.T @ Xv)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), gl)
    tc = stats.t.ppf(0.975, gl)
    tabla = pd.DataFrame({"coef": beta, "error_std": se, "t": t, "p_valor": p,
                          "IC95_inf": beta - tc * se, "IC95_sup": beta + tc * se}, index=X.columns)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    f = ((tss - rss) / (k - 1)) / (rss / gl) if k > 1 and rss > 0 else np.nan
    return {"tabla": tabla, "eps": float(beta[1]), "p": float(p[1]), "r2": r2,
            "r2_aj": 1 - (1 - r2) * (n - 1) / gl, "n": n, "k": k,
            "F": f, "p_F": float(stats.f.sf(f, k - 1, gl)) if np.isfinite(f) else np.nan}


def elasticidad_subconjunto(sub: pd.DataFrame):
    if len(sub) < 4 or sub["precio_promedio"].nunique() < 2:
        return None
    return ajustar_ols(disenio(sub), sub["ln_cantidad"].values)


@st.cache_data(show_spinner=False)
def modelo_global_y_segmentos(_agg, clave: str):
    g = ajustar_ols(disenio(_agg), _agg["ln_cantidad"].values)
    filas_c, filas_t = [], []
    for c, sub in _agg.groupby("pizza_category"):
        r = elasticidad_subconjunto(sub)
        filas_c.append({"Categoría": c, "N variantes": len(sub),
                        "Elasticidad": r["eps"] if r else np.nan,
                        "p-valor": r["p"] if r else np.nan, "R²": r["r2"] if r else np.nan,
                        "Estado": "estimada" if r else "no identificable"})
    for s, sub in _agg.groupby("pizza_size"):
        r = elasticidad_subconjunto(sub)
        filas_t.append({"Tamaño": s, "N variantes": len(sub),
                        "Elasticidad": r["eps"] if r else np.nan,
                        "p-valor": r["p"] if r else np.nan, "R²": r["r2"] if r else np.nan,
                        "Estado": "estimada" if r else "no identificable"})
    return g, pd.DataFrame(filas_c).set_index("Categoría"), pd.DataFrame(filas_t).set_index("Tamaño")


@st.cache_data(show_spinner="Entrenando benchmark (OLS vs Ridge vs Random Forest)…")
def benchmark_elasticidad(_df, clave: str):
    d = _df.assign(dow=_df["order_date"].dt.dayofweek, mes=_df["order_date"].dt.month)
    cel = (d.groupby(["pizza_name", "pizza_category", "pizza_size", "dow", "mes"])
           .agg(q=("quantity", "sum"), p=("unit_price", "mean")).reset_index())
    cel = cel[cel["q"] > 0].reset_index(drop=True)
    cat = pd.get_dummies(cel[["pizza_category", "pizza_size"]], drop_first=True)
    cal = pd.get_dummies(cel[["dow", "mes"]].astype(str), drop_first=True)
    X_lin = pd.concat([pd.DataFrame({"ln_p": np.log(cel["p"])}), cat, cal], axis=1).astype(float)
    y_log, y = np.log(cel["q"].values), cel["q"].values.astype(float)
    X_rf = pd.concat([cel[["p"]], pd.get_dummies(cel[["pizza_category", "pizza_size"]]),
                      cel[["dow", "mes"]]], axis=1).astype(float)
    kf = KFold(5, shuffle=True, random_state=42)

    def met(real, pred):
        ss_res, ss_tot = ((real - pred) ** 2).sum(), ((real - real.mean()) ** 2).sum()
        return (float(np.mean(np.abs(real - pred))), float(np.sqrt(np.mean((real - pred) ** 2))),
                float(1 - ss_res / ss_tot))

    filas = []
    for nombre, mod in [("OLS log-log", LinearRegression()), ("Ridge log-log (α=1)", Ridge(alpha=1.0))]:
        pred = np.exp(cross_val_predict(mod, X_lin, y_log, cv=kf))
        mae, rmse, r2 = met(y, pred)
        coef = mod.fit(X_lin, y_log).coef_[0]
        filas.append({"Modelo": nombre, "Elasticidad (ε)": coef, "MAE (uds)": mae, "RMSE (uds)": rmse, "R² (uds)": r2})
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    pred = cross_val_predict(rf, X_rf, y, cv=kf)
    mae, rmse, r2 = met(y, pred)
    rf.fit(X_rf, y)
    arr, aba = X_rf.copy(), X_rf.copy()
    arr["p"], aba["p"] = X_rf["p"] * 1.1, X_rf["p"] * 0.9
    eps_rf = np.log(max(rf.predict(arr).sum(), 1e-9) / max(rf.predict(aba).sum(), 1e-9)) / np.log(1.1 / 0.9)
    filas.append({"Modelo": "Random Forest (benchmark)", "Elasticidad (ε)": eps_rf,
                  "MAE (uds)": mae, "RMSE (uds)": rmse, "R² (uds)": r2})
    return pd.DataFrame(filas).set_index("Modelo"), len(cel)


def construir_elasticidades(modo, eps_manual, forzar, eps_global, tabla_cat, cats):
    """Devuelve (dict categoría→ε usada, lista de categorías que cayeron al valor global)."""
    eps, respaldo = {}, []
    for c in cats:
        if modo.startswith("Global"):
            eps[c] = eps_global
        elif modo.startswith("Manual"):
            eps[c] = eps_manual
        else:
            e = tabla_cat["Elasticidad"].get(c, np.nan)
            if not np.isfinite(e) or (forzar and e >= 0):
                eps[c] = eps_global
                respaldo.append(c)
            else:
                eps[c] = float(e)
    return eps, respaldo


def mult_demanda(pesos: pd.Series, deltas: dict, eps: dict, extra=0.0) -> float:
    """Σ w_c · (1+Δ_c)^(ε_c+extra). extra=0 → unidades; extra=1 → ingresos."""
    return float(sum(w * (1 + deltas[c] / 100.0) ** (eps[c] + extra) for c, w in pesos.items()))


def mult_insumos(shares: pd.DataFrame, deltas: dict, eps: dict) -> pd.Series:
    f = pd.Series({c: (1 + deltas[c] / 100.0) ** eps[c] for c in shares.columns})
    return (shares * f).sum(axis=1)


def simular_precio(df_seg: pd.DataFrame, eps: dict, grilla) -> pd.DataFrame:
    base = df_seg.groupby("pizza_category").agg(q=("quantity", "sum"), r=("total_price", "sum"))
    q0, r0 = base["q"].sum(), base["r"].sum()
    filas = []
    for d in grilla:
        k = 1 + d / 100.0
        q = sum(base.loc[c, "q"] * k ** eps[c] for c in base.index)
        r = sum(base.loc[c, "r"] * k ** (1 + eps[c]) for c in base.index)
        filas.append({"Δ precio (%)": d, "Precio medio ($)": r / q, "Unidades": q,
                      "Ingreso ($)": r, "Δ ingreso (%)": (r / r0 - 1) * 100,
                      "Δ unidades (%)": (q / q0 - 1) * 100})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# 3. MODELO DE DEMANDA CON SERIES DE TIEMPO
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calcular_top5(_df, clave: str) -> pd.Series:
    return _df.groupby("pizza_name")["quantity"].sum().sort_values(ascending=False).head(5)


MIN_DIAS_BORDE = 4  # semanas de inicio/fin de año con menos días se descartan


def agregar_semanal(d: pd.DataFrame):
    """Serie semanal SIN huecos.

    Antes se eliminaban todas las semanas con < 7 días de venta, incluidas semanas del
    medio del año: eso rompía la continuidad (lag_1 pasaba a ser la semana de hace 2-3
    semanas y t dejaba de medir tiempo real). Ahora:
      · solo se recortan las semanas incompletas de los BORDES (inicio/fin) con < 4 días;
      · las semanas interiores incompletas se mantienen y se normalizan a 7 días
        (unidades × 7 / días con venta), para que todas midan el mismo ritmo semanal.
    """
    tmp = d.set_index("order_date")
    serie = tmp.resample("W").agg(ventas=("total_price", "sum"), pedidos=("order_id", "nunique"),
                                  unidades=("quantity", "sum"), precio=("unit_price", "mean"))
    dias = (d.assign(dia=d["order_date"].dt.normalize()).drop_duplicates("dia")
            .set_index("dia").resample("W").size())
    serie["dias_con_venta"] = dias.reindex(serie.index).fillna(0).values
    ok = serie["dias_con_venta"] >= MIN_DIAS_BORDE
    if not ok.any():
        return serie.iloc[0:0].reset_index().rename(columns={"order_date": "fecha"}), len(serie), 0
    primera, ultima = ok.idxmax(), ok[::-1].idxmax()
    rec = serie.loc[primera:ultima].copy()
    n_desc = len(serie) - len(rec)
    n_ajust = int((rec["dias_con_venta"] < 7).sum())
    k = 7 / rec["dias_con_venta"].replace(0, np.nan)
    for c in ["ventas", "pedidos", "unidades"]:
        rec[c] = rec[c] * k
    cols = ["ventas", "pedidos", "unidades", "precio"]
    rec[cols] = rec[cols].interpolate(limit_direction="both")  # semana sin ventas (si existiera)
    serie = rec.drop(columns="dias_con_venta").reset_index().rename(columns={"order_date": "fecha"})
    return serie, n_desc, n_ajust


def crear_features(serie: pd.DataFrame) -> pd.DataFrame:
    out = serie.copy()
    out["t"] = np.arange(len(out))
    doy = out["fecha"].dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    out["semana"] = out["fecha"].dt.isocalendar().week.astype(int)
    out["mes"] = out["fecha"].dt.month
    for k in range(1, 5):
        out[f"lag_{k}"] = out["unidades"].shift(k)
    out["media_4"] = out["unidades"].shift(1).rolling(4).mean()
    return out.dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def preparar_serie(_df, clave: str, productos: tuple, seleccion: str):
    d = _df[_df["pizza_name"].isin(productos)] if seleccion == OPC_TOP5 else _df[_df["pizza_name"] == seleccion]
    serie, n_desc, n_ajust = agregar_semanal(d)
    feat = crear_features(serie)
    pu = d.groupby("pizza_category")["quantity"].sum()
    pr = d.groupby("pizza_category")["total_price"].sum()
    precio_unit = float(d["total_price"].sum() / d["quantity"].sum())
    cols = ["pizza_name", "pizza_category", "pizza_size"]
    return serie, feat, (n_desc, n_ajust), pu / pu.sum(), pr / pr.sum(), precio_unit, d[cols].copy()


def construir_modelo(nombre: str, n_est: int):
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=8, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    if nombre == M_PCA:  # reducción de dimensiones (retroalimentación del docente)
        return make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"), rf)
    return rf


@st.cache_data(show_spinner=False)
def ajustar_modelo(df_feat, nombre: str, n_est: int):
    m = construir_modelo(nombre, n_est)
    m.fit(df_feat[VARIABLES], df_feat["unidades"])
    return m


def _mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs((real - pred) / np.maximum(real, 1))) * 100)


@st.cache_data(show_spinner="Ejecutando backtesting…")
def backtest(df_feat, nombre: str, n_est: int, horizonte=4, paso=4):
    min_train = 28 if len(df_feat) >= 36 else max(10, len(df_feat) // 2)
    fechas, reales, preds = [], [], []
    for corte in range(min_train, len(df_feat) - horizonte + 1, paso):
        tr, te = df_feat.iloc[:corte], df_feat.iloc[corte:corte + horizonte]
        if nombre == M_NAIVE:
            p = te["lag_1"].values
        elif nombre == M_MA4:
            p = te["media_4"].values
        else:
            m = construir_modelo(nombre, n_est).fit(tr[VARIABLES], tr["unidades"])
            p = m.predict(te[VARIABLES])
        preds.extend(np.asarray(p, float)); reales.extend(te["unidades"].tolist()); fechas.extend(te["fecha"].tolist())
    preds, reales = np.array(preds), np.array(reales, float)
    mae = float(np.mean(np.abs(reales - preds)))
    rmse = float(np.sqrt(np.mean((reales - preds) ** 2)))
    return mae, rmse, _mape(reales, preds), pd.DataFrame({"fecha": fechas, "real": reales, "prediccion": preds.round(1)})


def pronosticar(df_feat, modelo, horizonte: int) -> pd.DataFrame:
    """Pronóstico recursivo: cada predicción alimenta los rezagos de la semana siguiente."""
    hist = df_feat["unidades"].tolist()
    ultima, t_act = df_feat["fecha"].iloc[-1], int(df_feat["t"].iloc[-1])
    precio_ref = float(df_feat["precio"].tail(4).mean())
    fechas, preds = [], []
    for paso in range(1, horizonte + 1):
        f = ultima + pd.Timedelta(weeks=paso)
        t_act += 1
        doy = f.dayofyear
        fila = pd.DataFrame([{"precio": precio_ref, "semana": int(f.isocalendar()[1]), "mes": f.month,
                              "t": t_act, "sin_doy": np.sin(2 * np.pi * doy / 365.25),
                              "cos_doy": np.cos(2 * np.pi * doy / 365.25),
                              "lag_1": hist[-1], "lag_2": hist[-2], "lag_3": hist[-3], "lag_4": hist[-4],
                              "media_4": float(np.mean(hist[-4:]))}])[VARIABLES]
        y = float(modelo.predict(fila)[0])
        hist.append(y); fechas.append(f); preds.append(y)
    return pd.DataFrame({"fecha": fechas, "prediccion": np.round(preds, 1)})


@st.cache_data(show_spinner=False)
def importancias_rf(df_feat, n_est: int) -> pd.Series:
    m = construir_modelo(M_RF, n_est).fit(df_feat[VARIABLES], df_feat["unidades"])
    return pd.Series(m.feature_importances_, index=VARIABLES).sort_values(ascending=False)


@st.cache_data(show_spinner=False)
def info_pca(df_feat):
    Xs = StandardScaler().fit_transform(df_feat[VARIABLES])
    p = PCA().fit(Xs)
    return p.explained_variance_ratio_, pd.DataFrame(p.components_[:3].T, index=VARIABLES, columns=["CP1", "CP2", "CP3"])


def acf(x, nlags=12):
    x = np.asarray(x, float) - np.mean(x)
    den = (x ** 2).sum()
    return np.array([(x[:-k] * x[k:]).sum() / den for k in range(1, nlags + 1)])


def descriptivos(df, columnas, moda_en=None) -> pd.DataFrame:
    """moda_en: columnas donde la moda tiene sentido (las demás muestran vacío, p. ej. t o
    sin_doy, donde todos los valores son distintos y la 'moda' sería solo el mínimo)."""
    filas = []
    for c in columnas:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        moda = s.mode() if (moda_en is None or c in moda_en) else pd.Series(dtype=float)
        filas.append({"Variable": c, "Media": s.mean(), "Moda": moda.iloc[0] if len(moda) else np.nan,
                      "Desv. estándar": s.std(), "Mínimo": s.min(), "Máximo": s.max()})
    return pd.DataFrame(filas).set_index("Variable").round(3)


# ----------------------------------------------------------------------------
# 4. MODELO DE INVENTARIO (REVISIÓN CONTINUA, s-Q)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Procesando consumo de insumos…")
def preparar_insumos(_df, clave: str, top_n: int):
    d = _df[["order_date", "quantity", "pizza_category", "pizza_ingredients"]].copy()
    d["ingredient"] = d["pizza_ingredients"].str.split(",")
    d = d.explode("ingredient")
    d["ingredient"] = d["ingredient"].str.strip()
    d = d[d["ingredient"] != ""]
    top = d.groupby("ingredient")["quantity"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    dt = d[d["ingredient"].isin(top)]
    consumo = dt.groupby([dt["order_date"].dt.normalize(), "ingredient"])["quantity"].sum().unstack(fill_value=0)
    idx = pd.date_range(_df["order_date"].min().normalize(), _df["order_date"].max().normalize(), freq="D")
    consumo = consumo.reindex(idx, fill_value=0)[top]
    shares = dt.groupby(["ingredient", "pizza_category"])["quantity"].sum().unstack(fill_value=0).reindex(top)
    shares = shares.div(shares.sum(axis=1), axis=0)
    return consumo, shares


def estadisticos_base(consumo: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Media": consumo.mean().round(2), "Moda": consumo.mode().iloc[0].round(2),
                         "Desv_Std": consumo.std().round(2), "Minimo": consumo.min().round(2),
                         "Maximo": consumo.max().round(2)})


def metricas_inventario(est: pd.DataFrame, factor, Z, L, S, H, dias) -> pd.DataFrame:
    """SS, ROP, EOQ y costos. 'factor' escala media y desviación (CV constante)."""
    media = (est["Media"] * factor).round(2)
    sd = (est["Desv_Std"] * factor).round(2)
    D = (media * dias).round(0)
    SS = np.ceil(Z * sd * np.sqrt(L)).astype(int)
    ROP = np.ceil(media * L + SS).astype(int)
    EOQ = np.round(np.sqrt(2 * D * S / H)).astype(int)
    dias_ped = pd.Series(np.where(D > 0, dias * EOQ / D.replace(0, np.nan), np.nan), index=est.index).round(1)
    costo = np.round(np.sqrt(2 * D * S * H)).astype(int)
    return pd.DataFrame({"Demanda_diaria": media, "Demanda_anual": D, "SS": SS, "ROP": ROP, "EOQ": EOQ,
                         "Dias_entre_pedidos": dias_ped, "Costo_EOQ": costo,
                         "Costo_con_SS": costo + H * SS})


def simular_ciclo(media, ROP, EOQ, SS, L, dias_sim=120):
    stock, camino, serie = EOQ + SS, None, []
    for t in range(dias_sim):
        if camino and t == camino[0]:
            stock += camino[1]; camino = None
        stock -= media
        if stock <= ROP and camino is None:
            camino = (t + int(L), EOQ)
        serie.append(stock)
    return np.array(serie)


# ----------------------------------------------------------------------------
# 5. PANEL LATERAL: PARÁMETROS GLOBALES (alimentan a los tres modelos)
# ----------------------------------------------------------------------------
st.title("🍕 Pizzería · Elasticidad + Demanda + Inventario")
st.caption("MPD · Grupo 2 — Modelos conectados: el precio mueve la demanda y la demanda mueve el inventario.")

st.sidebar.title("🎛️ Panel de control")
with st.sidebar.expander("📂 Datos", expanded=False):
    subido = st.file_uploader("Reemplazar CSV (opcional, separador ; o ,)", type=["csv"])
df = obtener_datos(subido)
clave = f"{len(df)}-{df['total_price'].sum():.2f}"
CATS = sorted(df["pizza_category"].unique())

agg = agg_variantes(df, clave)
ols_g, tabla_cat, tabla_tam = modelo_global_y_segmentos(agg, clave)
top5 = calcular_top5(df, clave)
productos_top5 = tuple(top5.index.tolist())

st.sidebar.subheader("💲 1 · Escenario de precios")
delta_global = st.sidebar.slider("Cambio de precio global (%)", -30, 30, 0, 1)
extra = {}
with st.sidebar.expander("Ajuste adicional por categoría (puntos %)"):
    for c in CATS:
        extra[c] = st.slider(f"{c}", -30, 30, 0, 1, key=f"extra_{c}")
deltas = {c: float(np.clip(delta_global + extra[c], -90, 200)) for c in CATS}
modo = st.sidebar.selectbox("Elasticidad a utilizar",
                            ["Global (OLS log-log)", "Por categoría (OLS)", "Manual"])
eps_manual = st.sidebar.slider("ε manual", -3.0, -0.05, -1.0, 0.05, disabled=(modo != "Manual"))
forzar = st.sidebar.checkbox("Si la categoría no es identificable o ε ≥ 0, usar la global", value=True)
aplicar_precio = st.sidebar.checkbox("Propagar el precio a demanda e inventario", value=True)

st.sidebar.subheader("📈 2 · Demanda (series de tiempo)")
serie_sel = st.sidebar.selectbox("Serie a pronosticar", [OPC_TOP5] + list(productos_top5))
modelo_sel = st.sidebar.selectbox("Modelo de pronóstico", [M_AUTO, M_RF, M_PCA])
horizonte = st.sidebar.slider("Horizonte (semanas)", 4, 12, 8)
n_est = st.sidebar.slider("Árboles del Random Forest", 100, 500, 300, 50)
aplicar_pron = st.sidebar.checkbox("Propagar el pronóstico al inventario", value=True)

st.sidebar.subheader("📦 3 · Inventario (revisión continua)")
top_n = st.sidebar.slider("N.º de insumos principales", 5, 15, 10)
L = st.sidebar.number_input("Lead time L (días)", 1, 30, 2)
nivel = st.sidebar.selectbox("Nivel de servicio", list(Z_TABLA.keys()), index=1)
Z = Z_TABLA[nivel]
S_ped = st.sidebar.number_input("Costo por pedido S (S/)", 1.0, 1000.0, 45.0, 1.0)
H_men = st.sidebar.number_input("Costo de mantener H (S/ por porción-año)", 0.1, 50.0, 1.0, 0.1)
dias_anio = st.sidebar.number_input("Días del año comercial", 300, 366, 365)
tc = st.sidebar.number_input("Tipo de cambio (S/ por US$, supuesto)", 1.0, 10.0, 3.70, 0.01,
                             help="Las ventas del dataset están en US$; S y H están en soles. "
                                  "Se usa para expresar ambos en la misma moneda.")

# ----------------------------------------------------------------------------
# 6. CÁLCULO ENCADENADO (Elasticidad → Demanda → Inventario)
# ----------------------------------------------------------------------------
eps, respaldo = construir_elasticidades(modo, eps_manual, forzar, ols_g["eps"], tabla_cat, CATS)
cero = {c: 0.0 for c in CATS}
d_efectivos = deltas if aplicar_precio else cero

# --- Demanda
serie, df_feat, (n_desc, n_ajust), pes_u, pes_r, precio_unit, d_ser = preparar_serie(
    df, clave, productos_top5, serie_sel)
# Backtesting de los 4 modelos ANTES de elegir: así el modo automático usa el de menor error
bt_res = {n: backtest(df_feat, n, n_est) for n in [M_NAIVE, M_MA4, M_RF, M_PCA]}
if modelo_sel == M_AUTO:
    modelo_usado = min([M_RF, M_PCA], key=lambda n: bt_res[n][2])
else:
    modelo_usado = modelo_sel
mejor_global = min(bt_res, key=lambda n: bt_res[n][2])
modelo_fit = ajustar_modelo(df_feat, modelo_usado, n_est)
fc = pronosticar(df_feat, modelo_fit, horizonte)
media_hist = float(serie["unidades"].mean())
g_pron = float(fc["prediccion"].mean() / media_hist) if aplicar_pron else 1.0
m_unid_serie = mult_demanda(pes_u, d_efectivos, eps, 0.0)
m_ing_serie = mult_demanda(pes_r, d_efectivos, eps, 1.0)
fc["escenario"] = (fc["prediccion"] * m_unid_serie).round(1)
fc["ingreso_base"] = fc["prediccion"] * precio_unit
fc["ingreso_escenario"] = fc["ingreso_base"] * m_ing_serie

# --- Inventario
consumo, shares = preparar_insumos(df, clave, top_n)
est = estadisticos_base(consumo)
m_insumo = mult_insumos(shares, d_efectivos, eps)
factor_esc = g_pron * m_insumo
# Tres niveles, con UNA sola definición de "base" en toda la app:
#   histórico  = consumo observado 2015 (factor 1)
#   base       = pronóstico con el precio actual (factor g)   ← todas las variaciones "vs base"
#   escenario  = pronóstico × efecto del precio (factor g · m)
met_hist = metricas_inventario(est, 1.0, Z, L, S_ped, H_men, dias_anio)
met_base = metricas_inventario(est, g_pron, Z, L, S_ped, H_men, dias_anio)
met_esc = metricas_inventario(est, factor_esc, Z, L, S_ped, H_men, dias_anio)

# ----------------------------------------------------------------------------
# 7. INTERFAZ
# ----------------------------------------------------------------------------
tab_panel, tab_dem, tab_el, tab_inv, tab_met = st.tabs(
    ["🧭 Panel integrado", "📈 Demanda (series de tiempo)", "💲 Elasticidad de precios",
     "📦 Inventario", "📚 Metodología y supuestos"])

# ============================ PANEL INTEGRADO ================================
with tab_panel:
    st.header("Panel integrado: un solo escenario, tres modelos")
    st.markdown(
        "**Flujo de la app:** `Precio (Δ%) × Elasticidad (ε)` → **multiplicador de demanda** → "
        "`Pronóstico de series de tiempo` fija el nivel esperado (**factor g**) → "
        "**consumo de insumos** → `SS, ROP, EOQ y costo anual`. "
        "Modifique cualquier parámetro del panel lateral y todo lo que sigue en la cadena se recalcula."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Δ precio global", f"{delta_global:+d} %", help="Panel lateral, sección 1")
    c2.metric("Elasticidad usada (media ponderada)", f"{sum(pes_u[c] * eps[c] for c in pes_u.index):.2f}")
    c3.metric("Multiplicador de demanda (precio)", f"×{m_unid_serie:.3f}", f"{(m_unid_serie - 1) * 100:+.1f} %")
    c4.metric("Factor de pronóstico g", f"×{g_pron:.3f}", f"{(g_pron - 1) * 100:+.1f} %",
              help="Pronóstico medio semanal ÷ media semanal histórica de la serie seleccionada")

    ing_b, ing_e = fc["ingreso_base"].sum(), fc["ingreso_escenario"].sum()
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(f"Unidades proyectadas ({horizonte} sem.)", f"{fc['escenario'].sum():,.0f}",
              f"{(fc['escenario'].sum() / fc['prediccion'].sum() - 1) * 100:+.1f} % vs base")
    c6.metric(f"Ingreso proyectado ({horizonte} sem.)", f"${ing_e:,.0f}", f"{(ing_e / ing_b - 1) * 100:+.1f} % vs base")
    cb, ce = met_base["Costo_EOQ"].sum(), met_esc["Costo_EOQ"].sum()
    c7.metric("Costo anual de inventario (Σ insumos)", f"S/ {ce:,.0f}", f"{(ce / cb - 1) * 100:+.1f} % vs base",
              delta_color="inverse")
    rb, re_ = met_base["ROP"].sum(), met_esc["ROP"].sum()
    c8.metric("Σ Puntos de reorden", f"{re_:,.0f}", f"{(re_ / rb - 1) * 100:+.1f} % vs base", delta_color="inverse")
    ch = met_hist["Costo_EOQ"].sum()
    st.caption(
        f"**Base** = pronóstico de series de tiempo con el precio actual (Δ precio = 0 %); todas las variaciones se miden "
        f"contra ella. Frente al consumo histórico 2015, la base ya implica {(cb / ch - 1) * 100:+.1f} % en costo de "
        f"inventario (efecto de g). Costo de inventario ≈ **US$ {ce / tc:,.0f}** · ingreso proyectado ≈ "
        f"**S/ {ing_e * tc:,.0f}** (tipo de cambio supuesto S/ {tc:.2f} por US$)."
    )

    cA, cB = st.columns([3, 2])
    with cA:
        st.subheader("Demanda semanal: histórico, pronóstico base y escenario")
        hist = df_feat[["fecha", "unidades"]].tail(16)
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        ax.plot(hist["fecha"], hist["unidades"], marker="o", color=COLOR_BASE, label="Histórico")
        ax.plot(fc["fecha"], fc["prediccion"], marker="o", ls="--", color="#F4A300", label="Pronóstico base (precio actual)")
        ax.plot(fc["fecha"], fc["escenario"], marker="s", ls="-", color=COLOR_ESC, label="Escenario (precio + elasticidad)")
        ax.axvline(df_feat["fecha"].iloc[-1], color="#6B8E23", ls=":")
        ax.set_ylabel("Unidades por semana"); ax.legend(fontsize=8); estilo(ax)
        mostrar(fig)
    with cB:
        st.subheader("Sensibilidad al precio (todo el sistema)")
        grilla = np.arange(-30, 31, 5)
        filas = []
        for d in grilla:
            dd = {c: float(np.clip(d + extra[c], -90, 200)) for c in CATS}
            mu = mult_demanda(pes_u, dd, eps, 0.0)
            mi = mult_demanda(pes_r, dd, eps, 1.0)
            mt = metricas_inventario(est, g_pron * mult_insumos(shares, dd, eps), Z, L, S_ped, H_men, dias_anio)
            filas.append({"d": d, "ingreso": (mi - 1) * 100, "unidades": (mu - 1) * 100,
                          "costo": (mt["Costo_EOQ"].sum() / cb - 1) * 100})
        sens = pd.DataFrame(filas)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(sens["d"], sens["ingreso"], marker="o", color=COLOR_ESC, label="Ingreso")
        ax.plot(sens["d"], sens["unidades"], marker="o", color="#F4A300", label="Unidades")
        ax.plot(sens["d"], sens["costo"], marker="o", color="#6B8E23", label="Costo inventario")
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(delta_global, color=COLOR_BASE, ls=":")
        ax.set_xlabel("Δ precio global (%)"); ax.set_ylabel("Variación vs base (%)")
        ax.legend(fontsize=8); estilo(ax)
        mostrar(fig)

    st.subheader("Inventario: base vs escenario")
    comp = pd.DataFrame({
        "Demanda diaria histórica": met_hist["Demanda_diaria"],
        "Demanda diaria base": met_base["Demanda_diaria"], "Demanda diaria escenario": met_esc["Demanda_diaria"],
        "ROP base": met_base["ROP"], "ROP escenario": met_esc["ROP"],
        "EOQ base": met_base["EOQ"], "EOQ escenario": met_esc["EOQ"],
        "Costo base (S/)": met_base["Costo_EOQ"], "Costo escenario (S/)": met_esc["Costo_EOQ"],
        "Costo escenario (US$)": (met_esc["Costo_EOQ"] / tc).round(0),
        "Factor total (g·m)": factor_esc.round(3)})
    st.dataframe(comp)
    st.download_button("⬇️ Descargar comparación (CSV)", csv_bytes(comp), "comparacion_inventario.csv", "text/csv")
    if serie_sel == OPC_TOP5:
        st.info("El pronóstico se calcula sobre el **Top 5 de productos** y su razón de crecimiento (g) se aplica "
                "a todos los insumos. Para un producto individual, seleccione otra serie en el panel lateral.")

# ============================== DEMANDA ======================================
with tab_dem:
    st.header(f"Proyección de demanda con series de tiempo — {serie_sel}")
    st.write(f"Modelo usado: **{modelo_usado}**"
             + (" (elegido automáticamente por menor MAPE)" if modelo_sel == M_AUTO else "")
             + f" · {n_est} árboles · horizonte de {horizonte} semanas. "
             "Variable objetivo: unidades semanales; predictores: precio, calendario, rezagos y media móvil.")
    if serie_sel == OPC_TOP5:
        st.write("Top 5 por unidades: " + ", ".join(f"**{p}** ({int(u):,})" for p, u in top5.items()))
    if n_desc or n_ajust:
        st.info(f"Serie continua de {len(serie)} semanas: se recortaron {n_desc} semana(s) de borde con menos de "
                f"{MIN_DIAS_BORDE} días de datos y {n_ajust} semana(s) interiores con días sin venta se normalizaron "
                "a 7 días (unidades × 7 ÷ días con venta). Así no hay huecos y los rezagos son semanas consecutivas.")

    s1, s2 = st.columns(2)
    with s1:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(serie["fecha"], serie["unidades"], color=COLOR_ESC, marker=".")
        ax.set_title("Serie semanal de unidades"); ax.set_ylabel("Unidades"); estilo(ax)
        mostrar(fig)
    with s2:
        a = acf(df_feat["unidades"].values, 12)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(range(1, 13), a, color="#F4A300")
        lim = 1.96 / np.sqrt(len(df_feat))
        ax.axhline(lim, color="gray", ls="--", lw=1); ax.axhline(-lim, color="gray", ls="--", lw=1)
        ax.set_title("Autocorrelación (ACF) — justifica el uso de rezagos"); ax.set_xlabel("Rezago (semanas)")
        estilo(ax); mostrar(fig)

    st.subheader("Variables de entrada y estadística descriptiva")
    st.dataframe(df_feat[["fecha", "unidades"] + VARIABLES].head(10))
    st.dataframe(descriptivos(df_feat, ["unidades"] + VARIABLES, moda_en={"unidades", "mes"}))
    st.caption("La moda solo se reporta donde tiene sentido; en t, semana o sin/cos todos los valores son distintos.")
    cats_tab = pd.DataFrame([{"Variable": c, "Moda": d_ser[c].mode().iloc[0],
                              "Frecuencia de la moda (%)": round((d_ser[c] == d_ser[c].mode().iloc[0]).mean() * 100, 1),
                              "N.º categorías": d_ser[c].nunique()} for c in ["pizza_name", "pizza_category", "pizza_size"]]
                            ).set_index("Variable")
    st.dataframe(cats_tab)

    imp = importancias_rf(df_feat, n_est)
    claves = ["unidades"] + imp.head(3).index.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(14, 5))
    for j, v in enumerate(claves):
        axes[0, j].hist(df_feat[v], bins=12, color="#F4A300", edgecolor=COLOR_BASE)
        axes[0, j].set_title(f"Histograma — {v}", fontsize=9)
        bp = axes[1, j].boxplot(df_feat[v], vert=False, patch_artist=True)
        bp["boxes"][0].set_facecolor(COLOR_ESC); bp["boxes"][0].set_alpha(0.6)
        axes[1, j].set_title(f"Diagrama de caja — {v}", fontsize=9); axes[1, j].set_yticks([])
        estilo(axes[0, j]); estilo(axes[1, j])
    mostrar(fig)

    st.subheader("Reducción de dimensiones (PCA) e importancia de variables")
    evr, carg = info_pca(df_feat)
    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar(range(1, len(evr) + 1), evr.cumsum() * 100, color="#6B8E23")
        ax.axhline(95, color=COLOR_ESC, ls="--"); ax.set_xlabel("N.º de componentes")
        ax.set_ylabel("Varianza acumulada (%)"); ax.set_title("PCA sobre las 11 variables"); estilo(ax)
        mostrar(fig)
        n95 = int(np.searchsorted(evr.cumsum(), 0.95) + 1)
        st.caption(f"Con {n95} componentes se retiene ≥ 95 % de la varianza (variante «Random Forest + PCA»).")
    with p2:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.barh(imp.index[::-1], imp.values[::-1], color=COLOR_ESC)
        ax.set_title("Importancia de variables (Random Forest)"); estilo(ax)
        mostrar(fig)

    st.subheader("Validación: backtesting con ventana expansiva")
    st.caption("Entrenamiento mínimo de 28 semanas, horizonte de 4 y paso de 4; se compara contra dos referencias simples.")
    comp_bt = pd.DataFrame([{"Modelo": n, "MAE": r[0], "RMSE": r[1], "MAPE (%)": r[2]}
                            for n, r in bt_res.items()]).set_index("Modelo").round(2)
    st.dataframe(comp_bt)
    st.success(f"Menor MAPE en el backtesting: **{mejor_global}** ({comp_bt.loc[mejor_global, 'MAPE (%)']:.2f} %). "
               f"Modelo usado para el pronóstico: **{modelo_usado}** ({comp_bt.loc[modelo_usado, 'MAPE (%)']:.2f} %).")
    if comp_bt.loc[M_MA4, "MAPE (%)"] < comp_bt.loc[modelo_usado, "MAPE (%)"]:
        st.warning("La media móvil de 4 semanas tiene menor error que el modelo usado: con un solo año de datos, "
                   "el modelo de machine learning aún no supera a una referencia simple en esta serie.")
    st.caption("En «Random Forest + PCA», el escalado y el PCA se ajustan dentro de cada fold (pipeline), "
               "solo con la parte de entrenamiento: no hay fuga de información hacia la prueba.")
    tb = bt_res[modelo_usado][3]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(tb["fecha"], tb["real"], marker="o", color=COLOR_ESC, label="Real")
    ax.plot(tb["fecha"], tb["prediccion"], marker="o", color="#F4A300", label=f"Predicción — {modelo_usado}")
    ax.legend(); ax.set_ylabel("Unidades"); estilo(ax)
    mostrar(fig)

    st.subheader(f"Pronóstico recursivo a {horizonte} semanas")
    tabla_fc = fc.rename(columns={"fecha": "Semana", "prediccion": "Unidades (base)",
                                  "escenario": "Unidades (escenario precio)",
                                  "ingreso_base": "Ingreso base ($)", "ingreso_escenario": "Ingreso escenario ($)"})
    tabla_fc["Semana"] = tabla_fc["Semana"].dt.strftime("%d/%m/%Y")
    tabla_fc[["Ingreso base ($)", "Ingreso escenario ($)"]] = tabla_fc[["Ingreso base ($)", "Ingreso escenario ($)"]].round(0)
    st.dataframe(tabla_fc.set_index("Semana"))
    st.download_button("⬇️ Descargar pronóstico (CSV)", csv_bytes(tabla_fc.set_index("Semana")), "pronostico.csv", "text/csv")
    st.caption("Supuesto: el precio de referencia del modelo es el promedio de las últimas 4 semanas. "
               "El efecto de un cambio de precio se incorpora mediante la elasticidad (columna «escenario»). "
               "El error tiende a acumularse en los últimos pasos porque los rezagos son predicciones previas.")

# ============================== ELASTICIDAD ==================================
with tab_el:
    st.header("Elasticidad precio-demanda (regresión log-log)")
    st.subheader("1 · Análisis de ventas")
    st.dataframe(descriptivos(df, ["unit_price", "quantity", "total_price"]))
    cc1, cc2 = st.columns(2)
    cc1.write("**Categoría (líneas de pedido)**"); cc1.dataframe(df["pizza_category"].value_counts().rename("líneas"))
    cc2.write("**Tamaño (líneas de pedido)**"); cc2.dataframe(df["pizza_size"].value_counts().rename("líneas"))
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for ax, v in zip(axes[:3], ["unit_price", "quantity", "total_price"]):
        ax.hist(df[v], bins=20, color="#F4A300", edgecolor=COLOR_BASE); ax.set_title(f"Histograma — {v}", fontsize=9); estilo(ax)
    for ax, v in zip(axes[3:], ["unit_price", "total_price"]):
        bp = ax.boxplot(df[v], patch_artist=True); bp["boxes"][0].set_facecolor(COLOR_ESC)
        ax.set_title(f"Caja — {v}", fontsize=9); estilo(ax)
    mostrar(fig)

    st.subheader("2 · Modelo global OLS: ln(cantidad) ~ ln(precio) + tamaño + categoría")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Elasticidad global ε", f"{ols_g['eps']:.3f}")
    m2.metric("p-valor de ε", f"{ols_g['p']:.4f}")
    m3.metric("R² / R² ajustado", f"{ols_g['r2']:.3f} / {ols_g['r2_aj']:.3f}")
    m4.metric("Variantes (n)", f"{ols_g['n']}")
    st.dataframe(ols_g["tabla"].round(4))
    if ols_g["p"] > 0.05:
        st.warning("ε global **no es significativa al 5 %**: el precio, por sí solo, explica poco de la variación entre "
                   "variantes. Úsela como escenario referencial, no como estimación definitiva.")
    st.write("Como |ε| < 1, la demanda es **inelástica**: un alza de precio aumenta el ingreso." if abs(ols_g["eps"]) < 1
             else "Como |ε| ≥ 1, la demanda es **elástica**: un alza de precio reduce el ingreso.")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for c, col in zip(CATS, PALETA):
        sub = agg[agg["pizza_category"] == c]
        ax.scatter(sub["ln_precio"], sub["ln_cantidad"], color=col, label=c, alpha=0.8)
    ax.set_xlabel("ln(precio promedio)"); ax.set_ylabel("ln(cantidad total)"); ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)

    st.subheader("3 · Elasticidad por segmento")
    e1, e2 = st.columns(2)
    e1.write("**Por categoría** (OLS con control de tamaño)"); e1.dataframe(tabla_cat.round(4))
    e2.write("**Por tamaño** (OLS con control de categoría)"); e2.dataframe(tabla_tam.round(4))
    st.caption("«No identificable»: dentro del segmento el precio queda determinado por el tamaño (p. ej., todas las "
               "Chicken cuestan lo mismo para un tamaño dado), por lo que no hay variación para estimar ε. "
               "Los segmentos con pocas variantes producen estimaciones inestables.")
    if modo.startswith("Por categoría"):
        st.warning("Está usando elasticidades por categoría. Con 18–25 variantes por categoría pueden resultar muy "
                   "dispares; verifique su significancia antes de decidir.")
    if respaldo:
        st.info("Se usó la elasticidad global para: " + ", ".join(respaldo))

    st.subheader("4 · Benchmark: OLS vs Ridge vs Random Forest")
    bench, n_cel = benchmark_elasticidad(df, clave)
    st.dataframe(bench.round(3))
    st.caption(f"Validación cruzada de 5 folds sobre {n_cel:,} celdas (variante × día de la semana × mes). "
               "ε del Random Forest = elasticidad implícita al simular ±10 % de precio.")

    st.subheader("5 · Simulador de precios (con la elasticidad del panel lateral)")
    f1, f2 = st.columns(2)
    seg_cat = f1.selectbox("Categoría", ["Todas"] + CATS)
    seg_tam = f2.selectbox("Tamaño", ["Todos"] + sorted(df["pizza_size"].unique()))
    df_seg = df.copy()
    if seg_cat != "Todas":
        df_seg = df_seg[df_seg["pizza_category"] == seg_cat]
    if seg_tam != "Todos":
        df_seg = df_seg[df_seg["pizza_size"] == seg_tam]
    if df_seg.empty:
        st.warning("El segmento seleccionado no tiene ventas.")
    else:
        sim = simular_precio(df_seg, eps, list(range(-30, 31, 5)))
        opt = sim.loc[sim["Ingreso ($)"].idxmax()]
        k1, k2, k3 = st.columns(3)
        k1.metric("Ingreso base", f"${df_seg['total_price'].sum():,.0f}")
        k2.metric("Δ precio que maximiza el ingreso", f"{opt['Δ precio (%)']:+.0f} %")
        k3.metric("Ingreso en el óptimo", f"${opt['Ingreso ($)']:,.0f}", f"{opt['Δ ingreso (%)']:+.1f} %")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(sim["Δ precio (%)"], sim["Ingreso ($)"], marker="o", color=COLOR_ESC, label="Ingreso")
        ax.axvline(delta_global, color=COLOR_BASE, ls=":", label="Escenario del panel")
        ax.set_xlabel("Δ precio (%)"); ax.set_ylabel("Ingreso ($)"); ax.legend(); estilo(ax)
        mostrar(fig)
        st.dataframe(sim.round(2).set_index("Δ precio (%)"))
        if opt["Δ precio (%)"] in (sim["Δ precio (%)"].min(), sim["Δ precio (%)"].max()):
            st.warning("El óptimo cae en el **borde** del rango simulado: con |ε| < 1 el ingreso crece con el precio. "
                       "El modelo no incluye costos, competencia ni caídas no lineales de la demanda; no lo extrapole.")

        st.subheader("6 · Simulador por historia (solo precios observados)")
        hist_p = (df_seg.assign(v=df_seg["pizza_name"] + "|" + df_seg["pizza_size"]).groupby("unit_price")
                  .agg(Unidades=("quantity", "sum"), Ingreso=("total_price", "sum"), Variantes=("v", "nunique")))
        hist_p["Ingreso por variante"] = hist_p["Ingreso"] / hist_p["Variantes"]
        st.dataframe(hist_p.round(2))
        st.success(f"Precio histórico con mayor ingreso total: **${hist_p['Ingreso'].idxmax():.2f}** · "
                   f"con mayor ingreso por variante: **${hist_p['Ingreso por variante'].idxmax():.2f}**.")

# ============================== INVENTARIO ===================================
with tab_inv:
    st.header("Gestión de inventarios — revisión continua (s, Q)")
    st.write(f"Supuestos vigentes: L = **{L} días**, servicio **{nivel}** (Z = {Z}), S = **S/ {S_ped:,.2f}**, "
             f"H = **S/ {H_men:,.2f}**, {dias_anio} días. Factor de escenario: g (pronóstico) × m (precio) por insumo.")
    st.subheader("Estadística diaria de consumo (porciones/día, histórico)")
    st.dataframe(est)

    st.subheader("Métricas de inventario — escenario actual")
    out = met_esc.copy()
    out.insert(0, "Factor", factor_esc.round(3))
    st.dataframe(out)
    st.download_button("⬇️ Descargar métricas (CSV)", csv_bytes(out), "metricas_inventario.csv", "text/csv")
    st.caption("SS = Z·σ·√L · ROP = μ·L + SS · EOQ = √(2·D·S/H) · Costo_EOQ = √(2·D·S·H) · "
               "Costo_con_SS = Costo_EOQ + H·SS. Unidades: μ y σ en porciones/día, L en días, "
               f"D = μ × {dias_anio} en porciones/año, S y H en S/. D anualiza el ritmo de demanda pronosticado para "
               f"las próximas {horizonte} semanas (supuesto: ese ritmo se mantiene); el plan semana a semana "
               "está en la tabla de requerimientos.")

    g1, g2 = st.columns(2)
    for ax_col, campo, color, titulo in [(g1, "ROP", "#e66101", "Punto de reorden (ROP)"),
                                         (g2, "EOQ", "#2b83ba", "Lote económico (EOQ)")]:
        with ax_col:
            idx = np.arange(len(met_esc))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(idx + 0.2, met_base[campo].values, 0.4, color="#BBBBBB", label="Base (pronóstico, precio actual)")
            ax.barh(idx - 0.2, met_esc[campo].values, 0.4, color=color, label="Escenario")
            ax.set_yticks(idx); ax.set_yticklabels(met_esc.index, fontsize=8); ax.invert_yaxis()
            ax.set_title(titulo); ax.legend(fontsize=8); estilo(ax)
            mostrar(fig)

    st.subheader("Ciclo de inventario simulado")
    ins = st.selectbox("Insumo", met_esc.index.tolist())
    r = met_esc.loc[ins]
    ciclo = simular_ciclo(r["Demanda_diaria"], r["ROP"], r["EOQ"], r["SS"], L)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(ciclo, color=COLOR_ESC, label="Stock")
    ax.axhline(r["ROP"], color="#F4A300", ls="--", label=f"ROP = {r['ROP']:.0f}")
    ax.axhline(r["SS"], color="#6B8E23", ls=":", label=f"SS = {r['SS']:.0f}")
    ax.set_xlabel("Días"); ax.set_ylabel("Porciones"); ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)

    st.subheader(f"Requerimiento semanal proyectado por insumo ({horizonte} semanas)")
    if aplicar_pron:
        f_t = (fc["prediccion"] / media_hist).values
    else:
        f_t = np.ones(horizonte)
    plan = pd.DataFrame(np.ceil(np.outer(est["Media"].values * 7 * m_insumo.values, f_t)),
                        index=est.index, columns=fc["fecha"].dt.strftime("%d/%m/%Y")).astype(int)
    st.dataframe(plan)
    st.download_button("⬇️ Descargar plan (CSV)", csv_bytes(plan), "plan_requerimiento.csv", "text/csv")
    st.caption("Requerimiento = consumo diario histórico × 7 × (pronóstico semanal ÷ media histórica) × multiplicador de precio.")

# ============================== METODOLOGÍA ==================================
with tab_met:
    st.header("Metodología, supuestos y limitaciones")
    st.markdown(
        """
**Datos.** 48 620 líneas de pedido de 2015 (fecha en formato día/mes/año).

**Elasticidad.** OLS log-log sobre variantes (pizza × categoría × tamaño):
`ln(cantidad) = β₀ + ε·ln(precio) + tamaño + categoría`. Benchmarks: Ridge y Random Forest.

**Demanda.** Serie semanal continua: se recortan solo las semanas incompletas de los bordes del año y las
semanas interiores con días sin venta se normalizan a 7 días, para no romper los rezagos. Variables de
calendario, rezagos 1–4 y media móvil. Random Forest (300 árboles, profundidad 8) con backtesting de ventana
expansiva y pronóstico recursivo. Se añade una variante con **PCA** (ajustada dentro de cada fold) y dos
referencias simples (ingenuo y media móvil); en modo automático se usa el modelo de ML con menor MAPE.

**Inventario.** Revisión continua: SS = Z·σ·√L, ROP = μ·L + SS, EOQ = √(2DS/H).

**Conexión entre modelos.**
1. Precio: cada categoría recibe Δ_c; la demanda se multiplica por (1+Δ_c)^ε_c y el ingreso por (1+Δ_c)^(1+ε_c).
2. Pronóstico: g = pronóstico medio semanal ÷ media semanal histórica de la serie.
3. Inventario: μ' = μ × g × m_insumo, con σ' = σ × g × m_insumo (coeficiente de variación constante).

**Escenarios y base.** Histórico = consumo 2015 · **Base** = pronóstico con el precio actual (factor g) ·
Escenario = base × efecto del precio (factor m). Todas las variaciones «vs base» se miden contra la base.

**Unidades y monedas.** Demanda en pizzas/semana; insumos en porciones/día (una porción = un ingrediente en
una pizza vendida). Ventas e ingresos en US$ (moneda del dataset); S y H en S/; se convierten con el tipo de
cambio del panel lateral (supuesto).

**Limitaciones.**
- Los precios no cambian en el tiempo dentro de 2015: la elasticidad se identifica con variación *entre* variantes
  (corte transversal), por lo que puede mezclarse con diferencias de popularidad entre productos.
- ε global no es significativa; las elasticidades por categoría son inestables. Trate el simulador como escenario.
- No hay datos de costos: el «óptimo de ingreso» no es un óptimo de utilidad.
- El pronóstico cubre el Top 5; su factor g se extrapola a todos los insumos.
- Los costos S y H son supuestos del informe, no datos observados.
        """
    )
