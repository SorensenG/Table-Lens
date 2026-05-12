from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.patches import Rectangle


st.set_page_config(
    page_title="Table Lens",
    page_icon="TL",
    layout="wide",
)


NUMERIC_COLOR = "#2f7ec1"
MISSING_COLOR = "#d9dde3"
BACKGROUND_COLOR = "#ffffff"
GRID_COLOR = "#e7e9ee"
FOCUS_EDGE = "#1d2433"


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    kind: str
    missing: int
    unique: int


@st.cache_data(show_spinner=False)
def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)
        except Exception as exc:  # pragma: no cover - shown to the user in Streamlit
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Nao foi possivel ler o CSV. Tentativas: " + " | ".join(errors))


def infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "quantitativa"

    clean = series.dropna().astype(str).str.strip()
    if clean.empty:
        return "nominal"

    numeric_ratio = pd.to_numeric(clean, errors="coerce").notna().mean()
    unique_count = clean.nunique(dropna=True)
    if numeric_ratio > 0.92 and unique_count > 8:
        return "quantitativa"
    if unique_count <= 12:
        return "nominal/ordinal"
    return "nominal"


def profile_columns(df: pd.DataFrame) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    for column in df.columns:
        series = df[column]
        profiles.append(
            ColumnProfile(
                name=str(column),
                kind=infer_kind(series),
                missing=int(series.isna().sum()),
                unique=int(series.nunique(dropna=True)),
            )
        )
    return profiles


def numeric_normalization(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(np.nan, index=series.index)
    if min_value == max_value:
        return pd.Series(0.5, index=series.index)
    return (values - min_value) / (max_value - min_value)


def color_for_category(value: Any, palette: dict[str, tuple[float, float, float]]) -> Any:
    if pd.isna(value):
        return MISSING_COLOR
    return palette.get(str(value), "#b8c0cc")


def build_category_palettes(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    palettes: dict[str, dict[str, Any]] = {}
    base_colors = sns.color_palette("Set2", 8) + sns.color_palette("tab20", 20)
    for column in columns:
        values = sorted(df[column].dropna().astype(str).unique().tolist())
        palettes[column] = {
            value: base_colors[index % len(base_colors)]
            for index, value in enumerate(values)
        }
    return palettes


def compact_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.3g}"
    text = str(value)
    return text if len(text) <= 18 else text[:15] + "..."


def render_table_lens(
    df: pd.DataFrame,
    columns: list[str],
    type_overrides: dict[str, str],
    focus_position: int,
    focus_radius: int,
    title: str,
):
    visible = df[columns].reset_index(drop=True)
    row_count = len(visible)
    col_count = len(columns)

    if row_count == 0 or col_count == 0:
        return None

    focus_rows = set(
        range(
            max(0, focus_position - focus_radius),
            min(row_count, focus_position + focus_radius + 1),
        )
    )
    heights = np.array([0.95 if row in focus_rows else 0.18 for row in range(row_count)])
    total_height = float(heights.sum())
    fig_width = max(10.0, col_count * 1.45)
    fig_height = min(20.0, max(5.5, total_height * 0.30 + 1.6))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    category_columns = [col for col in columns if type_overrides[col] != "quantitativa"]
    palettes = build_category_palettes(visible, category_columns)
    numeric_cache = {
        col: numeric_normalization(visible[col])
        for col in columns
        if type_overrides[col] == "quantitativa"
    }

    y = 0.0
    for row_index in range(row_count):
        row_height = float(heights[row_index])
        is_focus = row_index in focus_rows
        row_alpha = 1.0 if is_focus else 0.88

        if is_focus:
            ax.add_patch(
                Rectangle(
                    (-0.08, y + 0.02),
                    col_count + 0.16,
                    row_height - 0.04,
                    facecolor="#f5f7fa",
                    edgecolor=FOCUS_EDGE,
                    linewidth=0.7,
                    zorder=0,
                )
            )

        for col_index, column in enumerate(columns):
            cell_x = float(col_index)
            cell_y = y + row_height * 0.12
            cell_height = row_height * 0.76
            cell_width = 0.92
            value = visible.at[row_index, column]
            kind = type_overrides[column]

            ax.add_patch(
                Rectangle(
                    (cell_x, cell_y),
                    cell_width,
                    cell_height,
                    facecolor="#f8f9fb",
                    edgecolor=GRID_COLOR,
                    linewidth=0.28,
                    zorder=1,
                )
            )

            if pd.isna(value):
                ax.add_patch(
                    Rectangle(
                        (cell_x, cell_y),
                        cell_width,
                        cell_height,
                        facecolor=MISSING_COLOR,
                        edgecolor="none",
                        alpha=0.7,
                        zorder=2,
                    )
                )
            elif kind == "quantitativa":
                normalized = numeric_cache[column].iloc[row_index]
                if pd.notna(normalized):
                    ax.add_patch(
                        Rectangle(
                            (cell_x + 0.02, cell_y + cell_height * 0.15),
                            max(0.02, float(normalized) * (cell_width - 0.04)),
                            cell_height * 0.70,
                            facecolor=NUMERIC_COLOR,
                            edgecolor="none",
                            alpha=row_alpha,
                            zorder=3,
                        )
                    )
            else:
                ax.add_patch(
                    Rectangle(
                        (cell_x + 0.02, cell_y + cell_height * 0.12),
                        cell_width - 0.04,
                        cell_height * 0.76,
                        facecolor=color_for_category(value, palettes[column]),
                        edgecolor="none",
                        alpha=row_alpha,
                        zorder=3,
                    )
                )

            if is_focus:
                ax.text(
                    cell_x + 0.04,
                    y + row_height * 0.54,
                    compact_value(value),
                    ha="left",
                    va="center",
                    fontsize=6.4,
                    color="#172033",
                    zorder=4,
                )

        y += row_height

    ax.set_xlim(-0.18, col_count)
    ax.set_ylim(total_height, -0.35)
    ax.set_xticks(np.arange(col_count) + 0.46)
    ax.set_xticklabels(columns, rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0,
        total_height + 0.23,
        "Barras azuis = quantitativas normalizadas | Cores = categorias | Linhas altas = foco",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#596273",
    )
    fig.tight_layout()
    return fig


def sort_dataframe(df: pd.DataFrame, column: str | None, ascending: bool) -> pd.DataFrame:
    if not column or column == "(sem ordenacao)":
        return df.copy()
    return df.sort_values(by=column, ascending=ascending, na_position="last").reset_index(drop=True)


def focus_interval(row_count: int, focus_row: int, focus_radius: int) -> tuple[int, int]:
    start = max(1, focus_row - focus_radius)
    end = min(row_count, focus_row + focus_radius)
    return start, end


def sidebar_controls(df: pd.DataFrame):
    profiles = profile_columns(df)
    default_columns = [profile.name for profile in profiles[: min(10, len(profiles))]]

    with st.sidebar:
        st.header("Dados")
        st.metric("Linhas", f"{len(df):,}".replace(",", "."))
        st.metric("Colunas", f"{len(df.columns):,}".replace(",", "."))

        selected_columns = st.multiselect(
            "Colunas",
            options=[str(col) for col in df.columns],
            default=default_columns,
        )

        max_rows = st.slider(
            "Linhas exibidas",
            min_value=20,
            max_value=min(800, max(20, len(df))),
            value=min(220, max(20, len(df))),
            step=20,
        )

        sort_options = ["(sem ordenacao)"] + selected_columns
        sort_column = st.selectbox("Ordenar por", sort_options, index=0)
        ascending = st.toggle("Ordem crescente", value=True)

        focus_radius = st.slider("Raio do foco", min_value=0, max_value=10, value=2)

        st.divider()
        st.subheader("Tipos")
        type_overrides: dict[str, str] = {}
        for column in selected_columns:
            inferred = infer_kind(df[column])
            options = ["quantitativa", "nominal/ordinal", "nominal"]
            default_index = options.index(inferred) if inferred in options else 2
            type_overrides[column] = st.selectbox(
                column,
                options=options,
                index=default_index,
                key=f"type-{column}",
            )

    return selected_columns, max_rows, sort_column, ascending, focus_radius, type_overrides


def show_column_summary(df: pd.DataFrame):
    summary = pd.DataFrame(
        [
            {
                "coluna": profile.name,
                "tipo inferido": profile.kind,
                "ausentes": profile.missing,
                "unicos": profile.unique,
            }
            for profile in profile_columns(df)
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def main():
    st.title("Table Lens")

    uploaded = st.file_uploader("Carregar CSV", type=["csv"])

    try:
        if uploaded is not None:
            df = read_csv_bytes(uploaded.getvalue())
            dataset_name = uploaded.name
        else:
            st.info("Carregue um arquivo CSV para iniciar.")
            return
    except Exception as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.warning("O CSV foi lido, mas nao tem linhas.")
        return

    selected_columns, max_rows, sort_column, ascending, focus_radius, type_overrides = sidebar_controls(df)
    if not selected_columns:
        st.warning("Selecione pelo menos uma coluna.")
        return

    sorted_df = sort_dataframe(df, sort_column, ascending)
    visible_df = sorted_df.head(max_rows).copy()

    row_count = len(visible_df)
    default_focus_row = min(row_count, max(1, row_count // 2))
    current_focus_row = int(st.session_state.get("focus_row", default_focus_row))
    current_focus_row = min(row_count, max(1, current_focus_row))
    st.session_state["focus_row"] = current_focus_row
    interval_start, interval_end = focus_interval(row_count, current_focus_row, focus_radius)
    interval_text = (
        f"linha {interval_start}"
        if interval_start == interval_end
        else f"linhas {interval_start}-{interval_end}"
    )

    # Mostrar linhas ao usuário em base 1 (mais intuitivo).
    focus_row = st.slider(
        f"Foco atual: {interval_text}",
        min_value=1,
        max_value=max(1, row_count),
        value=current_focus_row,
        step=1,
        key="focus_row",
    )
    # Converter para índice 0-based usado internamente
    focus_position = max(0, focus_row - 1)
    interval_start, interval_end = focus_interval(row_count, focus_row, focus_radius)
    st.caption(f"Intervalo exibido no foco: linhas {interval_start}-{interval_end}")

    left, right = st.columns([3, 1])
    with left:
        fig = render_table_lens(
            visible_df,
            selected_columns,
            type_overrides,
            focus_position,
            focus_radius,
            f"{dataset_name} - {len(visible_df)} linhas exibidas",
        )
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with right:
        st.subheader("Foco")
        start = max(0, focus_position - focus_radius)
        end = min(len(visible_df), focus_position + focus_radius + 1)
        st.dataframe(visible_df.iloc[start:end][selected_columns], use_container_width=True)

    tabs = st.tabs(["Amostra", "Resumo das colunas", "Estatisticas"])
    with tabs[0]:
        st.dataframe(visible_df[selected_columns].head(80), use_container_width=True)
    with tabs[1]:
        show_column_summary(df[selected_columns])
    with tabs[2]:
        numeric_columns = [
            column
            for column in selected_columns
            if type_overrides.get(column) == "quantitativa"
        ]
        if numeric_columns:
            st.dataframe(df[numeric_columns].describe().T, use_container_width=True)
        else:
            st.info("Nenhuma coluna quantitativa selecionada.")


if __name__ == "__main__":
    main()
