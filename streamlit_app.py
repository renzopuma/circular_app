import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.set_page_config(layout="wide")
# Treatment
st.title("Dashboard")
st.markdown("Circular economy v2")


@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    return data


##Load data
coef = load_data("./data/io_eora_circular_expanded.csv")
indi = load_data("./data/indicators_expanded.csv")
indi["year"] = indi["Year"]

#Apply labels

conditions  = [indi['krent_quart'] == 1, 
               indi['krent_quart'] == 2,
               indi['krent_quart'] == 3,
               indi['krent_quart'] == 4]

choices = ['Low Kr quartile', 
           'Mid low kr quartile', 
           'Mid high kr quartile', 
           'High kr quartile']

indi['Kr quartile'] = np.select(conditions, choices, default="")


conditions  = [indi['exp_orintd'] == 1, 
               indi['exp_orintd'] == 2,
               indi['exp_orintd'] == 3]

choices = ['Lower exp oriented', 
           'Mid exp oriented', 
           'Higher exp oriented']

indi['EXP-oriented'] = np.select(conditions, choices, default="")




conditions  = [indi['mp_cost_dpndc'] == 1, 
               indi['mp_cost_dpndc'] == 2,
               indi['mp_cost_dpndc'] == 3]

choices = ['Lower import cost dependence', 
           'Mid import cost depedence', 
           'Higher import cost depedence']

indi['MP Cost Dependence'] = np.select(conditions, choices, default="")




conditions  = [indi['local_mkt'] == 1, 
               indi['local_mkt'] == 2,
               indi['local_mkt'] == 3]

choices = ['Local markets production', 
           'Mix local and non local markets production', 
           'Non-local markets production']

indi['Local Mkt'] = np.select(conditions, choices, default="")


conditions  = [indi['flow_cost_of_production'] == 1, 
               indi['flow_cost_of_production'] == 2,
               indi['flow_cost_of_production'] == 3,
               indi['flow_cost_of_production'] == 4,
               indi['flow_cost_of_production'] == 5,
               indi['flow_cost_of_production'] == 6,
               indi['flow_cost_of_production'] == 7,
               indi['flow_cost_of_production'] == 8,
               indi['flow_cost_of_production'] == 9]

choices = ['Local oriented - Not import cost dependence', 
           'Local oriented - Some import cost dependence',
           'Local oriented - High import cost dependence', 
           'Local and export oriented - Not import cost dependence', 
           'Local and export oriented - Some import cost dependence', 
           'Local and export oriented - High import cost dependence', 
           'Export oriented - Not import cost dependence', 
           'Export oriented - Some import cost dependence',
           'Export oriented - High import cost dependence']

indi['Flow/Costs of Production'] = np.select(conditions, choices, default="")

indi['CE type'] = indi['ce_type']

country_list = list(indi["code_country"].unique())
sector_list = list(indi["sector"].unique())
year_list = list(indi.year.unique())
# var_list = list(indi.columns)


@st.cache_data
def prefilter(df):
    df = df[df["code_country"].isin(country_list)]
    df = df[df["year"].isin(year_list)]
    return df


coef = prefilter(coef)


def filter(df, type):
    if type == "c":
        df1 = df[df["code_country"].isin(selected_ctry)]
    else:
        df1 = df[df["code_country"].isin(bench)]

    df1 = df1[df1["year"].isin(selected_year)]
    return df1


def edges_creation(df, indi, source_cat, target_cat, s_level, t_level, type):
    df = df.merge(
        indi[["code_country", "year", "sector_num", source_cat]],
        left_on=["code_country", "year", "from_sector"],
        right_on=["code_country", "year", "sector_num"],
    )
    df = df.drop(["sector_num"], axis=1)
    df = df.merge(
        indi[["code_country", "year", "sector_num", target_cat]],
        left_on=["code_country", "year", "to_sector"],
        right_on=["code_country", "year", "sector_num"],
    )
    df = df.drop(["sector_num"], axis=1)

    if source_cat == target_cat:
        source_cat = source_cat + "_x"
        target_cat = target_cat + "_y"

    if type == "c":
        df = (
            df[["code_country", "year", source_cat, target_cat, "sec_"]]
            .groupby(["code_country", source_cat, target_cat])
            .sum()
            .reset_index()
        )
    else:
        df = (
            df[["code_country", "year", source_cat, target_cat, "sec_"]]
            .groupby([source_cat, target_cat])
            .sum()
            .reset_index()
        )
    df = df.rename(columns={source_cat: "source_label", target_cat: "target_label"})
    df["source"] = df["source_label"] + "_" + s_level
    df["target"] = df["target_label"] + "_" + t_level
    return df


def sankey_db(df, indi, type):
    ntw = {name: pd.DataFrame() for name in net}
    net_lbl = {name: pd.DataFrame() for name in nlb}

    # edges
    ntw["c1"] = edges_creation(df, indi, sel_c1, sel_c2, "c1", "c2", type)
    ntw["c2"] = edges_creation(df, indi, sel_c2, sel_c3, "c2", "c3", type)
    ntw["c3"] = edges_creation(df, indi, sel_c3, sel_c4, "c3", "c4", type)

    edges = pd.concat(ntw).reset_index()

    # nodes
    net_lbl["s"] = edges[["source", "source_label"]]
    net_lbl["t"] = edges[["target", "target_label"]]
    net_lbl["s"] = net_lbl["s"].rename(
        columns={"source": "node", "source_label": "label"}
    )
    net_lbl["t"] = net_lbl["t"].rename(
        columns={"target": "node", "target_label": "label"}
    )
    net_lbl = pd.concat(net_lbl, axis=0).set_index("node").sort_index().reset_index()
    net_lbl = net_lbl.drop_duplicates(subset=["node", "label"])

    return edges, net_lbl


def sankey(edges, net_lbl, c):
    nodes = np.unique(edges[["source", "target"]], axis=None)
    nodes = pd.Series(index=nodes, data=range(len(nodes)))

    fig = go.Figure(
        go.Sankey(
            textfont=dict(color="rgba(0,0,0,1)", size=14),
            node={
                "label": net_lbl.label.to_list(),
                "pad": 50,
                "thickness": 40,
                "line": dict(color="white", width=0.12),
                # "color": d5[i].color.tolist(),
                # "x":x,
                # "y":y
            },
            link={
                "source": nodes.loc[edges["source"]],
                "target": nodes.loc[edges["target"]],
                "value": edges["sec_"],
                #'color': d4[i]['link_color']
            },
        )
    )

    fig.update_layout(
        hovermode="x",
        # Update our title, set font to 36px, bluesteel colour, and make it bold
        title="<span style='font-size:36px;color:steelblue;'><b>" + c + "</b></span>",
        font=dict(size=10, color="white"),
        paper_bgcolor="#F8F8ff",
    )
    return fig


def bgrapf_sector(df_bench, df_country, var, ctry):
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{}]],
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.001,
    )

    fig.append_trace(
        go.Bar(
            x=df_country["sector"], y=df_country[var], name=ctry, marker_color="#277935"
        ),
        1,
        1,
    )

    fig.append_trace(
        go.Bar(
            x=df_bench["sector"], y=df_bench[var], name="BENCH", marker_color="#702035"
        ),
        1,
        1,
    )

    return fig



# Main design
y1 = st.slider(
    "Select a range of values",
    year_list[0],
    year_list[len(year_list) - 1],
    (year_list[0], year_list[0]),
)


selected_year = year_list
selected_year = [x for x in selected_year if (x >= y1[0]) & (x <= y1[1])]

rws1 = st.columns(2)

with rws1[0]:
    selected_ctry = st.selectbox("Select a country", country_list, 15)
    selected_ctry = [selected_ctry]

##Select bench
with rws1[1]:
    bench = st.multiselect("Select benchmark", country_list, country_list[:2])
id_codes = ["code_country", "year", "sector"]

varl = ["output_exp", "va_exp", "kbs", "l", "expo_exp", "imports", "krent"]


varc = [
    'rir',
    'reir',
    'vmd_mining',
    'vmd_petro',
    'vmd_chem',
    'vmd_metal',
    'vmd',
    'wor',
    'csmu_manu',
    'csmu_food',
    'csmu_text',
    'csmu_cpuelec',
    'csmu_eleceq',
    'csmu_mcheq',
    'csmu']


vartxt = ['Flow/Costs of Production', 
          'Local Mkt',
          'EXP-oriented',
          'CE type', 
          'MP Cost Dependence', 
          'Kr quartile']


# Create the initial bar chart for the selected variable

def indi_bench(df, varlist):
    df["code_country"] = "Benchmark"
    df = (
        df[id_codes + varlist]
        .groupby(["code_country", "sector"])
        .median()
        .reset_index()
    )
    return df
indi_c = filter(indi, "c")
indi_b = filter(indi, "b")

indi_fb = indi_bench(indi_b.copy(), varl + varc)


# --- Primer cajon ---

container1 = st.container()
with container1:
    cont1 = st.columns(3)
    with cont1[0]:
        io_var1 = st.selectbox("Select IO variable", varl, 0)
        st.plotly_chart(bgrapf_sector(indi_fb, indi_c, io_var1, selected_ctry[0]))
    with cont1[1]:
        io_var2 = st.selectbox("Select IO variable", varl, 1)
        st.plotly_chart(bgrapf_sector(indi_fb, indi_c, io_var2, selected_ctry[0]))
    with cont1[2]:
        io_var3 = st.selectbox("Select IO variable", varl, 2)
        st.plotly_chart(bgrapf_sector(indi_fb, indi_c, io_var3, selected_ctry[0]))




# Bar graph




c_var = st.selectbox("Select circular variable", varc, 1)

st.plotly_chart(bgrapf_sector(indi_fb, indi_c, c_var, selected_ctry[0]))

st.dataframe(indi_c[id_codes + vartxt].reset_index(drop=True).set_index('sector'))


var_list = ["CE type", "Flow/Costs of Production", "Local Mkt", "sector"]
sel_c1 = st.selectbox("Select a category for level 1", var_list, 3)
sel_c2 = st.selectbox("Select a category for level 2", var_list, 0)
sel_c3 = st.selectbox("Select a category for level 3", var_list, 2)
sel_c4 = st.selectbox("Select a category for level 4", var_list, 3)


coef_c = filter(coef, "c")


coef_b = filter(coef, "b")


net = ["c1", "c2", "c3"]
nlb = ["s", "t"]


df_sk1 = coef_c[["code_country", "year", "from_sector", "to_sector", "sec_"]]
df_sk2 = coef_b[["code_country", "year", "from_sector", "to_sector", "sec_"]]


edges_c, nodes_c = sankey_db(df_sk1, indi_c, "c")
edges_b, nodes_b = sankey_db(df_sk2, indi_b, "b")


st.plotly_chart(sankey(edges_c, nodes_c, selected_ctry[0]))
st.plotly_chart(sankey(edges_b, nodes_b, "Benchmark"))


###For comparasion
st.markdown("comparasion")


##Select sector
rel_sector = st.multiselect("Select relevant sectors", sector_list, sector_list[:2])