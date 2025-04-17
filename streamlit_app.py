import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection

st.set_page_config(layout="wide")
# Treatment
st.title("Dashboard")
st.markdown("Circular economy v2")






coef_path=[st.secrets['coef_path0'], 
           st.secrets['coef_path1'], 
           st.secrets['coef_path2'], 
           st.secrets['coef_path3'], 
           st.secrets['coef_path4'], 
           st.secrets['coef_path5']]

indi_path=[st.secrets['indi_path0'], 
           st.secrets['indi_path1'], 
           st.secrets['indi_path2'], 
           st.secrets['indi_path3'], 
           st.secrets['indi_path4'], 
           st.secrets['indi_path5']]

conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data
def load_data(path_list):
    dfs = {}
    
    i = 0
    for df in path_list:
        dfs[i] = conn.read(spreadsheet=df)
        i +=1
    data = pd.concat(dfs).reset_index(drop=True)
    return data


coef = load_data(coef_path)
indi = load_data(indi_path)


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


def sankey_db(df, indi, sel, type, rel):
    ntw = {name: pd.DataFrame() for name in net}
    net_lbl = {name: pd.DataFrame() for name in nlb}

    # edges
    ntw["c1"] = edges_creation(df, indi, sel[0], sel[1], "c1", "c2", type)
    ntw["c2"] = edges_creation(df, indi, sel[1], sel[2], "c2", "c3", type)
    ntw["c3"] = edges_creation(df, indi, sel[2], sel[3], "c3", "c4", type)
    
    if rel[0] != 'all':
        ntw['c1'] = ntw['c1'][ntw['c1']['source_label']==rel[0]]
    
    if rel[1] != 'all':
        ntw['c1'] = ntw['c1'][ntw['c1']['target_label']==rel[1]]
        ntw['c2'] = ntw['c2'][ntw['c2']['source_label']==rel[1]]
        
    if rel[2] != 'all':
        ntw['c2'] = ntw['c2'][ntw['c2']['target_label']==rel[2]]
        ntw['c3'] = ntw['c3'][ntw['c3']['source_label']==rel[2]]
    
    if rel[3] != 'all':
        ntw['c3'] = ntw['c3'][ntw['c3']['target_label']==rel[3]]

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
        #font=dict(size=10, color="white"),
        #paper_bgcolor= "rgba(50, 0, 0, 0)",
        height = 900
    )
    return fig

def bgrapf_sector(df_bench, df_country, varx, vary, ctry):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_country[varx], 
            y=df_country[vary], 
            name=ctry, 
            marker_color="#590A27"
        ))

    fig.add_trace(
        go.Bar(
            x=df_bench[varx], 
            y=df_bench[vary], 
            name="BENCH", 
            marker_color="#BF8173"
        ))
    fig.update_layout(
        height = 600
    )
    return fig


def stck_bar(df, varx, vary, varlist):
    fig = go.Figure()
    for i in range(len(varlist)):
        fig.add_trace(
            go.Bar(name=varlist[i], 
                    x=df[varx], 
                    y=df[vary[i]]
                    )
        )

    #data=[
    #go.Bar(name=varlist[0], x=df[varx], y=df[vary[0]]),
    #go.Bar(name=varlist[1], x=df[varx], y=df[vary[1]]),
    #go.Bar(name=varlist[2], x=df[varx], y=df[vary[2]])])

    fig.update_layout(barmode='stack')
    return fig


# Main design
y1 = st.slider(
    "Select a range of years",
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

varl_dic={'Total output':"output_exp", 
          'Total value added':"va_exp", 
          'Centrality':'kbs', 
          'Labor rents': 'l', 
          'Capital rents': 'k',
          'Total exports':'expo_exp',
          'Total imports': 'imports', 
          'Ratio capital rents-output':'krent',
          'Spillover (priorized indirect connections)': "sp2"}

varl_m = ['Total output', 
         'Total value added', 
         'Centrality', 
         'Labor rents', 
         'Capital rents',
         'Total exports',
         'Total imports', 
         'Ratio capital rents-output', 
         'Spillover (priorized indirect connections)']

varl = ["output_exp", "va_exp", "kbs", "l", 'k', "expo_exp", "imports", "krent", "sp2", 'expend_clsum', 'demand_exp', 'spill_rwsum','hhi_ups', 'hhi_dws' ]



varc_dic =  {'Recycling input rate':'rir',
            'Maintenance & Repair input rate':'reir',
            'Virgin material dependency: Minning':'vmd_mining',
            'Virgin material dependency: Petrochemicals':'vmd_petro',
            'Virgin material dependency: Chemicals':'vmd_chem',
            'Virgin material dependency: Basic metals':'vmd_metal',
            'Virgin material dependency: Total':'vmd',
            'Waste output rate':'wor',
            'Cross sector material utilization: Manufacture':'csmu_manu',
            'Cross sector material utilization: Food':'csmu_food',
            'Cross sector material utilization: Textiles':'csmu_text',
            'Cross sector material utilization: Computer electronics':'csmu_cpuelec',
            'Cross sector material utilization: Electrical equipment':'csmu_eleceq',
            'Cross sector material utilization: Machinery':'csmu_mcheq',
            'Cross sector material utilization: Total':'csmu'}

varc_m =  ['Recycling input rate',
           'Maintenance & Repair input rate',
           'Virgin material dependency: Minning',
           'Virgin material dependency: Petrochemicals',
           'Virgin material dependency: Chemicals',
           'Virgin material dependency: Basic metals',
           'Virgin material dependency: Total',
           'Waste output rate',
           'Cross sector material utilization: Manufacture',
           'Cross sector material utilization: Food',
           'Cross sector material utilization: Textiles',
           'Cross sector material utilization: Computer electronics',
           'Cross sector material utilization: Electrical equipment',
           'Cross sector material utilization: Machinery',
           'Cross sector material utilization: Total']


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

def collapser(df, varlist):
    df["code_country"] = "Benchmark"
    df = (
        df[id_codes + varlist]
        .groupby(["code_country", "sector", 'year'])
        .median()
        .reset_index()
    )
    return df
indi_c = filter(indi, "c")
indi_b = filter(indi, "b")

indi_fb = collapser(indi_b.copy(), varl + varc)


# --- Primer cajon ---

rel_sector = st.selectbox("Select relevant sector", sector_list, 1)

outp_c = indi_c['output_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0]
outp_b = indi_fb['output_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0]

outp_c2 = indi_c['output_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0]
outp_b2 = indi_fb['output_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0]

outp_c_delta = ((outp_c/outp_c2)-1)*100
outp_b_delta = ((outp_b/outp_b2)-1)*100

va_c = ((indi_c['va_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0])/outp_c)*100
va_b = ((indi_fb['va_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0])/outp_b)*100

va_c2 = ((indi_c['va_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0])/outp_c2)*100
va_b2 = ((indi_fb['va_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0])/outp_b2)*100

va_c_delta = ((va_c/va_c2)-1)*100
va_b_delta = ((va_b/va_b2)-1)*100

exp_c = ((indi_c['expend_clsum'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0])/outp_c)*100
exp_b = ((indi_fb['expend_clsum'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0])/outp_b)*100

exp_c2 = ((indi_c['expend_clsum'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0])/outp_c2)*100
exp_b2 = ((indi_fb['expend_clsum'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0])/outp_b2)*100

exp_c_delta = ((exp_c/exp_c2)-1)*100
exp_b_delta = ((exp_b/exp_b2)-1)*100




container1 = st.container()
with container1:
    st.markdown("Overall Market size")
    cont1 = st.columns(3)
    with cont1[0]:
        st.markdown("Total output of selected sector")
        io_var1 ='output_exp'
        st.plotly_chart(bgrapf_sector(indi_fb[indi_fb['sector']==rel_sector], 
                                      indi_c[indi_c['sector']==rel_sector], 
                                      "year",
                                      io_var1, 
                                      selected_ctry[0]), key="cht0")
        
        st.metric(label=f'{selected_ctry[0]} - {selected_year[-1]}', 
                    value=f'${outp_c:.2f}K', 
                    delta=f'{outp_c_delta:.2f}%', 
                    border=True) 
        st.metric(label=f'BENCH - {selected_year[-1]}', 
                    value=f'${outp_b:.2f}K', 
                    delta=f'{outp_b_delta:.2f}%', 
                    border=True) 
        
        
    with cont1[1]:
        st.markdown("Total value added of selected sector")
        io_var2 ='va_exp'
        st.plotly_chart(bgrapf_sector(indi_fb[indi_fb['sector']==rel_sector], 
                                      indi_c[indi_c['sector']==rel_sector],  
                                      "year",
                                      io_var2, 
                                      selected_ctry[0]), key="cht1")
        
        st.metric(label=f'Value added (% of output) {selected_ctry[0]} - {selected_year[-1]}', 
                    value=f'{va_c:.2f}%', 
                    delta=f'{va_c_delta:.2f}%', 
                    border=True) 
        st.metric(label=f'Value added (% of output) BENCH - {selected_year[-1]}', 
                    value=f'{va_b:.2f}%', 
                    delta=f'{va_b_delta:.2f}%', 
                    border=True) 
        
    with cont1[2]:
        st.markdown("Total expenditures of selected sector")
        io_var3 ='expend_clsum'
        st.plotly_chart(bgrapf_sector(indi_fb[indi_fb['sector']==rel_sector], 
                                      indi_c[indi_c['sector']==rel_sector], 
                                      "year",
                                      io_var3, 
                                      selected_ctry[0]), key="cht2")
        st.metric(label=f'Expenditures (% of output) {selected_ctry[0]} - {selected_year[-1]}', 
                    value=f'{exp_c:.2f}%', 
                    delta=f'{exp_c_delta:.2f}%', 
                    border=True) 
        st.metric(label=f'Expenditures (% of output) BENCH - {selected_year[-1]}', 
                    value=f'{exp_b:.2f}%', 
                    delta=f'{exp_b_delta:.2f}%', 
                    border=True) 




# --- Segundo cajon ---

#Calcular cajones de metricas




expo_c = ((indi_c['expo_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0])/outp_c)*100
expo_b = ((indi_fb['expo_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0])/outp_b)*100

expo_c2 = ((indi_c['expo_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0])/outp_c2)*100
expo_b2 = ((indi_fb['expo_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0])/outp_b2)*100

expo_c_delta = ((expo_c/expo_c2)-1)*100
expo_b_delta = ((expo_b/expo_b2)-1)*100

intc_c = ((indi_c['spill_rwsum'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0])/outp_c)*100
intc_b = ((indi_fb['spill_rwsum'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0])/outp_b)*100

intc_c2 = ((indi_c['spill_rwsum'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0])/outp_c2)*100
intc_b2 = ((indi_fb['spill_rwsum'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0])/outp_b2)*100

intc_c_delta = ((intc_c/intc_c2)-1)*100
intc_b_delta = ((intc_b/intc_b2)-1)*100


demd_c = ((indi_c['demand_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0])/outp_c)*100
demd_b = ((indi_fb['demand_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0])/outp_b)*100

demd_c2 = ((indi_c['demand_exp'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0])/outp_c2)*100
demd_b2 = ((indi_fb['demand_exp'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0])/outp_b2)*100

demd_c_delta = ((demd_c/demd_c2)-1)*100
demd_b_delta = ((demd_b/demd_b2)-1)*100


#Extraer ouput y calcular porcentajes


container2 = st.container()
with container2:
    st.markdown("Domestic use vs exports")
    var_tot = ['spill_rwsum', 'demand_exp', 'expo_exp']
    var_lab = ['Internal consumption', 'Demand of final users', 'Exports']
    ##Corregir demanda de usuarios finales
    #stck_bar(df, varx, vary, varlist)
    cont2 = st.columns(2)
    with cont2[0]:
        st.markdown(selected_ctry[0])
        st.plotly_chart(stck_bar(indi_c[indi_c['sector']==rel_sector], 
                                 'year', 
                                 var_tot, 
                                 var_lab), key="cht3")
        st.markdown("% total output")
        cont2a = st.columns(3)
        
        with cont2a[0]:
            st.metric(label=f'Exports - {selected_year[-1]}', 
                    value=f'{expo_c:.2f}%', 
                    delta=f'{expo_c_delta:.2f}%', 
                    border=True)   
        with cont2a[1]:
            st.metric(label=f'Demand of final users - {selected_year[-1]}', 
                    value=f'{demd_c:.2f}%', 
                    delta=f'{demd_c_delta:.2f}%', 
                    border=True)
        with cont2a[2]:
            st.metric(label=f'Internal consumption - {selected_year[-1]}', 
                    value=f'{intc_c:.2f}%', 
                    delta=f'{intc_c_delta:.2f}%', 
                    border=True)
        
        
        
    with cont2[1]:
        st.markdown("BENCH")
        st.plotly_chart(stck_bar(indi_fb[indi_fb['sector']==rel_sector], 
                                 'year', 
                                 var_tot, 
                                 var_lab), key="cht4")
        st.markdown("% total output")
        cont2b = st.columns(3)
        with cont2b[0]:
            st.metric(label=f'Exports - {selected_year[-1]}', 
                    value=f'{expo_b:.2f}%', 
                    delta=f'{expo_b_delta:.2f}%', 
                    border=True)   
        with cont2b[1]:
            st.metric(label=f'Demand of final users - {selected_year[-1]}', 
                    value=f'{demd_b:.2f}%', 
                    delta=f'{demd_b_delta:.2f}%', 
                    border=True)
        with cont2b[2]:
            st.metric(label=f'Internal consumption - {selected_year[-1]}', 
                    value=f'{intc_b:.2f}%', 
                    delta=f'{intc_b_delta:.2f}%', 
                    border=True)








# --- Tercer cajon ---


st.markdown("Retained in the economy vs wasted")


container3 = st.container(height=1100)

sel_c1 = "Local Mkt"
sel_c2 = "sector"
sel_c3 = "CE type"
sel_c4 = "sector"

sel = [sel_c1, sel_c2, sel_c3, sel_c4]
coef_c = filter(coef, "c")
coef_b = filter(coef, "b")
net = ["c1", "c2", "c3"]
nlb = ["s", "t"]

fil = ['all', rel_sector, 'R-basic', 'all']

df_sk1 = coef_c[["code_country", "year", "from_sector", "to_sector", "sec_"]]
df_sk2 = coef_b[["code_country", "year", "from_sector", "to_sector", "sec_"]]

edges_c, nodes_c = sankey_db(df_sk1, indi_c, sel, "c", fil)
edges_b, nodes_b = sankey_db(df_sk2, indi_b, sel, "b", fil)


wor_c = indi_c['wor'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0]*100
wor_b = indi_fb['wor'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0]*100

wor_c2 = indi_c['wor'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0]*100
wor_b2 = indi_fb['wor'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0]*100

wor_c_delta = ((wor_c/wor_c2)-1)*100
wor_b_delta = ((wor_b/wor_b2)-1)*100

with container3:
    #st.dataframe(indi_c['wor'][(indi_c['sector']==rel_sector)])
    cont3 = st.columns(2)
    with cont3[0]:
        st.plotly_chart(sankey(edges_c, nodes_c, selected_ctry[0]), key="cht5")
        st.metric(label=f'Waste output rate {selected_year[-1]}', value=f'{wor_c:.3f}% of total', delta=f'{wor_c_delta:.3f}%', border=True)
    with cont3[1]:
        st.plotly_chart(sankey(edges_b, nodes_b, "Benchmark"), key="cht6")
        st.metric(label=f'Waste output rate {selected_year[-1]}', value=f'{wor_b:.3f}% of total', delta=f'{wor_b_delta:.3f}%', border=True)


#2 color on benchmark for connection


# --- Cuarto cajon ---
vmd_c = indi_c['vmd'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0]*100
vmd_b = indi_fb['vmd'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0]*100

vmd_c2 = indi_c['vmd'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0]*100
vmd_b2 = indi_fb['vmd'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0]*100

vmd_c_delta = ((vmd_c/vmd_c2)-1)*100
vmd_b_delta = ((vmd_b/vmd_b2)-1)*100


container4 = st.container()
with container4:
    st.markdown("Use of virgin materials")
    var_tot = ['vmd_mining', 'vmd_petro', 'vmd_chem', 'vmd_metal']
    var_lab = ['VMD: Mining', 'VMD: Petroleum', 'VMD: Chemicals', 'VMD: Basic Metals']
    ##Corregir demanda de usuarios finales
    #stck_bar(df, varx, vary, varlist)
    cont4 = st.columns(2)
    with cont4[0]:
        st.markdown(selected_ctry[0])
        st.plotly_chart(stck_bar(indi_c[indi_c['sector']==rel_sector], 
                                 'year', 
                                 var_tot, 
                                 var_lab), key="cht7")
        st.metric(label=f'Virgin material dependence (VMD) {selected_year[-1]}', value=f'{vmd_c:.3f}% of total', delta=f'{vmd_c_delta:.3f}%', border=True)
    with cont4[1]:
        st.markdown("BENCH")
        st.plotly_chart(stck_bar(indi_fb[indi_fb['sector']==rel_sector], 
                                 'year', 
                                 var_tot, 
                                 var_lab), key="cht8")
        st.metric(label=f'Virgin material dependence (VMD) {selected_year[-1]}', value=f'{vmd_b:.3f}% of total', delta=f'{vmd_b_delta:.3f}%', border=True)




# --- Quinto cajon ---


uhhi_c = indi_c['hhi_ups'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0]
uhhi_b = indi_fb['hhi_ups'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0]

uhhi_c2 = indi_c['hhi_ups'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0]
uhhi_b2 = indi_fb['hhi_ups'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0]

uhhi_c_delta = ((uhhi_c/uhhi_c2)-1)*100
uhhi_b_delta = ((uhhi_b/uhhi_b2)-1)*100



dhhi_c = indi_c['hhi_dws'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-1])].iloc[0]
dhhi_b = indi_fb['hhi_dws'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-1])].iloc[0]

dhhi_c2 = indi_c['hhi_dws'][(indi_c['sector']==rel_sector) & (indi_c['year']==selected_year[-2])].iloc[0]
dhhi_b2 = indi_fb['hhi_dws'][(indi_fb['sector']==rel_sector) & (indi_fb['year']==selected_year[-2])].iloc[0]

dhhi_c_delta = ((dhhi_c/dhhi_c2)-1)*100
dhhi_b_delta = ((dhhi_b/dhhi_b2)-1)*100





#Terminar de calcular el hhi para los cajones

container5 = st.container()
with container5:
    st.markdown("Supply Chain concentration")
    cont5 = st.columns(2)
    with cont5[0]:
        st.markdown("Concentration in upstream connections (HHI)")
        io_var1 ='hhi_ups'
        st.plotly_chart(bgrapf_sector(indi_fb[indi_fb['sector']==rel_sector], 
                                      indi_c[indi_c['sector']==rel_sector], 
                                      "year",
                                      io_var1, 
                                      selected_ctry[0]), key="cht9")
        #st.markdown("Upstream concentration (HHI)")
        st.metric(label=f'{selected_ctry[0]} - {selected_year[-1]}', 
                  value=f'{uhhi_c:.4f}', 
                  delta=f'{uhhi_c_delta:.3f}%', 
                  delta_color="inverse",
                  border=True)
        st.metric(label=f'BENCH - {selected_year[-1]}', 
                  value=f'{uhhi_b:.4f}', 
                  delta=f'{uhhi_b_delta:.3f}%', 
                  delta_color="inverse",
                  border=True)
    with cont5[1]:
        st.markdown("Concentration in downstream connections (HHI)")
        io_var1 ='hhi_dws'
        st.plotly_chart(bgrapf_sector(indi_fb[indi_fb['sector']==rel_sector], 
                                      indi_c[indi_c['sector']==rel_sector], 
                                      "year",
                                      io_var1, 
                                      selected_ctry[0]), key="cht10")
        #st.markdown("Downstream concentration (HHI)")
        st.metric(label=f'{selected_ctry[0]} - {selected_year[-1]}', 
                  value=f'{dhhi_c:.4f}', 
                  delta=f'{dhhi_c_delta:.3f}%', 
                  delta_color="inverse",
                  border=True)
        st.metric(label=f'BENCH - {selected_year[-1]}', 
                  value=f'{dhhi_b:.4f}', 
                  delta=f'{dhhi_b_delta:.3f}%', 
                  delta_color="inverse",
                  border=True)







##Select sector
