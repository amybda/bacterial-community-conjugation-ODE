# Make a more user-friendly interface than the pure class code or the Jupyter notebook

# Import modules
from ODE_model import ode_lotka  # noqa: I001
import os
import pandas as pd
import numpy as np
import numpy.random as rd
from string import ascii_uppercase
import plotly.graph_objects as go
import plotly.express as px
from shiny import reactive
from shiny.express import input, render, ui
from shiny.types import FileInfo
from shinywidgets import render_plotly
from faicons import icon_svg as icon


# --- THE STYLE AND GUI OF THE PAGE ---
with ui.tags.head():
    ui.tags.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css"),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"),
    ui.tags.script(src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js"),
    ui.tags.script("""document.addEventListener('DOMContentLoaded', function() {renderMathInElement(document.body, {
                        delimiters: [{left: "$$", right: "$$", display: true},
                                     {left: "\\[", right: "\\]", display: true},
                                     {left: "$", right: "$", display: false}] }); }); """)
ui.tags.style("""
    .tab-content {padding: 5px !important;}
    .nav-link {border-radius: 15px !important}
    .popover {max-width: 1000px}
    .card-body {position: relative !important}""")
ui.page_opts(title="ODE-Based Modelling of Conjugation in a Bacterial Community in Response to Drug Perturbation", fillable=True)


# --- FILEPATHS ---
# File paths
project_root = os.path.dirname(__file__)
if os.path.isdir(os.path.join(project_root, "Data")):
    data_dir = os.path.join(project_root, "Data")
else:
    data_dir = os.path.join(os.path.dirname(project_root), "outputs")


# --- SHARED VARIABLES ---
empty_results = {'status': 1, 't': np.zeros((10,)), 'y': np.zeros((4,401))}
change_species_alpha = reactive.value(pd.DataFrame())
change_species_r = reactive.value(pd.DataFrame())
change_species_results = reactive.value(empty_results)
change_conn_alpha = reactive.value(pd.DataFrame())
change_conn_r = reactive.value(pd.DataFrame())
change_conn_results = reactive.value(empty_results)
change_sigma_si_alpha = reactive.value(pd.DataFrame())
change_sigma_si_r = reactive.value(pd.DataFrame())
change_sigma_si_results = reactive.value(empty_results)
change_Pm_alpha = reactive.value(pd.DataFrame())
change_Pm_r = reactive.value(pd.DataFrame())
change_Pm_results = reactive.value(empty_results)
change_growth_alpha = reactive.value(pd.DataFrame())
change_growth_r = reactive.value(pd.DataFrame())
change_growth_results = reactive.value(empty_results)
change_tdrug_alpha = reactive.value(pd.DataFrame())
change_tdrug_r = reactive.value(pd.DataFrame())
change_tdrug_results = reactive.value(empty_results)
change_drug_strength_alpha = reactive.value(pd.DataFrame())
change_drug_strength_r = reactive.value(pd.DataFrame())
change_drug_strength_results = reactive.value(empty_results)
change_sus_alpha = reactive.value(pd.DataFrame())
change_sus_r = reactive.value(pd.DataFrame())
change_sus_results = reactive.value(empty_results)
change_tdr_alpha = reactive.value(pd.DataFrame())
change_tdr_r = reactive.value(pd.DataFrame())
change_tdr_results = reactive.value(empty_results)
change_compare_conj_alpha = reactive.value(pd.DataFrame())
change_compare_conj_r = reactive.value(pd.DataFrame())
change_compare_conj_results = reactive.value(empty_results)
change_conj_alpha = reactive.value(pd.DataFrame())
change_conj_r = reactive.value(pd.DataFrame())
change_conj_results = reactive.value(empty_results)
model_alpha = reactive.value(pd.DataFrame())
model_alpha_checks = reactive.value(0)
model_r = reactive.value(pd.DataFrame())
model_results = reactive.value(empty_results)



# --- MAIN OUTPUTS PANEL ---
with ui.navset_card_pill(id="main_tabs"):


    # INTRODUCTION TEXT TO MODEL
    with ui.nav_panel("Introduction"):
        ui.markdown("""
            # **Overview**
            The experiment we are working on is to determining the effect of various drugs on the gut microbiome, which then may enable or prevent the spread of antibiotic resistance genes \\(ARGs). The current theory is that if drugs such as antibiotics successfully peturb the microbiome, if the donor and recipient are resistant, they can overgrow, increasing the odds they would come in contact with each other, allowing conjugation to occur. We also want to understand the influence of spatial structure within the gut - where different species are localised in the gut microbiome - on this phenomenon. While we are currently conducting *in vitro* and *in vivo* mouse experiments, we would also like to develop a model which explores the dynamics of this.
            <br><br>
            There are multiple different modelling approaches for studying the microbiome ([Kumar, 2019](https://www.nature.com/articles/s41564-019-0491-9#Sec11)). The two most relevant methods are ordinary differential equation (ODE)-based modelling, and agent-based modelling (ABM). Here we will focus on ODE-based modelling. ODE-based modelling has been used to model monocultures or simple co-culture systems (minimal species), using rate equations to measure the change in different variables in time - such as species growth or plasmid transfer. The rate equations can be used to create mass balances (a set of ODEs that can be solved using standard computational solvers). These ODEs allow the simulation of dynamic interactions between species, but they often have to be simplified to reduce the number of variables and parameters.
            <br><hr>
            ### [Generalised Lotka-Volterra, gLV (1987)](https://stefanoallesina.github.io/ICTS_2025/ch1.html)
            LV models are widely used to describe the time evolution of ecosystems, often in a classical predator-prey LV model (where two competing populations directly affect each other's growth, either positively or negatively) ([Succurro, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5906705/#s2)). Model parameters, including intitial conditions, quantify direct interactions to determine the stability of the system and can be numerically solved. [Stein *et al.* (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3861043/#s4) implemented an expansion of the gLV model, which includes a term to model susceptibility, $\\epsilon_{i}$, to an external perturbation such as a drug $p(t)$.
            <br><br>
            To construct this programme, I used code from Laura de Nies (Stracy Lab), in combination with information from Coyte's papers:
            * [The ecology of the microbiome: Networks, competition, and stability (2015)](https://www.science.org/doi/full/10.1126/science.aad2602#abstract)
            * [Horizontal gene transfer and ecological interactions jointly control microbiome stability (2022)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001847#sec008)
            <br><br>
            The equation used here to model growth is: $$\\frac{dx_i(t)}{dt} = x_i(t) \\left(r_i \\sum_j \\alpha_{ij}x_j(t) + \\epsilon_i p(t) \\right) + \\gamma$$ where:<br><br>
            * $\\frac{dx_i(t)}{dt}$ is the change in community density
            * $x(t)$ is a vector containing the densities of all the populations $1,...,n$ at time $t$
            * $r$ is a vector of the growth rates (or death rates when negative) measuring the growth/decline of population $i$ when growing alone at low density
            * $\\alpha$ is an $n \\times n$ matrix of interaction coefficients, with self-inhibition (or growth) vector $s_i(x)$ as the diagonal matrix
            * $\\epsilon$ is the susceptibility of a species to the perturbation (in this case the drug)
            * $p$ is the overall strength of the perturbation introduced
            * $\\gamma$ is the term considering plasmid transfer, and how that affects the population (ie. changing some cells from recipients to transconjugants)
            <br><hr>
            ### <u>This Programme</u>
            The goal of this programme is to allow users to interactively explore the effect of adjusting different parameters in the Lotka-Volterra model on the number of transconjugants formed. Vist the `Explore Parameters` tab to identify the impact of each parameter on the model. Check out the `Run Model` tab to run a model complete with all your own parameters.<br><br>""")


    # PARAMETER PANELS
    with ui.nav_panel("Explore Parameters"):  # noqa: SIM117
        with ui.navset_pill_list(id="params_tabs", widths=(2,10)):


            # DETERMINING SPECIES NUMBER
            with ui.nav_panel("Modifying Species"):
                ui.markdown("""
                ## **Determining Species Number**
                The model is initialised for a run using `ode_lotka(n_species)`.The parameter `n_species` is used to determine the number of species in the community, excluding the donor, recipient and transconjugants. Therefore, a value of `n_species = 1` would have 4 species total. The number of species determines how many linked ODE equations need to be solved, with the number of entries in $x(t)$ and the size of the interaction matrix.<br><br>
                In this model, the recipient species and the transconjugant species are treated as the same - namely, they have the same interactions with other species. As a default, the donor is also the same species as the recipient and transconjugant, with the same interactions as both. However, there is an option to set the donor and recipient to be different species with different interactions. Either way, the donor, recipient and transconjugants can have different suceptibilities to the drug to model the effects of receiving resistance genes.<br><br>
                The starting species density `x0` is a parameter entered as an array with `3 + n_species` entries. Species (including donor and recipient when 'spiked in') are likely to not remain at the starting density, as the system will always try to tend to equilibrium based on the ODEs. This may not reflect real experimental behaviour, where species remain in stationary phase at the OD they are introduced. In this model, you can run an equilibrium step - to let the community find equilibrium without any drug perturbations or pathogens, or start the 'main simulation' right away without finding equilibrium.
                <hr>""")
                @render_plotly(width='100%')
                def nspecies_trans():
                    x = np.linspace(0, 26, 27) 
                    x_rev = x[::-1]
                    trans_nspecies = np.load(f'{data_dir}/change_nspecies.npy')
                    # Render figure layouts
                    fig = go.Figure()
                    fig.update_layout(plot_bgcolor='#eff5fa')
                    # Render plot
                    fig.add_trace(go.Scatter(x=np.concat([x,x_rev]), y=np.concat([trans_nspecies.max(axis=1),trans_nspecies.min(axis=1)[::-1]]), fill='toself', line_color='LightSteelBlue', fillcolor='LightSteelBlue', showlegend=False, name='Number of Species'))
                    fig.add_trace(go.Scatter(x=x, y=trans_nspecies.mean(axis=1), line_color='SteelBlue', showlegend=False, name='Number of Species'))
                    fig.update_layout(font={'size': 14},
                                      height=445,
                                      title='<b>Number of Commensal Species on Transconjugant Population Density<b>',
                                      title_font={'size': 20},
                                      xaxis_title='Number of Commensal Species',
                                      xaxis={'tickfont': {'size': 18}},
                                      yaxis_title='Final Transconjugant Pop Density',
                                      yaxis={'tickfont': {'size':18}})
                    return fig
                # Dynamically input number of species
                ui.markdown("""#### Play Around with Community Composition and Starting Densities<br><br>""")
                ui.input_numeric(id='params_n_species', label='Number of Species in the Community:', value=0, min=0, max=26, step=1)
                with ui.layout_columns(col_widths=(4,4,4)):
                    ui.input_numeric(id='params_x0_donor', label='Input Starting Density for Donor Species', value=0.5, min=0.01, step=0.01, width='100%')
                    ui.input_numeric(id='params_x0_recip', label='Input Starting Density for Recipient Species', value=0.5, min=0.01, step=0.01, width='100%')
                    ui.input_numeric(id='params_x0_trans', label='Input Starting Density for Transconjugant Species', value=0, min=0, step=0.01, width='100%')
                # Edit starting densities for community members depending on their number
                with ui.panel_conditional("input.params_n_species >= 1"):
                    ui.markdown("""##### Input Starting Density for Community""")
                with ui.panel_conditional("input.params_n_species >= 1"):  # noqa: SIM117
                    with ui.layout_column_wrap(width='105px'):
                        ui.input_numeric(id='params_x0_spec1', label='Species A', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 2"):
                            ui.input_numeric(id='params_x0_spec2', label='Species B', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 3"):
                            ui.input_numeric(id='params_x0_spec3', label='Species C', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 4"):
                            ui.input_numeric(id='params_x0_spec4', label='Species D', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 5"):
                            ui.input_numeric(id='params_x0_spec5', label='Species E', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 6"):
                            ui.input_numeric(id='params_x0_spec6', label='Species F', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 7"):
                            ui.input_numeric(id='params_x0_spec7', label='Species G', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 8"):
                            ui.input_numeric(id='params_x0_spec8', label='Species H', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 9"):
                            ui.input_numeric(id='params_x0_spec9', label='Species I', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 10"):
                            ui.input_numeric(id='params_x0_spec10', label='Species J', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 11"):
                            ui.input_numeric(id='params_x0_spec11', label='Species K', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 12"):
                            ui.input_numeric(id='params_x0_spec12', label='Species L', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 13"):
                            ui.input_numeric(id='params_x0_spec13', label='Species M', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 14"):
                            ui.input_numeric(id='params_x0_spec14', label='Species N', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 15"):
                            ui.input_numeric(id='params_x0_spec15', label='Species O', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 16"):
                            ui.input_numeric(id='params_x0_spec16', label='Species P', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 17"):
                            ui.input_numeric(id='params_x0_spec17', label='Species Q', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 18"):
                            ui.input_numeric(id='params_x0_spec18', label='Species R', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 19"):
                            ui.input_numeric(id='params_x0_spec19', label='Species S', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 20"):
                            ui.input_numeric(id='params_x0_spec20', label='Species T', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 21"):
                            ui.input_numeric(id='params_x0_spec21', label='Species U', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 22"):
                            ui.input_numeric(id='params_x0_spec22', label='Species V', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 23"):
                            ui.input_numeric(id='params_x0_spec23', label='Species W', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 24"):
                            ui.input_numeric(id='params_x0_spec24', label='Species X', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 25"):
                            ui.input_numeric(id='params_x0_spec25', label='Species Y', value=1, min=0.01, step=0.01)
                        with ui.panel_conditional("input.params_n_species >= 26"):
                            ui.input_numeric(id='params_x0_spec26', label='Species Z', value=1, min=0.01, step=0.01)
                ui.input_action_button(id='species_new_alpha', label='Randomise a new interaction matrix')
                ui.markdown("""<br><br>""")
                with ui.layout_columns(col_widths=[8, 4]):
                    # Info about model
                    with ui.card():
                        with ui.card_header():
                            ui.markdown("""<br><b>Viewing the Model</b>""")
                            with ui.popover(placement="right", id="params_n_species_model"):
                                ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                ui.markdown("""
                                ##### Parameters
                                <u>Parameters for the interaction matrix</u>
                                - `connectivity` (probability of interactions): 1
                                - `sigma` (interspecies interaction strength standard deviation): 0.075
                                - `P_m` (proportion of species reactions that are positive): 0
                                - `self_inhibition` (intra-specific interaction strength): 0.1
                                - `intrinsic_growth` (instrinsic growth rate): 1""")
                                "The interaction matrix used for this run is:"
                                @render.data_frame
                                def n_species_render_interaction_matrix():
                                    return render.DataGrid(change_species_alpha_init, height='114px')
                                @reactive.effect
                                async def _():
                                    await n_species_render_interaction_matrix.update_data(change_species_alpha.get())
                                "The growth rates used for this run is:"
                                @render.data_frame
                                def n_species_render_r():
                                    return render.DataGrid(change_species_r_init)
                                @reactive.effect
                                async def _():
                                    await n_species_render_r.update_data(change_species_r.get())
                                ui.markdown("""
                                <br><u>Parameters for the model</u>
                                - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                - `susceptibility` (donor, recipient and transconjugant): 0.05
                                - `susceptibility` (community): 0.2
                                - `drug_level` (magnitude of perturbation): 0.5""")
                        ui.markdown("""This example simulation lets you play around with different numbers of 'commensal' species in the community, and the starting densities for each species. In this example, after the initial population densities are introduced, the simulation is allowed to find an equilibrium for the first 500 timesteps before introducing the donor and recipient species, as well as the drug which perturbs the community. Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                    # Transconjugant final numbers
                    with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                        "Transconjugant Population Density"
                        @render.ui
                        def n_trans_change_species():
                            return f"{change_species_results.get()['y'][2][-1]:.4f}"
                # Run model with updated inputs
                @reactive.effect
                def _():
                    # Set up parameters
                    global change_species, change_species_alpha_init, change_species_r_init
                    # Determine initial densities
                    params_x0 = [input.params_x0_donor(), input.params_x0_recip(), input.params_x0_trans()]
                    for i in range(1, input.params_n_species() + 1):
                        exec(f"params_x0.append(input.params_x0_spec{i}())")  # noqa: S102
                    # Determine susceptibility
                    params_susceptibility = [0.05, 0.05, 0.05] + ([0.2] * input.params_n_species())
                    # Run model
                    change_species = ode_lotka(input.params_n_species())  # initialise model
                    change_species.randomise_interaction_matrix(connectivity=1, sigma=0.075, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_species_alpha.set( pd.DataFrame(change_species.alpha, columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.params_n_species())]) )
                    change_species_alpha_init = change_species_alpha.get()
                    change_species_r.set( pd.DataFrame(change_species.r.reshape(1,len(change_species.r)), 
                                                       columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.params_n_species())]) )
                    change_species_r_init = change_species_r.get()
                    change_species.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=False, t_drug=500, t_dr=500)  # solve ode
                    change_species_results.set(change_species.results)
                # Generate new interaction matrix for model
                @reactive.effect
                @reactive.event(input.species_new_alpha)
                def _():
                    # Determine initial densities
                    params_x0 = [input.params_x0_donor(), input.params_x0_recip(), input.params_x0_trans()]
                    for i in range(1, input.params_n_species() + 1):
                        exec(f"params_x0.append(input.params_x0_spec{i}())")  # noqa: S102
                    # Determine susceptibility
                    params_susceptibility = [0.05, 0.05, 0.05] + ([0.2] * input.params_n_species())
                    # Run model
                    change_species.randomise_interaction_matrix(connectivity=1, sigma=0.075, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_species_alpha.set( pd.DataFrame(change_species.alpha, columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.params_n_species())]) )
                    change_species_r.set( pd.DataFrame(change_species.r.reshape(1,len(change_species.r)),
                                                       columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.params_n_species())]) )
                    change_species.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=False, t_drug=500, t_dr=500)  # solve ode
                    change_species_results.set(change_species.results)
                # Plot
                @render_plotly(width='100%')
                def plot_change_species():
                    return plot_species_densities(change_species_results.get(), input.params_n_species())


            # BUILDING THE INTERACTION MATRIX           
            with ui.nav_panel("Building the Interaction Matrix"):
                ui.markdown("""
                #### <b>Building the Interaction Matrix</b>
                To build the interaction matrix alpha, $\\alpha_{ij}$, we used some parameters to then determine how the matrix was randomly populated. The matrix can be completely randomly populated based on the parameters using the `randomise_interaction_matrix(*parameters)` function.
                <br><br>
                The matrix can also be entered manually using two different functions. `enter_custom_interaction_matrix(*parameters)` will make some corrections, such as ensuring the transconjugants have the same parameters as the recipients, ensuring the diagonal has the same self-inhibition rates as entered, and will calculate the growth rate and stability from the values entered. There is also `set_interaction_matrix(alpha, intrinsic_growth)` function which lets you set the interaction matrix and growth rate from entered values and automatically assumes the values are good. This web app will not interact with either of those functions, as that can lead to an unsolvable ODE equation. If you want to be able to play with these functions, please download the files from the GitHub and run in your desired Python environment.
                <br><br>
                Here, we'll focus on the effects of changing different parameters to randomise the interaction matrix. To determine if the ODEs can be solved and the system reaches equilibrium with the randomly generated interaction matrix, the Jacobian is calculated and its eigenvalues are found. The interaction matrix and growth rates calculated and used can be displayed using `display_matrix_used()`. For reproducibility, the interaction matrix and growth rates can be saved to .csv files using `save_interaction_matrix_r()`, which can then be reloaded, using `upload_interaction_matrix_r()`.""")
                
                # Go through each of the parameters in building an interaction matrix
                with ui.accordion(id='build_interaction_matrix', open=['Connectivity', 'Self Inhibition & Interaction Strength', 'Proportion of Positive Interactions', 'Intrinsic Growth Rate']):

                    # Connectivity
                    with ui.accordion_panel('Connectivity'):
                        ui.markdown("""
                        $C\\in[0,1]$ `connectivity`: the connectivity between different species in the community, the fraction of all `n_species` species that a single species interacts with 
                        - this is set to $0$ if there is no interaction for different species
                        - if there is an interaction between them, the value of $\\alpha_{ij}$ is assigned to a proportion of entries equal to $C$
                        """)
                        @render_plotly(width='100%')
                        def connectivity_trans():
                            x = np.linspace(0, 1, 51) 
                            x_rev = x[::-1]
                            trans_connectivty = np.load(f'{data_dir}/change_connectivity.npy')
                            # Render figure layouts
                            fig = go.Figure()
                            fig.update_layout(plot_bgcolor='#eff5fa')
                            # Render plot
                            fig.add_trace(go.Scatter(x=np.concat([x,x_rev]), y=np.concat([trans_connectivty.max(axis=1),trans_connectivty.min(axis=1)[::-1]]), fill='toself', line_color='LightSteelBlue', fillcolor='LightSteelBlue', showlegend=False, name='Connectivity'))
                            fig.add_trace(go.Scatter(x=x, y=trans_connectivty.mean(axis=1), line_color='SteelBlue', showlegend=False, name='Connectivity'))
                            fig.update_layout(font={'size': 14},
                                              height=445,
                                              title='<b>Connectivity on Transconjugant Population Density<b>',
                                              title_font={'size': 20},
                                              xaxis_title='Connectivity',
                                              xaxis={'tickfont': {'size': 18}},
                                              yaxis_title='Final Transconjugant Pop Density',
                                              yaxis={'tickfont': {'size':18}})
                            return fig
                        with ui.layout_columns(col_widths=[5, 7]):
                            with ui.layout_columns(col_widths=[12]):
                                with ui.card():
                                    ui.input_slider(id='params_connectivity', label='Input Connectivity', value=0.8, min=0, max=1, step=0.01, width='100%')
                                    ui.input_action_button(id='connectivity_new_alpha', label='Randomise a new interaction matrix', width='100%')
                                # Transconjugant final numbers
                                with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                                    "Transconjugant Population Density"
                                    @render.ui
                                    def n_trans_change_conn():
                                        return f"{change_conn_results.get()['y'][2][-1]:.4f}"
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""<br><b>Viewing the Model</b>""")
                                    with ui.popover(placement="right", id="params_conn_model"):
                                        ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                        ui.markdown("""
                                        ##### Parameters
                                        <u>Parameters for the interaction matrix</u>
                                        - `sigma` (interspecies interaction strength standard deviation): 0.2
                                        - `P_m` (proportion of species reactions that are positive): 0
                                        - `self_inhibition` (intra-specific interaction strength): 0.1
                                        - `intrinsic_growth` (instrinsic growth rate): 1""")
                                        "The interaction matrix used for this run is:"
                                        @render.data_frame
                                        def conn_render_interaction_matrix():
                                            return render.DataGrid(change_conn_alpha_init, height='175px')
                                        @reactive.effect
                                        async def _():
                                            await conn_render_interaction_matrix.update_data(change_conn_alpha.get())
                                        "The growth rates used are:"
                                        @render.data_frame
                                        def conn_render_r():
                                            return render.DataGrid(change_conn_r_init)
                                        @reactive.effect
                                        async def _():
                                            await conn_render_r.update_data(change_conn_r.get())
                                        ui.markdown("""
                                        <br><u>Parameters for the model</u>
                                        - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                        - `x0` (initial community densities): 1
                                        - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                        - `susceptibility` (donor, recipient and transconjugant): 0.05
                                        - `susceptibility` (community): 0.2
                                        - `drug_level` (magnitude of perturbation): 0.5""")
                                ui.markdown("""This example simulation lets you play around with different connectivity, $C$, proportions between species in the community. The more connected the community, the more of an impact the densities of each species will have on each other's growth and population density. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                        # Run model
                        @reactive.effect
                        def _():
                            # Set up parameters
                            global change_conn, change_conn_alpha_init, change_conn_r_init
                            params_x0 = [0.5, 0.5, 0, 1, 1, 1, 1, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2]
                            # Run model
                            change_conn = ode_lotka(n_species=5)  # initialise model
                            change_conn.randomise_interaction_matrix(connectivity=input.params_connectivity(), sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_conn_alpha.set( pd.DataFrame(change_conn.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A','species_B','species_C','species_D','species_E']) )
                            change_conn_alpha_init = change_conn_alpha.get()
                            change_conn_r.set( pd.DataFrame(change_conn.r.reshape(1,len(change_conn.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A','species_B','species_C','species_D','species_E']) )
                            change_conn_r_init = change_conn_r.get()
                            change_conn.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_conn_results.set(change_conn.results)
                        # Generate new interaction matrix for model
                        @reactive.effect
                        @reactive.event(input.connectivity_new_alpha)
                        def _():
                            # Determine strength of interactions - needs lower interactions with more species to solve matrix
                            params_x0 = [0.5, 0.5, 0, 1, 1, 1, 1, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2]
                            # Run model
                            change_conn.randomise_interaction_matrix(connectivity=input.params_connectivity(), sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_conn_alpha.set( pd.DataFrame(change_conn.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A','species_B','species_C','species_D','species_E']) )
                            change_conn_r.set( pd.DataFrame(change_conn.r.reshape(1,len(change_conn.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A','species_B','species_C','species_D','species_E']) )
                            change_conn.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_conn_results.set(change_conn.results)
                        # Plot
                        @render_plotly(width='100%')
                        def plot_change_conn():
                            return plot_species_densities(change_conn_results.get(), 5)

                    # Self Inhibition and Interaction Strength
                    with ui.accordion_panel('Self Inhibition & Interaction Strength'):
                        ui.markdown("""
                        $\\sigma$ `sigma`: the standard deviation of the normal distribution that the strength of the interactions between species is drawn from
                        - the actual value of $\\alpha_{ij}$ given a value between -1 and +1 <br>

                        $s_i$ `self_inhibition`: a vector of the self inhibition rate, or effect of a species on itself (we assume each species competes to the same extent with members of their own species)
                        - diagonal entries in the interaction matrix will be set to have the value $−s_i$ <br>

                        donors, recipients, and transconjugant parameters
                        - between the recipients and transconjugants (and typically the donor), the interaction strength will be set to be the same for all entries
                        - additionally the interaction matrix entries between the recipient and transconjugant (and typically the donor) will be the same as the self-inhibition value""")
                        @render_plotly(width='100%')
                        def sigma_vs_strength():
                            sigma = np.linspace(0, 1.005, int(1.005/0.015)+1)
                            sigma_rev = sigma[::-1]
                            alpha_mean_negative = np.load(f'{data_dir}/change_negative_sigma_alpha.npy')
                            alpha_mean_positive = np.load(f'{data_dir}/change_positive_sigma_alpha.npy')
                            # Render figure layouts
                            fig = go.Figure()
                            fig.update_layout(plot_bgcolor='#eff5fa')
                            # Render plot
                            fig.add_trace(go.Scatter(x=np.concat([sigma,sigma_rev]), y=np.concat([alpha_mean_negative.max(axis=1),alpha_mean_negative.min(axis=1)[::-1]]), fill='toself', line_color='LightSteelBlue', fillcolor='LightSteelBlue', showlegend=False, name='negative'))
                            fig.add_trace(go.Scatter(x=sigma, y=alpha_mean_negative.mean(axis=1), line_color='SteelBlue', showlegend=False,name='negative'))
                            fig.add_trace(go.Scatter(x=np.concat([sigma,sigma_rev]), y=np.concat([alpha_mean_positive.max(axis=1),alpha_mean_positive.min(axis=1)[::-1]]), fill='toself', line_color='LightSteelBlue', fillcolor='LightSteelBlue', showlegend=False, name='positive'))
                            fig.add_trace(go.Scatter(x=sigma, y=alpha_mean_positive.mean(axis=1), line_color='SteelBlue', showlegend=False, name='positive'))
                            fig.update_layout(font={'size': 14},
                                              height=445,
                                              title='<b>σ of Normal Distribution on Average Interaction Strength<b>',
                                              title_font={'size': 20},
                                              xaxis_title='Standard Deviation of Normal Distribution σ',
                                              xaxis={'tickfont': {'size': 18}},
                                              hovermode="x",
                                              yaxis_title='Average Interaction Strength in Matrix',
                                              yaxis={'tickfont': {'size':18}})
                            return fig
                        with ui.layout_columns(col_widths=[5, 7]):
                            with ui.layout_columns(col_widths=[12, 12]):
                                with ui.card():
                                    ui.input_slider(id='params_s_i', label='Input Self-Inhibition Value', value=0.1, min=5e-3, max=1, step=0.001, width='100%')
                                    ui.input_slider(id='params_sigma', label='Input Strength of Interaction', value=0.2, min=0.001, max=1.1, step=0.001, width='100%')
                                    ui.input_action_button(id='sigma_si_new_alpha', label='Randomise a new interaction matrix', width='100%')
                                # Transconjugant final numbers
                                with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                                    "Transconjugant Population Density"
                                    @render.ui
                                    def n_trans_change_sigma_si():
                                        return f"{change_sigma_si_results.get()['y'][2][-1]:.4f}"
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""<br><b>Viewing the Model</b>""")
                                    with ui.popover(placement="right", id="params_sigma_si_model"):
                                        ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                        ui.markdown("""
                                        ##### Parameters
                                        <u>Parameters for the interaction matrix</u>
                                        - `connectivity` (probability of interactions): 1
                                        - `P_m` (proportion of species reactions that are positive): 0
                                        - `intrinsic_growth` (instrinsic growth rate): 1""")
                                        "The interaction matrix used for this run is:"
                                        @render.data_frame
                                        def sigma_si_render_interaction_matrix():
                                            return render.DataGrid(change_sigma_si_alpha_init, height='175px')
                                        @reactive.effect
                                        async def _():
                                            await sigma_si_render_interaction_matrix.update_data(change_sigma_si_alpha.get())
                                        "The growth rates used are:"
                                        @render.data_frame
                                        def sigma_si_render_r():
                                            return render.DataGrid(change_sigma_si_r_init)
                                        @reactive.effect
                                        async def _():
                                            await sigma_si_render_r.update_data(change_sigma_si_r.get())
                                        ui.markdown("""
                                        <br><u>Parameters for the model</u>
                                        - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                        - `x0` (initial community densities): 1
                                        - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                        - `susceptibility` (donor, recipient and transconjugant): 0.05
                                        - `susceptibility` (community): 0.2
                                        - `drug_level` (magnitude of perturbation): 0.5""")
                                ui.markdown("""This example simulation lets you play around with different strengths of interaction (drawn from a normal distribution with standard deviation $\\sigma$) between two community species and the donor/recipients, as well as the self-inhibition values, $s_i$, of all species. $s_i$ values on the diagonals of the matrix must be stronger negative interactions than those between other species for the ODE to be solvable. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                        # Run model
                        @reactive.effect
                        def _():
                            # Set up parameters
                            global change_sigma_si, change_sigma_si_alpha_init, change_sigma_si_r_init
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_sigma_si = ode_lotka(n_species=1)  # initialise model
                            change_sigma_si.randomise_interaction_matrix(connectivity=1, sigma=input.params_sigma(), P_m=0, self_inhibition=input.params_s_i(), intrinsic_growth=1)  # initialise interaction matrix
                            change_sigma_si_alpha.set( pd.DataFrame(change_sigma_si.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sigma_si_alpha_init = change_sigma_si_alpha.get()
                            change_sigma_si_r.set( pd.DataFrame(change_sigma_si.r.reshape(1,len(change_sigma_si.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sigma_si_r_init = change_sigma_si_r.get()
                            change_sigma_si.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_sigma_si_results.set(change_sigma_si.results)
                        # Generate new interaction matrix for model
                        @reactive.effect
                        @reactive.event(input.sigma_si_new_alpha)
                        def _():
                            # Determine strength of interactions - needs lower interactions with more species to solve matrix
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_sigma_si.randomise_interaction_matrix(connectivity=1, sigma=input.params_sigma(), P_m=0, self_inhibition=input.params_s_i(), intrinsic_growth=1)  # initialise interaction matrix
                            change_sigma_si_alpha.set( pd.DataFrame(change_sigma_si.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sigma_si_r.set( pd.DataFrame(change_sigma_si.r.reshape(1,len(change_sigma_si.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sigma_si.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_sigma_si_results.set(change_sigma_si.results)
                        # Plot
                        @render_plotly(width='100%')
                        def plot_change_sigma_si():
                            return plot_species_densities(change_sigma_si_results.get(), 1)

                    # Proportion of Positive Interactions
                    with ui.accordion_panel('Proportion of Positive Interactions'):
                        ui.markdown("""
                        $P_m$ `P_m`: proportion of interactions that are positive, for assigning negative and positive interactions
                        - $P_m=0$: community is purely competitive (all values are less than 0)
                        - $P_m=0.5$: community contains a mixture of all interaction types
                        - $P_m=1$: community is purely cooperative (all values are greater than 0)
                        - these are values taken from a mix of potential interaction types, which are summed up by the $\\alpha_{ij}$ value
                            - $P_e$ (+/-): exploitative
                            - $P_c$ (-/-): competitive
                            - $P_m$ (+/+): cooperation
                            - $P_+$ (+/0): commensalism
                            - $P_−$ (-/0): ammensalism
                            - $P_0$ (0/0): zero interactions
                            - where $P_e+P_c+P_m+P_++P_−=1$""")
                        @render_plotly(width='100%')
                        def strength_vs_density():
                            alpha_mean_negative = np.load(f'{data_dir}/change_negative_sigma_alpha.npy')
                            alpha_mean_positive = np.load(f'{data_dir}/change_positive_sigma_alpha.npy')
                            final_change_x_negative = np.load(f'{data_dir}/change_negative_sigma_finalx.npy')
                            final_change_x_positive = np.load(f'{data_dir}/change_positive_sigma_finalx.npy')
                            final_change_x_mean = np.concat([final_change_x_negative, final_change_x_positive])
                            x = np.concat([alpha_mean_negative, alpha_mean_positive]).flatten()
                            # Render figure layouts
                            fig = go.Figure()
                            fig.update_layout(plot_bgcolor='#eff5fa')
                            # Render plot
                            fig.add_trace(go.Scatter(x=x, y=final_change_x_mean[:,:,0].flatten(), mode='markers', marker={'color':'black', 'size':3, 'opacity':0.25}, name='donor', legendgroup='donor'))
                            fig.add_trace(go.Scatter(x=x, y=final_change_x_mean[:,:,1].flatten(), mode='markers', marker={'color':'sienna', 'size':3, 'opacity':0.25}, name='recipient', legendgroup='recipient'))
                            fig.add_trace(go.Scatter(x=x, y=final_change_x_mean[:,:,2].flatten(), mode='markers', marker={'color':'darkviolet', 'size':3, 'opacity':0.25}, name='transconjugant', legendgroup='transconjugant'))
                            fig.add_trace(go.Scatter(x=x, y=final_change_x_mean[:,:,3].flatten(), mode='markers', marker={'color':'gold', 'size':3, 'opacity':0.25}, name='species_A', legendgroup='species_A'))
                            fig.update_layout(font={'size': 14},
                                              height=445,
                                              title='<b>Average Interaction Strength on Transconjugant Population Density<b>',
                                              title_font={'size': 20},
                                              xaxis_title='Average Interaction Strength in Matrix',
                                              xaxis={'tickfont': {'size': 18}},
                                              yaxis_title='Change from Initial Pop Density',
                                              yaxis={'tickfont': {'size':18}},
                                              legend={"title": 'Species', "groupclick": 'togglegroup'})
                            return fig
                        with ui.layout_columns(col_widths=[5, 7]):
                            with ui.layout_columns(col_widths=[12, 12]):
                                with ui.card():
                                    ui.input_slider(id='params_Pm', label='Input Proportion of Positive Interactions', value=0.2, min=0, max=1, step=0.01, width='100%')
                                    ui.input_action_button(id='Pm_new_alpha', label='Randomise a new interaction matrix', width='100%')
                                # Transconjugant final numbers
                                with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                                    "Transconjugant Population Density"
                                    @render.ui
                                    def n_trans_change_Pm():
                                        return f"{change_Pm_results.get()['y'][2][-1]:.4f}"
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""<br><b>Viewing the Model</b>""")
                                    with ui.popover(placement="right", id="params_Pm_model"):
                                        ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                        ui.markdown("""
                                        ##### Parameters
                                        <u>Parameters for the interaction matrix</u>
                                        - `connectivity` (probability of interactions): 1
                                        - `sigma` (interspecies interaction strength standard deviation): 0.2
                                        - `self_inhibition` (intra-specific interaction strength): 0.1
                                        - `intrinsic_growth` (instrinsic growth rate): 1""")
                                        "The interaction matrix used for this run is:"
                                        @render.data_frame
                                        def Pm_render_interaction_matrix():
                                            return render.DataGrid(change_Pm_alpha_init, height='175px')
                                        @reactive.effect
                                        async def _():
                                            await Pm_render_interaction_matrix.update_data(change_Pm_alpha.get())
                                        "The growth rates used are:"
                                        @render.data_frame
                                        def Pm_render_r():
                                            return render.DataGrid(change_Pm_r_init)
                                        @reactive.effect
                                        async def _():
                                            await Pm_render_r.update_data(change_Pm_r.get())
                                        ui.markdown("""
                                        <br><u>Parameters for the model</u>
                                        - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                        - `x0` (initial community densities): 1
                                        - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                        - `susceptibility` (donor, recipient and transconjugant): 0.05
                                        - `susceptibility` (community): 0.2
                                        - `drug_level` (magnitude of perturbation): 0.5""")
                                ui.markdown("""This example simulation lets you play around with different proportions of positive interactions, $P_m$, that fill the interaction matrix. Positive interactions between species means more cooperation, whereas more negative interactions means competition that will inhibit growth. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                        # Run model
                        @reactive.effect
                        def _():
                            # Set up parameters
                            global change_Pm, change_Pm_alpha_init, change_Pm_r_init
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_Pm = ode_lotka(n_species=1)  # initialise model
                            change_Pm.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=input.params_Pm(), self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_Pm_alpha.set( pd.DataFrame(change_Pm.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_Pm_alpha_init = change_Pm_alpha.get()
                            change_Pm_r.set( pd.DataFrame(change_Pm.r.reshape(1,len(change_Pm.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_Pm_r_init = change_Pm_r.get()
                            change_Pm.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_Pm_results.set(change_Pm.results)
                        # Generate new interaction matrix for model
                        @reactive.effect
                        @reactive.event(input.Pm_new_alpha)
                        def _():
                            # Determine strength of interactions - needs lower interactions with more species to solve matrix
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_Pm.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=input.params_Pm(), self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_Pm_alpha.set( pd.DataFrame(change_Pm.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_Pm_r.set( pd.DataFrame(change_Pm.r.reshape(1,len(change_Pm.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_Pm.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_Pm_results.set(change_Pm.results)
                        # Plot
                        @render_plotly(width='100%')
                        def plot_change_Pm():
                            return plot_species_densities(change_Pm_results.get(), 1)

                    # Intrinsic growth rate
                    with ui.accordion_panel('Intrinsic Growth Rate'):
                        ui.markdown("""
                        Once the interaction matrix and species self-interaction is defined, we can then identify the species growth rate, $r_i$:
                        - we assume the community has a linearly asymptotically stable equilibrium at $x_i \\approx \\text{instrinsic growth} \\in i$, where the default value `instrinsic_growth = 1` $\\pm$ some random noise
                        - intrinsic growth can also be an array provided with `3 + n_species`, so that each community species has their own growth rate
                        - the array `steady_state` is the length of the `3 + n_species` is made, with each of the values being around `intrinsic_growth`, with some random noise around the value
                        - note that even if different intrinsic growth rates are provided for donor/recipient/transconjugant species, as they have the same interaction matrix parameters, they will have the same calculated $r$ values
                        - $r$ is then found by a matrix multiplication of $\\alpha$ and `steady_state`
                        - this introduces a trade-off in growth strategies, with species investing either in their own intrinsic growth rate or growth via cooperation with others, reducing unrealistic explosive behavior in highly cooperative communities""")
                        with ui.layout_columns(col_widths=[7, 5]):
                            with ui.card():
                                ui.markdown("""##### Input Intrinsic Growth Rate for""")
                                with ui.layout_columns(col_widths=[3,3,3,3, 6, 6]):
                                    ui.input_numeric(id='params_growth_donor', label='Donor', value=1, min=0.01, step=0.01)
                                    ui.input_numeric(id='params_growth_recip', label='Recipient', value=1, min=0.01, step=0.01)
                                    ui.input_numeric(id='params_growth_trans', label='Transconjugant', value=1, min=0.01, step=0.01)
                                    ui.input_numeric(id='params_growth_spec1', label='Species A', value=1, min=0.01, step=0.01)
                                    ui.input_action_button(id='equal_growths', label='Set all growth rates to donor', width='100%')
                                    ui.input_action_button(id='growth_new_alpha', label='Randomise a new interaction matrix', width='100%')
                                @reactive.effect
                                @reactive.event(input.equal_growths)
                                def _():
                                    ui.update_numeric(id='params_growth_recip', value=input.params_growth_donor())
                                    ui.update_numeric(id='params_growth_trans', value=input.params_growth_donor())
                                    ui.update_numeric(id='params_growth_spec1',value=input.params_growth_donor())
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""<br><b>Viewing the Model</b>""")
                                    with ui.popover(placement="right", id="params_growth_model"):
                                        ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                        ui.markdown("""
                                        ##### Parameters
                                        <u>Parameters for the interaction matrix</u>
                                        - `connectivity` (probability of interactions): 1
                                        - `sigma` (interspecies interaction strength standard deviation): 0.2
                                        - `self_inhibition` (intra-specific interaction strength): 0.1
                                        - `P_m` (proportion of species reactions that are positive): 0""")
                                        "The interaction matrix used for this run is:"
                                        @render.data_frame
                                        def growth_render_interaction_matrix():
                                            return render.DataGrid(change_growth_alpha_init, height='175px')
                                        @reactive.effect
                                        async def _():
                                            await growth_render_interaction_matrix.update_data(change_growth_alpha.get())
                                        "The growth rates used are:"
                                        @render.data_frame
                                        def growth_render_r():
                                            return render.DataGrid(change_growth_r_init)
                                        @reactive.effect
                                        async def _():
                                            await growth_render_r.update_data(change_growth_r.get())
                                        ui.markdown("""
                                        <br><u>Parameters for the model</u>
                                        - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                        - `x0` (initial community densities): 1
                                        - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                        - `susceptibility` (donor, recipient and transconjugant): 0.05
                                        - `susceptibility` (community): 0.2
                                        - `drug_level` (magnitude of perturbation): 0.5""")
                                ui.markdown("""This example simulation lets you play around with different intrinsic growth rates for each of the species, used to calculate $r$. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                         # Transconjugant final numbers
                        with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                            "Transconjugant Population Density"
                            @render.ui
                            def n_trans_change_growth():
                                return f"{change_growth_results.get()['y'][2][-1]:.4f}"
                        # Run model
                        @reactive.effect
                        def _():
                            # Set up parameters
                            global change_growth, change_growth_alpha_init, change_growth_r_init
                            params_growth = [input.params_growth_donor(), input.params_growth_recip(), input.params_growth_trans(), input.params_growth_spec1()]
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_growth = ode_lotka(n_species=1)  # initialise model
                            change_growth.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=params_growth)  # initialise interaction matrix
                            change_growth_alpha.set( pd.DataFrame(change_growth.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_growth_alpha_init = change_growth_alpha.get()
                            change_growth_r.set( pd.DataFrame(change_growth.r.reshape(1,len(change_growth.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_growth_r_init = change_growth_r.get()
                            change_growth.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_growth_results.set(change_growth.results)
                        # Generate new interaction matrix for model
                        @reactive.effect
                        @reactive.event(input.growth_new_alpha)
                        def _():
                            # Determine strength of interactions - needs lower interactions with more species to solve matrix
                            params_growth = [input.params_growth_donor(), input.params_growth_recip(), input.params_growth_trans(), input.params_growth_spec1()]
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_growth.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=params_growth)  # initialise interaction matrix
                            change_growth_alpha.set( pd.DataFrame(change_growth.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_growth_r.set( pd.DataFrame(change_growth.r.reshape(1,len(change_growth.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_growth.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True)  # solve ode
                            change_growth_results.set(change_growth.results)
                        # Plot
                        @render_plotly(width='100%')
                        def plot_change_growth():
                            return plot_species_densities(change_growth_results.get(), 1)


            # DRUG PERTURBATION        
            with ui.nav_panel("Drug Perturbation"):

                # Changing introduction and removal
                ui.markdown("""
                #### <b>Introducing Drug Perturbation</b>
                The overall `drug_effect` is determined by the drug strength $p$ at time $t$, as well as the `susceptibility` $\\epsilon$ of each species to the drug. The `drug_effect` term calculated is subtracted from each species' population density to model the reduced population growth of a species perturbed by a drug.<br><br>
                The effect of drug perturbation on the model is time-dependent, where the drug is introduced to the system at `t_drug` and removed at `drug_recovery`. The strength of the drug $p$ is then set as an array, `drug_time`. The array is set to `0` at `t < t_drug & t > t_drug_recovery` and set to `drug_level` at `t_drug < t < t_drug_recovery`.<br><br>
                At each timepoint `t`, `drug_time` is then indexed and multipled by the `susceptibility` for each species.""")
                with ui.layout_columns(col_widths=[5, 7]):
                    with ui.layout_columns(col_widths=[12, 12]):
                        with ui.card():
                            ui.input_slider(id='params_tdrug', label='Input Drug Introduction Time', value=0, min=0, max=2000, step=1, width='100%')
                            ui.input_slider(id='params_tdrug_recovery', label='Input Drug Removal Time', value=2000, min=0, max=2000, step=1, width='100%')
                            ui.input_action_button(id='tdrug_new_alpha', label='Randomise a new interaction matrix', width='100%')
                            @reactive.effect
                            def _():
                                if input.params_tdrug_recovery() < input.params_tdrug(): tdrug_recovery = input.params_tdrug() + 1
                                else:  tdrug_recovery = input.params_tdrug_recovery()
                                ui.update_slider(id='params_tdrug_recovery', value=tdrug_recovery)
                        # Transconjugant final numbers
                        with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                            "Transconjugant Population Density"
                            @render.ui
                            def n_trans_change_tdrug():
                                return f"{change_tdrug_results.get()['y'][2][-1]:.4f}"
                    with ui.card():
                        with ui.card_header():
                            ui.markdown("""<br><b>Viewing the Model</b>""")
                            with ui.popover(placement="right", id="params_tdrug_model"):
                                ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                ui.markdown("""
                                ##### Parameters
                                <u>Parameters for the interaction matrix</u>
                                - `connectivity` (probability of interactions): 1
                                - `sigma` (interspecies interaction strength standard deviation): 0.2
                                - `self_inhibition` (intra-specific interaction strength): 0.1
                                - `P_m` (proportion of species reactions that are positive): 0
                                - `intrinsic_growth` (instrinsic growth rate): 1""")
                                "The interaction matrix used for this run is:"
                                @render.data_frame
                                def tdrug_render_interaction_matrix(): return render.DataGrid(change_tdrug_alpha_init, height='175px')
                                @reactive.effect
                                async def _(): await tdrug_render_interaction_matrix.update_data(change_tdrug_alpha.get())
                                "The growth rates used are:"
                                @render.data_frame
                                def tdrug_render_r(): return render.DataGrid(change_tdrug_r_init)
                                @reactive.effect
                                async def _(): await tdrug_render_r.update_data(change_tdrug_r.get())
                                ui.markdown("""
                                <br><u>Parameters for the model</u>
                                - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                - `x0` (initial community densities): 1
                                - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                - `susceptibility` (donor, recipient and transconjugant): 0.05
                                - `susceptibility` (community): 0.2
                                - `drug_level` (magnitude of perturbation): 0.5""")
                        ui.markdown("""This example simulation lets you play around with different introduction and removal times of the drug into the community, allowing you to see perturbation as well as recovery if desired. If `t_drug = 0`, then the drug is entered in the community from the start, otherwise it is entered at the time provided, (or not at all if `t_drug = 2000`).  If `t_drug_removal = 2000`, then the drug is not removed from the community at all and remains for the whole simulation run. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run.""")
                # Run model
                @reactive.effect
                def _():
                    # Set up parameters
                    global change_tdrug, change_tdrug_alpha_init, change_tdrug_r_init
                    params_x0 = [0.5, 0.5, 0, 1]
                    params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                    if input.params_tdrug_recovery() == 0: tdrug_recovery = None
                    else: tdrug_recovery = int(input.params_tdrug_recovery())
                    # Run model
                    change_tdrug = ode_lotka(n_species=1)  # initialise model
                    change_tdrug.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_tdrug_alpha.set( pd.DataFrame(change_tdrug.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdrug_alpha_init = change_tdrug_alpha.get()
                    change_tdrug_r.set( pd.DataFrame(change_tdrug.r.reshape(1,len(change_tdrug.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdrug_r_init = change_tdrug_r.get()
                    change_tdrug.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True, t_drug=int(input.params_tdrug()), t_drug_recovery=tdrug_recovery)  # solve ode
                    change_tdrug_results.set(change_tdrug.results)
                # Generate new interaction matrix for model
                @reactive.effect
                @reactive.event(input.tdrug_new_alpha)
                def _():
                    # Parameters for drug recovery
                    params_x0 = [0.5, 0.5, 0, 1]
                    params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                    if input.params_tdrug_recovery() == 0: tdrug_recovery = None
                    else: tdrug_recovery = int(input.params_tdrug_recovery())
                    # Run model
                    change_tdrug.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_tdrug_alpha.set( pd.DataFrame(change_tdrug.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdrug_r.set( pd.DataFrame(change_tdrug.r.reshape(1,len(change_tdrug.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdrug.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True, t_drug=int(input.params_tdrug()), t_drug_recovery=tdrug_recovery)  # solve ode
                    change_tdrug_results.set(change_tdrug.results)
                # Plot
                @render_plotly(width='100%')
                def plot_change_tdrug():
                    return plot_species_densities(change_tdrug_results.get(), 1)

                # Go through each of the parameters affecting drug perturbation
                with ui.accordion(id='model_perturbation', open=['Drug Strength', 'Susceptibility']):

                    # Drug Strength
                    with ui.accordion_panel('Drug Strength'):
                        ui.markdown("""The drug strength $p$ is given as a proportion on a scale of $0$ to $1$ to be able to modulate the strength of the perturbation.""")
                        @render_plotly(width='100%')
                        def drug_strength_trans():
                            x = np.linspace(0, 1, 101)
                            x_rev = x[::-1]
                            trans_drug_strength = np.load(f'{data_dir}/change_drug_strength.npy')
                            # Render figure layouts
                            fig = go.Figure()
                            fig.update_layout(plot_bgcolor='#eff5fa')
                            # Render plot
                            fig.add_trace(go.Scatter(x=np.concat([x,x_rev]), y=np.concat([trans_drug_strength.max(axis=1),trans_drug_strength.min(axis=1)[::-1]]), fill='toself', line_color='LightSteelBlue', fillcolor='LightSteelBlue', showlegend=False, name='Drug Strength'))
                            fig.add_trace(go.Scatter(x=x, y=trans_drug_strength.mean(axis=1), line_color='SteelBlue', showlegend=False, name='Drug Strength'))
                            fig.update_layout(font={'size': 14},
                                              height=445,
                                              title='<b>Drug Strength on Transconjugant Population Density<b>',
                                              title_font={'size': 20},
                                              xaxis_title='Drug Strength',
                                              xaxis={'tickfont': {'size': 18}},
                                              yaxis_title='Final Transconjugant Pop Density',
                                              yaxis={'tickfont': {'size':18}})
                            return fig
                        with ui.layout_columns(col_widths=[5, 7]):
                            with ui.layout_columns(col_widths=[12]):
                                with ui.card():
                                    ui.input_slider(id='params_drug_strength', label='Input Drug Strength', value=0.5, min=0, max=1, step=0.01, width='100%')
                                    ui.input_action_button(id='drug_strength_new_alpha', label='Randomise a new interaction matrix', width='100%')
                                # Transconjugant final numbers
                                with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                                    "Transconjugant Population Density"
                                    @render.ui
                                    def n_trans_change_drug_strength():
                                        return f"{change_drug_strength_results.get()['y'][2][-1]:.4f}"
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""<br><b>Viewing the Model</b>""")
                                    with ui.popover(placement="right", id="params_drug_strength_model"):
                                        ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                        ui.markdown("""
                                        ##### Parameters
                                        <u>Parameters for the interaction matrix</u>
                                        - `connectivity` (probability of interactions): 1
                                        - `sigma` (interspecies interaction strength standard deviation): 0.2
                                        - `P_m` (proportion of species reactions that are positive): 0
                                        - `self_inhibition` (intra-specific interaction strength): 0.1
                                        - `intrinsic_growth` (instrinsic growth rate): 1""")
                                        "The interaction matrix used for this run is:"
                                        @render.data_frame
                                        def drug_strength_render_interaction_matrix():
                                            return render.DataGrid(change_drug_strength_alpha_init, height='175px')
                                        @reactive.effect
                                        async def _():
                                            await drug_strength_render_interaction_matrix.update_data(change_drug_strength_alpha.get())
                                        "The growth rates used are:"
                                        @render.data_frame
                                        def drug_strength_render_r():
                                            return render.DataGrid(change_drug_strength_r_init)
                                        @reactive.effect
                                        async def _():
                                            await drug_strength_render_r.update_data(change_drug_strength_r.get())
                                        ui.markdown("""
                                        <br><u>Parameters for the model</u>
                                        - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                        - `x0` (initial community densities): 1
                                        - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                        - `susceptibility` (donor, recipient and transconjugant): 0.05
                                        - `susceptibility` (community): 0.2""")
                                ui.markdown("""This example simulation lets you play around with different drug strengths, $p$, on a scale between 0 and 1. The higher the drug strength, the stronger an effect it will have on the bacterial species. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run.  The parameters used to run this interactive model are the same used to generate the graph above.""")
                        # Run model
                        @reactive.effect
                        def _():
                            # Set up parameters
                            global change_drug_strength, change_drug_strength_alpha_init, change_drug_strength_r_init
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_drug_strength = ode_lotka(n_species=1)  # initialise model
                            change_drug_strength.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_drug_strength_alpha.set( pd.DataFrame(change_drug_strength.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_drug_strength_alpha_init = change_drug_strength_alpha.get()
                            change_drug_strength_r.set( pd.DataFrame(change_drug_strength.r.reshape(1,len(change_drug_strength.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_drug_strength_r_init = change_drug_strength_r.get()
                            change_drug_strength.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=input.params_drug_strength(), equilibrium=True)  # solve ode
                            change_drug_strength_results.set(change_drug_strength.results)
                        # Generate new interaction matrix for model
                        @reactive.effect
                        @reactive.event(input.drug_strength_new_alpha)
                        def _():
                            # Determine strength of interactions - needs lower interactions with more species to solve matrix
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_drug_strength.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_drug_strength_alpha.set( pd.DataFrame(change_drug_strength.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_drug_strength_r.set( pd.DataFrame(change_drug_strength.r.reshape(1,len(change_drug_strength.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_drug_strength.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=input.params_drug_strength(), equilibrium=True)  # solve ode
                            change_drug_strength_results.set(change_drug_strength.results)
                        # Plot
                        @render_plotly(width='100%')
                        def plot_change_drug_strength():
                            return plot_species_densities(change_drug_strength_results.get(), 1)

                    # Susceptibility
                    with ui.accordion_panel('Susceptibility'):
                        ui.markdown("""Each species then can have varying susceptibilities $\\epsilon$ to the drug provided, including the donor, recipient and transconjugant species. In the below example, the transconjugant species has the same susceptibility as recipient, unless the donor has a low susceptiblity, in which case it takes the 'donated' antibiotic resistance gene which confers greater resistance.""")
                        @render_plotly(width='100%')
                        def susceptibility_trans():
                            graph_points = np.linspace(0, 1, 21)
                            u, v = np.meshgrid(graph_points, graph_points)
                            z0 = 0*np.ones(u.shape)
                            z1 = 1*np.ones(u.shape)
                            trans_drug_strength = np.load('./Data/change_susceptibility.npy').mean(axis=3)
                            # Render fixed community numbers
                            fig = go.Figure(go.Surface(x=u, y=v, z=z1, coloraxis="coloraxis", surfacecolor=np.rot90(trans_drug_strength[-1, :, ::-1]),
                                                       customdata=np.stack((u, v, trans_drug_strength[-1, :, :]), axis=2),
                                                       hovertemplate="donor: %{y}<br>recipient: %{x}<br>community: %{z}<br>transconjugants: %{customdata[2]}<extra></extra>"))  # community - at 1
                            fig.add_surface(x=u, y=v, z=z0, coloraxis="coloraxis", surfacecolor=np.rot90(trans_drug_strength[0, :, ::-1]),
                                            customdata=np.stack((u, v, trans_drug_strength[0, :, :]), axis=2),
                                            hovertemplate="donor: %{y}<br>recipient: %{x}<br>community: %{z}<br>transconjugants: %{customdata[2]}<extra></extra>")  # community - at 0
                            # Render fixed recipients
                            fig.add_surface(x=u, y=z0, z=v, coloraxis="coloraxis", surfacecolor=trans_drug_strength[:, :, 0],
                                            customdata=np.stack((u, z0, np.rot90(np.flip(trans_drug_strength[:, :, 0]))), axis=2),
                                            hovertemplate="donor: %{y}<br>recipient: %{x}<br>community: %{z}<br>transconjugants: %{customdata[2]}<extra></extra>")  # recipient at 0
                            fig.add_surface(x=u, y=z1, z=v, coloraxis="coloraxis", surfacecolor=trans_drug_strength[:, :, -1],
                                            customdata=np.stack((u, z1, np.rot90(np.flip(trans_drug_strength[:, :, -1]))), axis=2),
                                            hovertemplate="donor: %{y}<br>recipient: %{x}<br>community: %{z}<br>transconjugants: %{customdata[2]}<extra></extra>")  # recipient at 1

                            # Render fixed donors
                            fig.add_surface(x=z0, y=u, z=v, coloraxis="coloraxis", surfacecolor=trans_drug_strength[:, 0, :],
                                            customdata=np.stack((z0, u, np.rot90(np.flip(trans_drug_strength[:, 0, :]))), axis=2),
                                            hovertemplate="donor: %{y}<br>recipient: %{x}<br>community: %{z}<br>transconjugants: %{customdata[2]}<extra></extra>")  # donor at 0
                            fig.add_surface(x=z1, y=u, z=v, coloraxis="coloraxis", surfacecolor=trans_drug_strength[:, -1, :],
                                            customdata=np.stack((z1, u, np.rot90(np.flip(trans_drug_strength[:, -1, :]))), axis=2),
                                            hovertemplate="donor: %{y}<br>recipient: %{x}<br>community: %{z}<br>transconjugants: %{customdata[2]}<extra></extra>")  # donor at 1
                            # Layout
                            fig.update_layout(plot_bgcolor='#eff5fa', 
                                            margin={'l': 0, 'r': 0, 't': 50, 'b': 50},
                                            height=650,
                                            font={'size': 14}, title='<b>Susceptibility on Transconjugant Population Density<b>', title_font={'size': 20},
                                            scene = {'xaxis_title':'Recipient Susceptibility',
                                                     'xaxis':{'tickfont': {'size': 14}, 'autorange': 'reversed'},
                                                     'yaxis_title':'Donor Susceptibility',
                                                     'yaxis':{'tickfont': {'size':14}, 'autorange': 'reversed'},
                                                     'zaxis_title':'Species A Susceptibility',
                                                     'zaxis':{'tickfont': {'size':14}}},
                                            coloraxis_colorbar={'title':'Transconjugant<br>Pop Density'})
                            return fig
                        with ui.layout_columns(col_widths=[6, 6]):
                            with ui.layout_columns(col_widths=[12]):
                                with ui.card():
                                    with ui.layout_columns(col_widths=[6, 6, 6, 6]):
                                        ui.input_slider(id='params_sus_donor', label='Donor', value=0.05, min=0, max=1, step=0.01, width='100%')
                                        ui.input_slider(id='params_sus_recip', label='Recipient', value=0.05, min=0, max=1, step=0.01, width='100%')
                                        ui.input_slider(id='params_sus_trans', label='Transconjugant', value=0.05, min=0, max=1, step=0.01, width='100%')
                                        ui.input_slider(id='params_sus_spec1', label='Species A', value=0.2, min=0, max=1, step=0.01, width='100%')
                                    ui.input_action_button(id='sus_new_alpha', label='Randomise a new interaction matrix', width='100%')
                                # Transconjugant final numbers
                                with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                                    "Transconjugant Population Density"
                                    @render.ui
                                    def n_trans_change_sus():
                                        return f"{change_sus_results.get()['y'][2][-1]:.4f}"
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""<br><b>Viewing the Model</b>""")
                                    with ui.popover(placement="right", id="params_sus_model"):
                                        ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                        ui.markdown("""
                                        ##### Parameters
                                        <u>Parameters for the interaction matrix</u>
                                        - `connectivity` (probability of interactions): 1
                                        - `sigma` (interspecies interaction strength standard deviation): 0.2
                                        - `P_m` (proportion of species reactions that are positive): 0
                                        - `self_inhibition` (intra-specific interaction strength): 0.1
                                        - `intrinsic_growth` (instrinsic growth rate): 1""")
                                        "The interaction matrix used for this run is:"
                                        @render.data_frame
                                        def sus_render_interaction_matrix():
                                            return render.DataGrid(change_sus_alpha_init, height='175px')
                                        @reactive.effect
                                        async def _():
                                            await sus_render_interaction_matrix.update_data(change_sus_alpha.get())
                                        "The growth rates used are:"
                                        @render.data_frame
                                        def sus_render_r():
                                            return render.DataGrid(change_sus_r_init)
                                        @reactive.effect
                                        async def _():
                                            await sus_render_r.update_data(change_sus_r.get())
                                        ui.markdown("""
                                        <br><u>Parameters for the model</u>
                                        - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                        - `x0` (initial community densities): 1
                                        - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                        - `drug_strength`: 0.5""")
                                ui.markdown("""This example simulation lets you play around with different donor, recipient, transconjugant and community susceptibilities, $\\epsilon$, as a scale between 0 and 1. The higher the drug susceptibility, the stronger an effect it will have on the bacterial species. While in the example, the donor and transconjugants have the same susceptibilities, they can have differing susceptibilities - ie. gaining the plasmid gives an improved resistance compared to the recipient. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                        # Run model
                        @reactive.effect
                        def _():
                            # Set up parameters
                            global change_sus, change_sus_alpha_init, change_sus_r_init
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [input.params_sus_donor(), input.params_sus_recip(), input.params_sus_trans(), input.params_sus_spec1()]
                            # Run model
                            change_sus = ode_lotka(n_species=1)  # initialise model
                            change_sus.randomise_interaction_matrix(connectivity=1, sigma=0.6, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_sus_alpha.set( pd.DataFrame(change_sus.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sus_alpha_init = change_sus_alpha.get()
                            change_sus_r.set( pd.DataFrame(change_sus.r.reshape(1,len(change_sus.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sus_r_init = change_sus_r.get()
                            change_sus.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.1, equilibrium=True)  # solve ode
                            change_sus_results.set(change_sus.results)
                        # Generate new interaction matrix for model
                        @reactive.effect
                        @reactive.event(input.sus_new_alpha)
                        def _():
                            # Determine strength of interactions - needs lower interactions with more species to solve matrix
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [input.params_sus_donor(), input.params_sus_recip(), input.params_sus_trans(), input.params_sus_spec1()]
                            # Run model
                            change_sus.randomise_interaction_matrix(connectivity=1, sigma=0.6, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_sus_alpha.set( pd.DataFrame(change_sus.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sus_r.set( pd.DataFrame(change_sus.r.reshape(1,len(change_sus.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_sus.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.1, equilibrium=True)  # solve ode
                            change_sus_results.set(change_sus.results)
                        # Plot
                        @render_plotly(width='100%')
                        def plot_change_sus():
                            return plot_species_densities(change_sus_results.get(), 1)


            # MODELLING CONJUGATION       
            with ui.nav_panel("Modelling Conjugation"):
                ui.markdown("""
                #### <b>Modelling Conjugation</b>
                The donor and recipient are innoculated at `t_dr`, which can be any time later than `t = 0`. Both when running with an equilibrium step and when running without one and `t < t_dr`; the donor, recipient and transconjugant values are set to 0. During this time, the `plasmid_transer` term $\\gamma=0$ also ensures that no transconjugation occurs. Afterwards, when `t > t_dr`, the population density for the community calculated by the ODE solver is carried forward, but the donor and recipient values are taken from the `x0` value inputted by the user.""")
                with ui.layout_columns(col_widths=[5, 7]):
                    with ui.layout_columns(col_widths=[12, 12]):
                        with ui.card():
                            ui.input_slider(id='params_tdr', label='Input Donor and Recipient Introduction Time', value=0, min=0, max=2000, step=1, width='100%')
                            ui.input_action_button(id='tdr_new_alpha', label='Randomise a new interaction matrix', width='100%')
                        # Transconjugant final numbers
                        with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                            "Transconjugant Population Density"
                            @render.ui
                            def n_trans_change_tdr():
                                return f"{change_tdr_results.get()['y'][2][-1]:.4f}"
                    with ui.card():
                        with ui.card_header():
                            ui.markdown("""<br><b>Viewing the Model</b>""")
                            with ui.popover(placement="right", id="params_tdr_model"):
                                ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                ui.markdown("""
                                ##### Parameters
                                <u>Parameters for the interaction matrix</u>
                                - `connectivity` (probability of interactions): 1
                                - `sigma` (interspecies interaction strength standard deviation): 0.2
                                - `self_inhibition` (intra-specific interaction strength): 0.1
                                - `P_m` (proportion of species reactions that are positive): 0
                                - `intrinsic_growth` (instrinsic growth rate): 1""")
                                "The interaction matrix used for this run is:"
                                @render.data_frame
                                def tdr_render_interaction_matrix(): return render.DataGrid(change_tdr_alpha_init, height='175px')
                                @reactive.effect
                                async def _(): await tdr_render_interaction_matrix.update_data(change_tdr_alpha.get())
                                "The growth rates used are:"
                                @render.data_frame
                                def tdr_render_r(): return render.DataGrid(change_tdr_r_init)
                                @reactive.effect
                                async def _(): await tdr_render_r.update_data(change_tdr_r.get())
                                ui.markdown("""
                                <br><u>Parameters for the model</u>
                                - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                - `x0` (initial community densities): 1
                                - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                - `susceptibility` (donor, recipient and transconjugant): 0.05
                                - `susceptibility` (community): 0.2
                                - `drug_level` (magnitude of perturbation): 0.5""")
                        ui.markdown("""This example simulation lets you play around with different introduction and removal times of the drug into the community. This allows you to see perturbation as well as recovery if desired. If `t_drug = 0`, then the drug is entered in the community from the start, otherwise it is entered at the time provided, (or not at all if `t_drug = 2000`).  If `t_drug_removal = 2000`, then the drug is not removed from the community at all and remains for the whole simulation run. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix generated for this run.""")
                # Run model
                @reactive.effect
                def _():
                    # Set up parameters
                    global change_tdr, change_tdr_alpha_init, change_tdr_r_init
                    params_x0 = [0.5, 0.5, 0, 1]
                    params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                    # Run model
                    change_tdr = ode_lotka(n_species=1)  # initialise model
                    change_tdr.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_tdr_alpha.set( pd.DataFrame(change_tdr.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdr_alpha_init = change_tdr_alpha.get()
                    change_tdr_r.set( pd.DataFrame(change_tdr.r.reshape(1,len(change_tdr.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdr_r_init = change_tdr_r.get()
                    change_tdr.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True, t_dr=int(input.params_tdr()))  # solve ode
                    change_tdr_results.set(change_tdr.results)
                # Generate new interaction matrix for model
                @reactive.effect
                @reactive.event(input.tdr_new_alpha)
                def _():
                    # Parameters
                    params_x0 = [0.5, 0.5, 0, 1]
                    params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                    # Run model
                    change_tdr.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_tdr_alpha.set( pd.DataFrame(change_tdr.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdr_r.set( pd.DataFrame(change_tdr.r.reshape(1,len(change_tdr.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_tdr.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True, t_dr=int(input.params_tdr()))  # solve ode
                    change_tdr_results.set(change_tdr.results)
                # Plot
                @render_plotly(width='100%')
                def plot_change_tdr():
                    return plot_species_densities(change_tdr_results.get(), 1)

                ui.markdown("""
                To model conjugation, the system of equations uses $\\gamma$ for only the recipient and transconjugant species. This float value `plasmid_transfer` is removed from the recipient population density, and is added to the transconjugant population density values. The final number in the transconjugant population is then used to measure the conjugation that has occurred. There are then two methods that can be used to determine plasmid transfer.<br><br>
                #### Simple Conjugation
                The `simple` method takes `conjugation_rate` $\\Gamma$ and determines the plasmid_transfer rate: 
                $$\\gamma=x_\\text{recipients}[\\Gamma(x_\\text{donors}+x_\\text{transconjugants})]$$
                This therefore takes into account the number of donor and transconjugants that can donate the plasmid, and the number of transconjugants that could recieve it - accounting for physical number and potential spatial limitations of coming into contact and donating the plasmid. <br><br>
                This graph compares the final transconjugant population density with varied conjugation rate.""")
                @render_plotly(width='100%')
                def simple_conj_trans():
                    x = np.logspace(-6, -1, 501, base=10)*5
                    x_rev = x[::-1]
                    trans_simple_conj = np.load(f'{data_dir}/change_conjugation_rate.npy')
                    # Render figure layouts
                    fig = go.Figure()
                    fig.update_layout(plot_bgcolor='#eff5fa')
                    # Render plot
                    fig.add_trace(go.Scatter(x=np.concat([x,x_rev]), y=np.concat([trans_simple_conj.max(axis=1),trans_simple_conj.min(axis=1)[::-1]]), fill='toself', line_color='LightSteelBlue', fillcolor='LightSteelBlue', showlegend=False, name='Conjugation Rate'))
                    fig.add_trace(go.Scatter(x=x, y=trans_simple_conj.mean(axis=1), line_color='SteelBlue', showlegend=False, name='Conjugation Rate'))
                    fig.update_layout(font={'size': 14},
                                      height=445,
                                      title='<b>Altering Conjugation Parameters on Transconjugant Population Density<b>',
                                      title_font={'size': 20},
                                      xaxis_title='Conjugation Rate',
                                      xaxis={'tickfont': {'size': 18}, 'type': 'log'},
                                      yaxis_title='Strength of Spatial Inhibition',
                                      yaxis={'tickfont': {'size':18}})
                    return fig
                ui.markdown("""
                #### Spatial Competition on Conjugation
                The second method `proportion` also takes into account any other species in the community, and any spatial inhibition they may provide to conjugation. Even if species don't have any interaction via the interaction matrix, there should be an effect on conjugation rate due to the competition for space, and the limited ability for the donor to physically contact the recipient. The method takes `conjugation_rate` $\\Gamma$ and determines the `plasmid_transfer` rate:
                $$\\gamma=x_\\text{recipients}[\\Gamma(x_\\text{donors}+x_\\text{transconjugants})]\\times\\rho\\times\\frac{1}{\\lambda}$$
                where $\\rho = \\text{prop}\\_\\text{donor + transconjugant} \\times \\text{prop}\\_\\text{recipient}$<br><br>
                This method takes the simple `plasmid_transfer` rate and multiplies it by the proportion of cells in the population that can donate the plasmid and the proportion of cells in the population that can recieve the plasmid. Therefore, the more cells there are the community, the lower the proportion of donors and recipients, and the less likely there are to come into contact with each other to allow conjugation to occur, even if there are high numbers of donor and recipient cells. This method is modulated by `prop_strength` $\\lambda\\in[0,1]$. The user entered value is then rescaled to the range $\\lambda\\in[\\rho,1]$ to prevent a divide by 0 error and take into account the strength of the effect.""")
                @render_plotly(width='100%')
                def spatial_conj_trans():
                    x = np.logspace(-6, -1, 501, base=10)*5
                    y = np.linspace(0, 1, 101)
                    trans_spatial_conj = np.load(f'{data_dir}/change_prop_strength.npy').mean(axis=2)
                    # Render figure layouts
                    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=trans_spatial_conj.T,
                                                    colorbar={'title':'Transconjugant<br>Pop Density'},
                                                    hovertemplate="conjugation rate: %{x}<br>spatial inhibition strength: %{y}<br>transconjugant pop density: %{z}<extra></extra>"))
                    fig.update_layout(plot_bgcolor='#eff5fa',
                                      height=445,
                                      font={'size': 14},
                                      title='<b>Conjugation Rate on Change in Transconjugant Population Density<b>',
                                      title_font={'size': 20},
                                      xaxis_title='Conjugation Rate',
                                      xaxis={'tickfont': {'size': 18}, 'type': 'log'},
                                      yaxis_title='Strength of Spatial Inhibition',
                                      yaxis={'tickfont': {'size':18}})
                    return fig
                # Play with inhibitions
                ui.markdown("""<br>""")
                with ui.accordion(id='model_interaction_vs_spatial', open=[]):  # noqa: SIM117
                    with ui.accordion_panel('Compare the Inhibition on Conjugation of Interaction Strength vs Spatial Inhibition'):
                        @render_plotly(width='100%')
                        def inhibitions_trans():
                            x = np.linspace(0, 1.005, int(1.005/0.015)+1)
                            y = np.linspace(0, 1, 101)
                            trans_spatial_conj = np.load(f'{data_dir}/change_inhibition.npy').mean(axis=2)
                            # Render figure layouts
                            fig = go.Figure(data=go.Heatmap(x=x, y=y, z=trans_spatial_conj.T, zsmooth='best',
                                                            colorbar={'title':'Transconjugant<br>Pop Density'},
                                                            hovertemplate="interaction strength σ: %{x}<br>spatial inhibition strength: %{y}<br>transconjugant pop density: %{z}<extra></extra>"))
                            fig.update_layout(plot_bgcolor='#eff5fa',
                                              height=445,
                                              font={'size': 14},
                                              title='<b>Inhibition on Change in Transconjugant Population Density<b>',
                                              title_font={'size': 20},
                                              xaxis_title='Standard Deviation of Normal Distribution for Interaction Strength, σ',
                                              xaxis={'tickfont': {'size': 18}},
                                              yaxis_title='Strength of Spatial Inhibition',
                                              yaxis={'tickfont': {'size':18}})
                            return fig
                        with ui.layout_columns(col_widths=[6, 6]):
                            with ui.layout_columns(col_widths=[12]):
                                with ui.card():
                                    ui.input_slider(id='params_compare_sigma', label='Input Strength of Interaction', value=0.2, min=0.001, max=1.1, step=0.001, width='100%')
                                    ui.input_slider(id='params_compare_spatial', label='Input Strength of Spatial Inhibition', value=0, min=0, max=1, step=0.01, width='100%')
                                    ui.input_action_button(id='compare_conj_new_alpha', label='Randomise a new interaction matrix', width='100%')
                                # Transconjugant final numbers
                                with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                                    "Transconjugant Population Density"
                                    @render.ui
                                    def n_trans_change_compare_conj():
                                        return f"{change_compare_conj_results.get()['y'][2][-1]:.4f}"
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""<br><b>Viewing the Model</b>""")
                                    with ui.popover(placement="right", id="params_compare_conj_model"):
                                        ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                        ui.markdown("""
                                        ##### Parameters
                                        <u>Parameters for the interaction matrix</u>
                                        - `connectivity` (probability of interactions): 1
                                        - `P_m` (proportion of species reactions that are positive): 0
                                        - `self_inhibition` (intra-specific interaction strength): 0.1
                                        - `intrinsic_growth` (instrinsic growth rate): 1""")
                                        "The interaction matrix used for this run is:"
                                        @render.data_frame
                                        def compare_conj_render_interaction_matrix():
                                            return render.DataGrid(change_compare_conj_alpha_init, height='175px')
                                        @reactive.effect
                                        async def _():
                                            await compare_conj_render_interaction_matrix.update_data(change_compare_conj_alpha.get())
                                        "The growth rates used are:"
                                        @render.data_frame
                                        def compare_conj_render_r():
                                            return render.DataGrid(change_compare_conj_r_init)
                                        @reactive.effect
                                        async def _():
                                            await compare_conj_render_r.update_data(change_compare_conj_r.get())
                                        ui.markdown("""
                                        <br><u>Parameters for the model</u>
                                        - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                        - `x0` (initial community densities): 1
                                        - `conjugation_rate` (average plasmid conjugation rate): $5\\times 10^{-4}$
                                        - `drug_strength`: 0.5
                                        - `susceptibility` (donor, recipient and transconjugant): 0.05
                                        - `susceptibility` (community): 0.2""")
                                ui.markdown("""This example simulation lets you play around with the parameters that inhibit conjugation, to identify the contributions of competitive interactions (values determined by $\\sigma$) vs spatial inhibition (given by $\\rho$) and the number of transconjugants formed. In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                        # Run model
                        @reactive.effect
                        def _():
                            # Set up parameters
                            global change_compare_conj, change_compare_conj_alpha_init, change_compare_conj_r_init
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_compare_conj = ode_lotka(n_species=1)  # initialise model
                            change_compare_conj.randomise_interaction_matrix(connectivity=1, sigma=input.params_compare_sigma(), P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_compare_conj_alpha.set( pd.DataFrame(change_compare_conj.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_compare_conj_alpha_init = change_compare_conj_alpha.get()
                            change_compare_conj_r.set( pd.DataFrame(change_compare_conj.r.reshape(1,len(change_compare_conj.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_compare_conj_r_init = change_compare_conj_r.get()
                            change_compare_conj.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True, plasmid_method='proportion', prop_strength=input.params_compare_spatial())  # solve ode
                            change_compare_conj_results.set(change_compare_conj.results)
                        # Generate new interaction matrix for model
                        @reactive.effect
                        @reactive.event(input.compare_conj_new_alpha)
                        def _():
                            # Determine strength of interactions - needs lower interactions with more species to solve matrix
                            params_x0 = [0.5, 0.5, 0, 1]
                            params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                            # Run model
                            change_compare_conj.randomise_interaction_matrix(connectivity=1, sigma=input.params_compare_sigma(), P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                            change_compare_conj_alpha.set( pd.DataFrame(change_compare_conj.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_compare_conj_r.set( pd.DataFrame(change_compare_conj.r.reshape(1,len(change_compare_conj.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                            change_compare_conj.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=5e-4, susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True, plasmid_method='proportion', prop_strength=input.params_compare_spatial())  # solve ode
                            change_compare_conj_results.set(change_compare_conj.results)
                        # Plot
                        @render_plotly(width='100%')
                        def plot_change_compare_conj():
                            return plot_species_densities(change_compare_conj_results.get(), 1)
                # Play with conjugation parameters
                ui.markdown("""<br>""")
                ui.markdown("""#### Alter Conjugation Parameters""")
                with ui.layout_columns(col_widths=[6, 6]):
                    with ui.layout_columns(col_widths=[12]):
                        with ui.card():
                            ui.input_numeric(id='params_simple_conj', label='Input Conjugation Rate', value=5e-4, min=5e-10, max=5, width='100%')
                            ui.input_slider(id='params_spatial_conj', label='Input Strength of Spatial Inhibition', value=0, min=0, max=1, step=0.01, width='100%')
                            ui.input_action_button(id='conj_new_alpha', label='Randomise a new interaction matrix')
                        # Transconjugant final numbers
                        with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                            "Transconjugant Population Density"
                            @render.ui
                            def n_trans_change_conj():
                                return f"{change_conj_results.get()['y'][2][-1]:.4f}"
                    with ui.card():
                        with ui.card_header():
                            ui.markdown("""<br><b>Viewing the Model</b>""")
                            with ui.popover(placement="right", id="params_conj_model"):
                                ui.span(icon('circle-question', height="30px", width="30px"), style="position:absolute; top: 18px; right: 10px;")
                                ui.markdown("""
                                ##### Parameters
                                <u>Parameters for the interaction matrix</u>
                                - `connectivity` (probability of interactions): 1
                                - `sigma` (interspecies interaction strength standard deviation): 0.2
                                - `P_m` (proportion of species reactions that are positive): 0
                                - `self_inhibition` (intra-specific interaction strength): 0.1
                                - `intrinsic_growth` (instrinsic growth rate): 1""")
                                "The interaction matrix used for this run is:"
                                @render.data_frame
                                def conj_render_interaction_matrix():
                                    return render.DataGrid(change_conj_alpha_init, height='175px')
                                @reactive.effect
                                async def _():
                                    await conj_render_interaction_matrix.update_data(change_conj_alpha.get())
                                "The growth rates used are:"
                                @render.data_frame
                                def conj_render_r():
                                    return render.DataGrid(change_conj_r_init)
                                @reactive.effect
                                async def _():
                                    await conj_render_r.update_data(change_conj_r.get())
                                ui.markdown("""
                                <br><u>Parameters for the model</u>
                                - `x0` (initial donor/recipient densities): 0.5 (no transconjugants to start)
                                - `x0` (initial community densities): 1
                                - `drug_strength`: 0.5
                                - `susceptibility` (donor, recipient and transconjugant): 0.05
                                - `susceptibility` (community): 0.2""")
                        ui.markdown("""This example simulation lets you play around with different conjugation rates, $\\Gamma$, which is measured on a log scale. It also lets you adjust the strength of spatial inhibition, $\\rho$ on conjugation for the `proportion` method, where 0 means no spatial inhibition (only using the `simple` method). In this example, the community is allowed to find equilibrium before the donor and recipient are introduced (not plotted). Click the more info icon to see more information about the model, including the interaction matrix and growth rates generated for this run. The parameters used to run this interactive model are the same used to generate the graph above.""")
                # Run model
                @reactive.effect
                def _():
                    # Set up parameters
                    global change_conj, change_conj_alpha_init, change_conj_r_init
                    params_x0 = [0.5, 0.5, 0, 1]
                    params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                    # Run model
                    change_conj = ode_lotka(n_species=1)  # initialise model
                    change_conj.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_conj_alpha.set( pd.DataFrame(change_conj.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_conj_alpha_init = change_conj_alpha.get()
                    change_conj_r.set( pd.DataFrame(change_conj.r.reshape(1,len(change_conj.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_conj_r_init = change_conj_r.get()
                    change_conj.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=input.params_simple_conj(), susceptibility=params_susceptibility, drug_level=0.5, equilibrium=True, plasmid_method='proportion', prop_strength=input.params_spatial_conj())  # solve ode
                    change_conj_results.set(change_conj.results)
                # Generate new interaction matrix for model
                @reactive.effect
                @reactive.event(input.conj_new_alpha)
                def _():
                    # Determine strength of interactions - needs lower interactions with more species to solve matrix
                    params_x0 = [0.5, 0.5, 0, 1]
                    params_susceptibility = [0.05, 0.05, 0.05, 0.2]
                    # Run model
                    change_conj.randomise_interaction_matrix(connectivity=1, sigma=0.2, P_m=0, self_inhibition=0.1, intrinsic_growth=1)  # initialise interaction matrix
                    change_conj_alpha.set( pd.DataFrame(change_conj.alpha, columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_conj_r.set( pd.DataFrame(change_conj.r.reshape(1,len(change_conj.r)), columns=['donor', 'recipient', 'transconjugants', 'species_A']) )
                    change_conj.solve_ode(t_start=0, t_end=2000, x0=params_x0, conjugation_rate=input.params_simple_conj(), susceptibility=params_susceptibility, drug_level=0.1, equilibrium=True, plasmid_method='proportion', prop_strength=input.params_spatial_conj())  # solve ode
                    change_conj_results.set(change_conj.results)
                # Plot
                @render_plotly(width='100%')
                def plot_change_conj():
                    return plot_species_densities(change_conj_results.get(), 1)


    # RUN THE MODEL PANEL
    with ui.nav_panel("Run Model"):

        # Markdown
        ui.markdown("""
        ## **Running the Model**
        Here you can adjust any and all parameters to create a custom run of the model!<br><br>""")

        # Parameters
        with ui.layout_columns(col_widths=(4,4,4, 4,4,4, 12, 12, 6,6, 12), fill=False):
            # Interaction Matrix
            with ui.card(fill=False):
                with ui.card_header():
                    ui.markdown("""###### Interaction Matrix Parameters""")
                    with ui.popover(placement="right", id="params_model"):
                        ui.span(icon('circle-question', height="20px", width="20px"), style="position:absolute; top: 6px; right: 10px;")
                        "The interaction matrix used for this run is:"
                        @render.data_frame
                        def model_render_interaction_matrix():
                            return render.DataGrid(model_alpha_init)
                        @reactive.effect
                        async def _():
                            await model_render_interaction_matrix.update_data(model_alpha.get())
                        ui.markdown("<br>The growth rates used are:")
                        @render.data_frame
                        def model_render_r():
                            return render.DataGrid(model_r_init)
                        @reactive.effect
                        async def _():
                            await model_render_r.update_data(model_r.get())
                        ui.markdown("""<br>""")
                        with ui.layout_columns(col_widths=(6,6)):
                            @render.download(label="Download the interaction matrix file", filename='alpha.csv')
                            def alpha_files():
                                yield pd.DataFrame(model.alpha).to_csv()
                            @render.download(label="Download the growth rate file", filename='r.csv')
                            def r_files():
                                yield pd.DataFrame(model.r).to_csv()
                ui.input_checkbox(id='dr_equal', label='Donor and Recipient are Same Species?', value=True, width='100%')
                with ui.layout_columns(col_widths=(6,6, 6,6, 12, 6,6)):
                    ui.input_slider(id='connectivity', label='Connectivity', value=1, min=0, max=1, step=0.01, width='100%')
                    ui.input_slider(id='Pm', label='Positive Interactions', value=0, min=0, max=1, step=0.01, width='100%')
                    ui.input_slider(id='s_i', label='Self-Inhibition Value', value=0.1, min=1e-4, max=1, step=0.0001, width='100%')
                    ui.input_slider(id='sigma', label='Strength of Interaction', value=0.2, min=0.001, max=1.1, step=0.001, width='100%')
                    @render.text
                    def reenter_params():
                        if model_alpha_checks.get() == -1: return "The interaction matrix is hard to solve, re-enter different parameters."
                        else: return "The interaction matrix entered should be solveable."
                    ui.input_file("alpha_r_upload", "Upload an alpha and r .csv file!", multiple=True)  
                    @reactive.effect
                    def _():
                        set_alpha = False
                        set_r = False
                        files: list[FileInfo] | None = input.alpha_r_upload()
                        if files is not None:
                            for file in files:
                                if 'alpha' in file['name']:
                                    model.alpha = pd.read_csv(file["datapath"], index_col=0).to_numpy()
                                    n_species = model.alpha.shape[0] - 3
                                    ui.update_numeric(id='n_species', value=n_species)
                                    model.good_alpha = 1
                                    model_alpha.set( pd.DataFrame(model.alpha, columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(n_species)]) )
                                    set_alpha = True
                                elif 'r' in file['name']:
                                    model.r = pd.read_csv(file["datapath"], index_col=0).to_numpy().T[0]
                                    n_species = model.r.shape[0] - 3
                                    ui.update_numeric(id='n_species', value=n_species)
                                    model_r.set( pd.DataFrame(model.r.reshape(1,len(model.r)), columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(n_species)]) )
                                    set_r = True
                        if set_alpha and set_r:
                            x0 = [input.x0_donor(), input.x0_recip(), input.x0_trans()]
                            susceptibility = [input.sus_donor(), input.sus_recip(), input.sus_trans()]
                            for i in range(1, n_species + 1):
                                exec(f"x0.append(input.x0_spec{i}())")  # noqa: S102
                                exec(f"susceptibility.append(input.sus_spec{i}())")  # noqa: S102
                            model.solve_ode(t_start=0, t_end=input.tend(), x0=x0, conjugation_rate=input.simple_conj(), susceptibility=susceptibility, drug_level=input.drug_strength(),
                                            equilibrium=input.run_equilibrium(), plasmid_method='proportion', prop_strength=input.spatial_conj(),
                                            t_dr=input.tdr(), t_drug=input.tdrug(), t_drug_recovery=input.tdrug_recovery())  # solve ode
                            model_results.set(model.results)
                    ui.input_action_button(id='model_new_alpha', label='Randomly generate a new interaction matrix', width='100%')
                    @reactive.effect
                    @reactive.event(input.model_new_alpha)
                    def _():
                        # Set up parameters
                        for i in range(1, input.n_species() + 1):
                            exec(f"x0.append(input.x0_spec{i}())")  # noqa: S102
                            exec(f"growth.append(input.growth_spec{i}())")  # noqa: S102
                            exec(f"sus.append(input.sus_spec{i}())")  # noqa: S102
                        # Run model
                        model.randomise_interaction_matrix(connectivity=input.connectivity(), sigma=input.sigma(), P_m=input.Pm(), self_inhibition=input.s_i(), intrinsic_growth=growth, donor_equal_recipient=input.dr_equal())
                        model_alpha_checks.set(model.good_alpha)
                        if model_alpha_checks.get() == 1:
                            model_alpha.set( pd.DataFrame(model.alpha, columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())]) )
                            model_r.set( pd.DataFrame(model.r.reshape(1,len(model.r)),
                                                        columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())]) )
                            model.solve_ode(t_start=0, t_end=input.tend(), x0=x0, conjugation_rate=input.simple_conj(), susceptibility=susceptibility, drug_level=input.drug_strength(),
                                            equilibrium=input.run_equilibrium(), plasmid_method='proportion', prop_strength=input.spatial_conj(),
                                            t_dr=input.tdr(), t_drug=input.tdrug(), t_drug_recovery=input.tdrug_recovery())  # solve ode
                            model_results.set(model.results)
            # Drug Parameters
            with ui.card():
                with ui.card_header():
                    ui.markdown("""###### Drug Parameters""")
                ui.input_slider(id='tdrug', label='Drug Introduction Time', value=0, min=0, max=2000, step=1, width='100%')
                ui.input_slider(id='tdrug_recovery', label='Drug Removal Time', value=2000, min=0, max=2000, step=1, width='100%')
                @reactive.effect
                def _():
                    if input.tdrug() > input.tend(): tdrug = input.tend()
                    else: tdrug = input.tdrug()
                    if input.tdrug_recovery() > input.tend(): tdrug_recovery = input.tend()
                    else: tdrug_recovery = input.tdrug_recovery()
                    tdrug_recovery = max(tdrug_recovery, tdrug)
                    ui.update_slider(id='tdrug', value=tdrug, max=input.tend())
                    ui.update_slider(id='tdrug_recovery', value=tdrug_recovery, max=input.tend())
                ui.input_slider(id='drug_strength', label='Drug Strength', value=0.5, min=0, max=1, step=0.01, width='100%')                                                       
            # Conjugation Parameters
            with ui.card():
                with ui.card_header():
                    ui.markdown("""###### Conjugation Parameters""")
                ui.input_slider(id='tdr', label='Donor and Recipient Introduction Time', value=0, min=0, max=2000, step=1, width='100%')
                @reactive.effect
                def _():
                    if input.tdr() > input.tend(): tdr = input.tend()
                    else: tdr = input.tdr()
                    ui.update_slider(id='tdr', value=tdr, max=input.tend())
                ui.input_numeric(id='simple_conj', label='Conjugation Rate', value=5e-4, min=5e-10, max=5, width='100%')
                ui.input_slider(id='spatial_conj', label='Strength of Spatial Inhibition', value=0, min=0, max=1, step=0.01, width='100%')
            # Donor parameters
            with ui.card(fill=False):
                with ui.card_header():
                    ui.markdown("""###### Donor Parameters""")
                with ui.layout_columns(col_widths=(6,6,6,6)):
                    ui.input_numeric(id='x0_donor', label='Starting Density', value=0.5, min=0.01, step=0.01, width='100%')
                    ui.input_numeric(id='growth_donor', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                    ui.input_slider(id='sus_donor', label='Susceptibility', value=0.05, min=0, max=1, step=0.01, width='100%')
                    ui.input_action_button(id='equal_growths_model', label='Set all growth rates to donor', width='100%')
                    @reactive.effect
                    @reactive.event(input.equal_growths_model)
                    def _():
                        ui.update_numeric(id='growth_recip', value=input.growth_donor())
                        ui.update_numeric(id='growth_trans', value=input.growth_donor())
                        for i in range(1,26):
                            exec(f"ui.update_numeric(id='growth_spec{i}', value=input.growth_donor())")  # noqa: S102
            # Recipient Parameters
            with ui.card():
                with ui.card_header():
                    ui.markdown("""###### Recipient Parameters""")
                with ui.layout_columns(col_widths=(6,6,12)):
                    ui.input_numeric(id='x0_recip', label='Starting Density', value=0.5, min=0.01, step=0.01, width='100%')
                    ui.input_numeric(id='growth_recip', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                    ui.input_slider(id='sus_recip', label='Susceptibility', value=0.05, min=0, max=1, step=0.01, width='100%')
            # Transconjugant Parameters
            with ui.card():
                with ui.card_header():
                    ui.markdown("""###### Transconjugant Parameters""")
                with ui.layout_columns(col_widths=(6,6,12)):
                    ui.input_numeric(id='x0_trans', label='Starting Density', value=0, min=0.01, step=0.01, width='100%')
                    ui.input_numeric(id='growth_trans', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                    ui.input_slider(id='sus_trans', label='Susceptibility', value=0.05, min=0, max=1, step=0.01, width='100%')
            # General Parameters
            with ui.card(fill=False):
                with ui.card_header():
                    ui.markdown(""" General Parameters""")
                with ui.layout_columns(col_widths=(4,4,4, 6,6)):
                    ui.input_checkbox(id='run_equilibrium', label='Run Equilibrium Step?', value=True, width='100%')
                    ui.input_numeric(id='tend', label='Simulation End Time', value=2000, min=0, step=1, width='100%')
                    ui.input_numeric(id='n_species', label='Number of Species in the Community:', value=0, min=0, max=26, step=1,width='100%')
                    # Set to simple
                    ui.input_action_button(id='set_to_simple', label='Set parameters to a simple one-species community', width='100%')
                    @reactive.effect
                    @reactive.event(input.set_to_simple)
                    def _():
                        # Interaction matrix
                        ui.update_checkbox(id='dr_equal', value=True)
                        ui.update_slider(id='connectivity', value=1)
                        ui.update_slider(id='Pm', value=0)
                        ui.update_slider(id='s_i', value=0.1)
                        ui.update_slider(id='sigma', value=0.2)
                        # Drugs
                        ui.update_slider(id='tdrug', value=0)
                        ui.update_slider(id='tdrug_recovery', value=2000)
                        ui.update_slider(id='drug_strength', value=0.5)
                        # Conjugation
                        ui.update_slider(id='tdr', value=0)
                        ui.update_numeric(id='simple_conj', value=5e-4)
                        ui.update_slider(id='spatial_conj', value=0)
                        # Donor/Recip/Trans
                        for sp, x in zip(['donor', 'recip', 'trans'], [0.5,0.5,0]):
                            exec(f"ui.update_numeric(id='x0_{sp}', value={x})")  # noqa: S102
                            exec(f"ui.update_numeric(id='growth_{sp}', value=1)")  # noqa: S102
                            exec(f"ui.update_slider(id='sus_{sp}',  value=0.05)")  # noqa: S102
                        # Other parameters
                        ui.update_checkbox(id='run_equilibrium', value=True)
                        ui.update_numeric(id='tend', value=2000)
                        ui.update_numeric(id='n_species', value=1)
                        ui.update_numeric(id='x0_spec1', value=0.5)
                        ui.update_numeric(id='growth_spec1', value=1)
                        ui.update_slider(id='sus_spec1',  value=0.2)
                    ui.input_action_button(id='set_to_hungate', label='Set parameters to recreate in vitro experiments with a simple oligo-mouse-microbiota-19 community', width='100%')
                    @reactive.effect
                    @reactive.event(input.set_to_hungate)
                    def _():
                        # Interaction matrix
                        ui.update_checkbox(id='dr_equal', value=True)
                        ui.update_slider(id='connectivity', value=0.9)
                        ui.update_slider(id='Pm', value=0.1)
                        ui.update_slider(id='s_i', value=0.1)
                        ui.update_slider(id='sigma', value=0.075)
                        # Drugs
                        ui.update_slider(id='tdrug', value=0)
                        ui.update_slider(id='tdrug_recovery', value= int(500 + (5/24*500)))
                        ui.update_slider(id='drug_strength', value=0.25)
                        # Conjugation
                        ui.update_slider(id='tdr', value=500)
                        ui.update_numeric(id='simple_conj', value=5e-3)
                        ui.update_slider(id='spatial_conj', value=0.2)
                        # Donor/Recip/Trans
                        for sp, x, s in zip(['donor', 'recip', 'trans'], [0.5,0.5,0], [0.05, 0.08, 0.05]):
                            exec(f"ui.update_numeric(id='x0_{sp}', value={x})")  # noqa: S102
                            exec(f"ui.update_numeric(id='growth_{sp}', value=1)")  # noqa: S102
                            exec(f"ui.update_slider(id='sus_{sp}',  value={s})")  # noqa: S102
                        # Other parameters
                        ui.update_checkbox(id='run_equilibrium', value=False)
                        ui.update_numeric(id='tend', value=2000)
                        ui.update_numeric(id='n_species', value=19)
                        for i in range(1,20):
                            exec(f"ui.update_numeric(id='x0_spec{i}', value=(1 + 0.5 * rd.random()))")  # noqa: S102
                            exec(f"ui.update_numeric(id='growth_spec{i}', value=1)")  # noqa: S102
                            exec(f"ui.update_slider(id='sus_spec{i}',  value=(0.2 + 0.2 * rd.random()))")  # noqa: S102

            # All the species
            with ui.panel_conditional("input.n_species >= 1"):  # noqa: SIM117
                with ui.accordion(id='model_all_species', open=[]):
                    with ui.accordion_panel('Parameters for Each Species in Model'):
                        with ui.layout_columns(col_widths=(4,4,4)):
                            with ui.card():
                                with ui.card_header():
                                    ui.markdown("""###### Species A Parameters""")
                                with ui.layout_columns(col_widths=(4,4,4)):
                                    ui.input_numeric(id='x0_spec1', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                    ui.input_numeric(id='growth_spec1', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                    ui.input_slider(id='sus_spec1', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 2"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species B Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec2', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec2', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec2', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 3"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species C Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec3', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec3', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec3', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 4"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species D Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec4', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec4', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec4', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 5"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species E Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec5', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec5', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec5', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 6"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species F Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec6', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec6', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec6', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 7"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species G Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec7', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec7', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec7', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 8"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species H Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec8', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec8', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec8', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 9"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species I Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec9', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec9', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec9', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 10"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species J Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec10', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec10', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec10', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 11"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species K Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec11', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec11', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec11', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 12"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species L Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec12', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec12', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec12', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 13"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species M Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec13', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec13', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec13', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 14"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species N Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec14', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec14', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec14', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 15"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species O Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec15', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec15', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec15', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 16"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species P Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec16', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec16', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec16', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 17"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species Q Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec17', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec17', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec17', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 18"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species R Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec18', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec18', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec18', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 19"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species S Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec19', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec19', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec19', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 20"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species T Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec20', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec20', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec20', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 21"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species U Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec21', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec21', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec21', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 22"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species V Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec22', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec22', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec22', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 23"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species W Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec23', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec23', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec23', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 24"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species X Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec24', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec24', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec24', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 25"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species Y Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec25', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec25', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec25', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')
                            with ui.panel_conditional("input.n_species >= 26"):  # noqa: SIM117
                                with ui.card():
                                    with ui.card_header(): ui.markdown("""###### Species Z Parameters""")
                                    with ui.layout_columns(col_widths=(4,4,4)):
                                        ui.input_numeric(id='x0_spec26', label='Starting Density', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_numeric(id='growth_spec26', label='Instrinsic Growth', value=1, min=0.01, step=0.01, width='100%')
                                        ui.input_slider(id='sus_spec26', label='Susceptibility', value=0.2, min=0, max=1, step=0.01, width='100%')

            # Transconjugant final numbers
            with ui.value_box(showcase=icon('bacteria'), theme="purple"):
                "Transconjugant Population Density"
                @render.ui
                def n_trans_model():
                    return f"{model_results.get()['y'][2][-1]:.4f}"
            @render.download(label="Download the population density results", filename='pop_density.csv')
            def results_files():
                results = model_results.get().y.T
                comm_species = results.shape[1] - 3
                yield pd.DataFrame(results, columns=['donor', 'recipient', 'transconjugant'] + [f'species_{ascii_uppercase[i]}' for i in range(comm_species)]).to_csv()

            # Run model
            @reactive.effect
            def _():
                # Set up parameters
                global model, model_alpha_init, model_r_init, x0, growth, susceptibility
                x0 = [input.x0_donor(), input.x0_recip(), input.x0_trans()]
                growth = [input.growth_donor(), input.growth_recip(), input.growth_trans()]
                susceptibility = [input.sus_donor(), input.sus_recip(), input.sus_trans()]
                for i in range(1, input.n_species() + 1):
                    exec(f"x0.append(input.x0_spec{i}())")  # noqa: S102
                    exec(f"growth.append(input.growth_spec{i}())")  # noqa: S102
                    exec(f"susceptibility.append(input.sus_spec{i}())")  # noqa: S102
                # Run model
                model = ode_lotka(input.n_species())  # initialise model
                model.randomise_interaction_matrix(connectivity=input.connectivity(), sigma=input.sigma(), P_m=input.Pm(), self_inhibition=input.s_i(), intrinsic_growth=growth, donor_equal_recipient=input.dr_equal())
                model_alpha_checks.set(model.good_alpha)
                if model_alpha_checks.get() == 1:
                    model_alpha.set( pd.DataFrame(model.alpha, columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())]) )
                    model_alpha_init = pd.DataFrame(model.alpha, columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())])
                    model_r.set( pd.DataFrame(model.r.reshape(1,len(model.r)),
                                            columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())]) )
                    model_r_init = pd.DataFrame(model.r.reshape(1,len(model.r)),
                                            columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())])
                    model.solve_ode(t_start=0, t_end=input.tend(), x0=x0, conjugation_rate=input.simple_conj(), susceptibility=susceptibility, drug_level=input.drug_strength(),
                                    equilibrium=input.run_equilibrium(), plasmid_method='proportion', prop_strength=input.spatial_conj(),
                                    t_dr=input.tdr(), t_drug=input.tdrug(), t_drug_recovery=input.tdrug_recovery())  # solve ode
                    model_results.set(model.results)
            # Generate new interaction matrix for model
            @reactive.effect
            @reactive.event(input.species_new_alpha)
            def _():
                # Run model
                model.randomise_interaction_matrix(connectivity=input.connectivity(), sigma=input.sigma(), P_m=input.Pm(), self_inhibition=input.s_i(), intrinsic_growth=growth, donor_equal_recipient=input.dr_equal())
                model_alpha_checks.set(model.good_alpha)
                if model_alpha_checks.get() == 1:
                    model_alpha.set( pd.DataFrame(model.alpha, columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())]) )
                    model_r.set( pd.DataFrame(model.r.reshape(1,len(model.r)),
                                            columns=['donor', 'recipient', 'transconjugants'] + [f'species_{ascii_uppercase[i]}' for i in range(input.n_species())]) )
                    model.solve_ode(t_start=0, t_end=input.tend(), x0=x0, conjugation_rate=input.simple_conj(), susceptibility=susceptibility, drug_level=input.drug_strength(),
                                    equilibrium=input.run_equilibrium(), plasmid_method='proportion', prop_strength=input.spatial_conj(),
                                    t_dr=input.tdr(), t_drug=input.tdrug(), t_drug_recovery=input.tdrug_recovery())  # solve ode
                    model_results.set(model.results)
            # Plot
            @render_plotly(width='100%')
            def plot_model():
                return plot_species_densities(model_results.get(), input.n_species())




    # DARK MODE
    ui.nav_spacer()
    with ui.nav_control():
        ui.input_dark_mode(mode="light")


# --- FUNCTION TO MAKE THE PLOT ---
def plot_species_densities(species_df, n_species, species_to_plot=None):
    """Makes the plot that shows the bacterial densities over time
    Parameters:
        species_df: the species dataframe with all densities over time
        n_species: the total number of species to plot
        species_to_plot: list of the species to include; None = all
    """
    # List all labels to print
    species_dict = {}
    all_labels = ['Donor', 'Recipient', 'Transconjugants'] + [f'Species {ascii_uppercase[i]}' for i in range(n_species)]
    for s in range(len(all_labels)):
        species_dict.update({all_labels[s]: species_df.y[s]})
    labels = species_to_plot if species_to_plot is not None else all_labels
    # Render figure layouts
    fig = go.Figure()
    fig.update_layout(plot_bgcolor='#eff5fa')
    colours = px.colors.sequential.Rainbow
    # Render plot
    for i, species in enumerate(labels):
        if i == 0: colour = 'black'
        elif i == 1: colour = 'sienna'
        elif i == 2: colour = 'darkviolet'
        else: colour = colours[int(i / len(labels) * len(colours))]
        fig.add_trace(go.Scatter(x=species_df.t,
                                 y=species_dict[species],
                                 mode='lines',
                                 name=species,
                                 legendgroup=species,
                                 line={'color': colour}))
    fig.update_layout(font={'size': 14},
                      height=550,
                      title=f'<b>Transconjugant Growth with {n_species} Community Species<b>',
                      title_font={'size': 20},
                      xaxis_title='Time',
                      xaxis={'tickfont': {'size': 18}},
                      hovermode="x",
                      yaxis_title='Species Population Density',
                      yaxis={'tickfont': {'size':18}},
                      legend={"title": 'Species', "groupclick": 'togglegroup'})
    return fig