# THE CLASS THAT RUNS THE ODE MODEL
# Author: An Mei Daniels

# Import useful modules
import numpy as np  # noqa: I001
import numpy.random as rd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cycler import cycler


# Class function
class ode_lotka:
    """A class that allows the modelling the growth and conjugation rates of cells with antibiotic resistance genes (ARG) in a community."""

    def __init__(self, n_species):
        """Initialise the class, with the number of species you want to have in the community.
        
        Parameters
        -----------
        n_species : integer
            the number of species in the community (excluding the donor, recipient and transconjugant populations)
        good_alpha : boolean
            indicates if the interaction matrix array is set up nicely
        alpha : np.array
            an n_species x n_species matrix of interaction coefficients, with the -self_inhibition as the diagonal matrix
        r : np.array
            intrinsic growth rates (or death rates when negative) measuring the growth/decline of population i when growing alone at low density"""
        self.n_species = 3 + n_species  # donor, recipient and transconjugant + chosen number of species in community
        self.good_alpha = 0
        self.alpha = np.zeros([n_species, n_species])
        self.r = np.zeros([n_species])
    
    def randomise_interaction_matrix(self, connectivity, sigma, P_m, self_inhibition, intrinsic_growth=1, donor_equal_recipient=True):
        """A function that randomly populates the interaction matrix, based on certain input parameters. 
        It will repeat until it creates a matrix that will have a dynamic equilibrium (is stable) or if it is looped
        over 50600 times, in which case it will prompt the entry of new interaction matrix parameters to try to create
        a more solveable matrix. A matrix is determined as solvable if the maximum eigenvalue of its Jacobian is less than 0.
        
        Parameters
        -----------
        connectivity : float
            proportion of populations that have interactions with each other (between 0-1)
        sigma : float
            the standard deviation of the normal distribution where the strength of interactions between populations is pulled
        P_m : float
            proportion of those interacting species that have positive interactions instead of negative
        self_inhibition : float
            intra-specific interaction strength (assume a negative interaction with self)
        intrinsic_growth : float or list of float, default=1
            the intrinsic growth rate we'd like to find at steady state for
            if the value is a float, apply it to all species
            if the value is a list, use each indexed value for each species
        donor_equal_recipient : boolean, default=True
            determines whether the donor and recipient values should be set as the same species (reducing variability) or different species"""
        self.good_alpha = 0  # reset alpha to allow new reassignment if necessary
        checks = 0
        while self.good_alpha == 0:  # repeat until the matrix is stable
            # Randomly seed the connectivity of different species
            C = rd.uniform(size=(self.n_species, self.n_species))  # randomly generate interaction matrix
            C[C > (1-connectivity)] = 1   # set values above connectivity to 1 (are interacting)
            C[C <= (1-connectivity)] = 0  # set values below connectivity to 0 (are not interacting)
            # Randomly assign species interaction strengths between 0 and -1 (assume negative interaction)
            interspecies_interaction = (sigma * abs(rd.uniform(size=(self.n_species, self.n_species)))) * -1  # set random species interactions
            alpha = np.multiply(C, interspecies_interaction)  # give only connected populations the species interactions
            # Randomly set a proportion of those with positive interactions
            alpha[np.identity(self.n_species) == 1] = 0  # set diagonals to 0 to ignore these values
            non_zero = np.where(alpha<0)  # find the interacting species in the contact matrix
            switch_to_p = rd.random(len(non_zero[0]))  # randomly assign values to the interacting species
            switch_to_p = np.where(switch_to_p < P_m)[0]  # find the proportion of those that have positive interactions
            find_switch_row = non_zero[0][switch_to_p].tolist()  # row-values to change
            find_switch_col = non_zero[1][switch_to_p].tolist()  # col-values to change
            alpha[find_switch_row, find_switch_col] = alpha[find_switch_row, find_switch_col] * -1  # make these ones positive interactions
            # Set self-inhibition for all self-self interactions
            np.fill_diagonal(alpha, -self_inhibition )  # fill diagonal matrix with self-inhibition value
            # If donors and recipients are the same species, set those to the same thing
            if donor_equal_recipient:
                alpha[0,1] = -self_inhibition  # recipients self-inhibit donors
                alpha[0,2] = -self_inhibition  # transconjugants are recipients and also self-inhibit
                alpha[1] = alpha[0]  # recipients = donors
            # Set the transconjugants to have the same interactions as the recipients (as the same bacteria)
            alpha[1,2] = -self_inhibition  # recipient interaction with transconjugants is the same as self-inhibition
            alpha[2] = alpha[1]  # transconjugants = recipients
            # Finding r from self-inhibition
            steady_state = self.get_growth_rate(alpha, intrinsic_growth, donor_equal_recipient)
            # Calculate stability
            stability = self.get_stability(alpha, steady_state)
            if stability < 0:  # if the values of eigenvalues less than 0 it is stable
                self.good_alpha = 1  # allow it to break loop
                self.alpha = alpha  # assign alpha value
            else: checks += 1
            if checks > 50600:
                print('Interaction matrix parameters are hard to solve. Try entering different parameters and run again.')
                self.good_alpha = -1
                break

    def enter_custom_interaction_matrix(self, alpha, self_inhibition, intrinsic_growth=1, donor_equal_recipient=True):
        """A function that allows custom entry of an interaction matrix, and self inhibition rates, and then will calculate the growth rate and ensure it is stable.
        
        Parameters
        -----------
        alpha : np.array
            the interaction matrix we'd like to use for the equations
        self_inhibition : float
            intra-specific interaction strength (assume a negative interaction with self)
        intrinsic_growth : float or list of float, default=1
            the intrinsic growth rate we'd like to find at steady state for
            if the value is a float, apply it to all species
            if the value is a list, use each indexed value for each species
        donor_equal_recipient : boolean, default=True
            determines whether the donor and recipient values should be set as the same species (reducing variability) or different species"""
        self.good_alpha = 0  # reset alpha to allow new reassignment if necessary
        # Check the matrix is the correct size
        if alpha.shape != (self.n_species, self.n_species):
            print(f'Your current interaction matrix size is: {alpha.shape}.')
            print(f'Please make the interaction matrix a {self.n_species} x {self.n_species} array, to be equal to the number of species.')
            return
        # Overwrite self-inhibition from inputted value to update diagonal values of interaction matrix just in case
        np.fill_diagonal(alpha, -self_inhibition)  # fill diagonal matrix with negative self-inhibition value
        # If donors=recipients, overwrite the recipients to have the same interactions as the donors (as the same bacteria) just in case
        if donor_equal_recipient:
            alpha[0,1] = -self_inhibition  # recipients self-inhibit donors
            alpha[0,2] = -self_inhibition  # transconjugants are recipients and also self-inhibit
            alpha[1] = alpha[0]  # recipients = donors
        # Overwrite the transconjugants to have the same interactions as the recipients (as the same bacteria) just in case
        alpha[1,2] = -self_inhibition  # recipient interaction with transconjugants is the same as self-inhibition
        alpha[2] = alpha[1]  # transconjugants = recipientsn
        # Finding r from self-inhibition
        steady_state = self.get_growth_rate(alpha, intrinsic_growth, donor_equal_recipient)
        # Calculate stability of inputted matrix
        stability = self.get_stability(alpha, steady_state)
        if stability < 0:  # if the values of eigenvalues less than 0 it is stable
            self.good_alpha = 1  # allow it to break loop
            self.alpha = alpha  # assign alpha value
        else:
            print('This interaction matrix does not lead to steady state equilibrium - please try again with a new value.')
    
    def set_interaction_matrix(self, alpha, intrinsic_growth):
        """A function that allows you to set the interaction matrix, self inhibition rates, and growth rates without altering or checking it is stable.
        This is the most simplified version of setting it - use with caution as it may not always work.
        Use for reproducibility if you've created a set you know works and want to reuse it across tests.
        
        Parameters
        -----------
        alpha : np.array
            the interaction matrix we'd like to use for the equations
        intrinsic_growth : float or list of float, default=1
            the intrinsic growth rate we'd like to find at steady state for
            if the value is a float, apply it to all species
            if the value is a list, use each indexed value for each species"""
        self.alpha = alpha  # initialise interaction matrix
        self.r = intrinsic_growth  # set growth rate
        self.good_alpha = 1  # forces acceptance of manual values

    def upload_interaction_matrix_r(self, filepath):
        """A function that uses the interaction matrix and r from a file.
        
        Parameters
        -----------
        filepath : string
            the filepath the interaction matrix and r should be read from - they will have the suffix ``*_alpha.csv`` or ``*_r.csv`` respectively"""
        self.alpha = np.loadtxt(f'{filepath}_alpha.csv', delimiter=',')  # initialise interaction matrix
        self.r = np.loadtxt(f'{filepath}_r.csv', delimiter=',')  # set growth rate
        self.good_alpha = 1  # accept manual values

    def save_interaction_matrix_r(self, filepath):
        """A function that saves the interaction matrix and r to a file.
        
        Parameters
        ----------
        filepath : str
            the filepath the interaction matrix and r should be saved to
            they will have the suffix *_alpha.csv or *_r.csv respectively"""
        np.savetxt(f'{filepath}_alpha.csv', self.alpha, delimiter=',')
        np.savetxt(f'{filepath}_r.csv', self.r, delimiter=',')

    def get_growth_rate(self, alpha, intrinsic_growth):
        """Calculates the intrinsic growth rate based on self-inhibition rate (taken from the full interaction matrix to maximise the odds of a stable equilibrium).
        r is calculated as the matrix multiplication of ``alpha`` and the steady state.
        The `steady_state` is taken as ~ ``intrinsic_growth`` rate, as we assume the community has a linearly asymptotically stable equilibrium at ``x_1`` ~ ``intrinsic_growth``
       
        Parameters
        -----------
        alpha : np.array
            the interaction matrix we'd like to use for the equations
        intrinsic_growth : float or list of float, default=1
            the intrinsic growth rate we'd like to find at steady state for
            if the value is a float, apply it to all species
            if the value is a list, use each indexed value for each species
        
        Returns
        --------
        steady_state : np.array
            the steady_state value used to calculate the growth rates - the adjusted inputted intrinsic_growth"""
        if type(intrinsic_growth) == float or type(intrinsic_growth) == int:
            steady_state = abs(np.ones(self.n_species) * intrinsic_growth + 0.005*rd.random(self.n_species))  # taking the steady state growth condition and adding some random noise
        else: 
            steady_state = abs(np.array(intrinsic_growth) + 0.005*rd.random(self.n_species))  # taking the steady state growth condition and adding some random noise
        # Legacy - recipient and transconjugant (and potentially donor) have same interaction matrix params - edit growth rates to prevent double/triple the instrinsic growth provided
        # divide = 2  # default is growth rates becoming double what user entered
        # if donor_equal_recipient:  # if recipient also equals donor
        #     divide = 3  # growth rates would end up triple what user entered
        # steady_state[0] = steady_state[0]/divide  # reduce growth rate as donor and recipient are the same (to give growth rate entered)
        # steady_state[1] = steady_state[1]/divide  # reduce growth rate as recipient and transconjugant are the same (to give growth rate entered)
        # steady_state[2] = steady_state[2]/divide  # reduce growth rate as recipient and transconjugant are the same (to give growth rate entered)
        self.r = -np.matmul(alpha, steady_state)  # calculate the r and saves it
        return steady_state

    def get_stability(self, alpha, steady_state):
        """Calculates the stability from the Jacobian of the alpha and the self-inhibition rate, based on eigenvalues.
        
        Parameters
        -----------
        alpha : np.array
            the interaction matrix we'd like to use for the equations
        steady_state : np.array
            the steady_state value used to calculate the growth rates - the adjusted inputted intrinsic_growth
        
        Returns
        --------
        stability : float
            the maximum eigenvalue of the Jacobian of the interaction matrix
            if this is less than 0, then the matrix is likely to be stable"""
        jacobian = np.multiply(alpha,  np.tile(steady_state.T, [self.n_species,1]))  # calculates the Jacobian
        stability = max(np.linalg.eig(jacobian).eigenvalues.real)  # gets the eigenvalues to determine stability
        return stability

    def display_matrix_used(self):
        """Displays the matrix and growth rate used to run the model. Useful after random initialisation."""
        print(f'The interaction matrix used for this run is:\n{self.alpha}\n')
        print(f'The growth rate is:\n{self.r}\n')

    def lotka_drug_plasmid(self, t, X, conjugation_rate, susceptibility, drug_level, plasmid_method='simple', prop_strength=1):
        """The generalised Lotka-Volterra ODE equation to be solved, with terms added for plasmid transfer and drug perturbation.
        
        Parameters
        -----------
        t : integer
            timestep, the current timestep being solved by the ode integrator
        X : np.array
            an array of the population density at the given timestep for each species in the community
            - X[0] - the donor
            - X[1] - the recipient
            - X[2] - the transconjugant - generally 0 at start of experiment
            - X[3:] - the rest of the species communities
        conjugation_rate : float
            plasmid conjugation rate
        susceptibility : float
            susceptibility of the communities to the drug perturbation, from 0-1)
        drug_level : np.array
            the level of strength of the drug perturbation over time, the values in the array from 0-1
        plasmid_method : string, default='simple'
            method that plasmids are transferred
            - simple: only considers the number of potential recipients and donors
            - proportion: takes into account recipients and donors in total population and how other species might block contact
        prop_strength : float, default=1
            if using the 'proportion' plasmid_method, the strength of the effect of spatial inhibition, from 0-1
        
        Returns
        --------
        change_in_density : np.array
            an array of the the change in population density from initial population density X at between timestep t-1 and t"""
        dXdt = X * (self.r + np.matmul(self.alpha, X)) # calculate growth rates
        # Calculate plasmid transfer
        if plasmid_method == 'equilibrium':
            plasmid_transfer = 0
        if plasmid_method == 'simple':
            plasmid_transfer = self.simple_plasmid(X, conjugation_rate)
        elif plasmid_method == 'proportion':
            plasmid_transfer = self.prop_plasmids(X, conjugation_rate, prop_strength)
        # Calculate susceptibility to drug, based on time-dependent drug effect
        drug_effect = np.array(susceptibility) * drug_level[int(np.round(t))]
        # Define specific derivatives accounting for plasmids and return
        dDdt = dXdt[0] - drug_effect[0] * X[0]  # donor growth only affected by drugs
        dRdt = dXdt[1] - drug_effect[1] * X[1] - plasmid_transfer  # recipient affected by losing population to transconjugant pop
        dTdt = dXdt[2] - drug_effect[2] * X[2] + plasmid_transfer  # transconjugant gains population from recipients
        dCdt = dXdt[3:] - drug_effect[3:] * X[3:]  # community also only affected by drugs
        return np.concat([[dDdt], [dRdt], [dTdt], dCdt], axis=0)

    @staticmethod
    def simple_plasmid(X, conjugation_rate):
        """The simplest form of conjugation rates, just considering:
        - number of transconjugants based on number of recipients that can become transconjugants
        - number of donors that they can recieve the plasmid from
        
        Parameters
        -----------
        X : np.array
            an array of the population density at the given timestep for each species in the community
        conjugation_rate : float
            plasmid conjugation rate"""
        return X[1] * (conjugation_rate*(X[0] + X[2]))  # calculates the transfers based on number of recipients and how many they can receive from (donors + transconjugants)
    
    @staticmethod
    def prop_plasmids(X, conjugation_rate, prop_strength=1):
        """The next most complicated form of conjugation rates, considering:
        - number of transconjugants based on number of recipients that can become transconjugants
        - number of donors that they can recieve the plasmid from
        - the proportion of these compared to the rest of the population, that might block them from physically coming into contact
        
        Parameters
        -----------
        X : np.array
            an array of the population density at the given timestep for each species in the community
        conjugation_rate : float
            plasmid conjugation rate
        prop_strength : float, default=1
            if using the 'proportion' plasmid_method, the strength of the effect of spatial inhibition, from 0-1"""
        plasmid_transfer = X[1] * (conjugation_rate*(X[0] + X[2]))  # calculates the transfers based on number of recipients and how many they can receive from (donors + transconjugants)
        prop_donor = (X[0] + X[2])/np.sum(X)  # taking into account proportion of donors
        prop_recipient = X[1] / np.sum(X)  # and recipients
        prop_strength = ((prop_strength - 0) / (1 - 0)) * (1 - (prop_donor * prop_recipient)) + (prop_donor * prop_recipient) # rescale from 0-1 to proportion-1 (avoid div by 0 errors)
        return plasmid_transfer * prop_donor * prop_recipient * (1/prop_strength)  # on likelihood of the plasmid being transfered
    
    def solve_ode(self, t_start, t_end, x0, conjugation_rate, susceptibility, drug_level, equilibrium=True, plasmid_method='simple', prop_strength=1, t_dr=0, t_drug=0, t_drug_recovery=None):
        """The ode solver to get the results of the simulation. First checks the lists are the correct length, then allows the ode solver to run.
        
        Parameters
        -----------
        t_start : integer
            the starting timestep
        t_end : integer
            the ending timestep
        x0 : np.array
            an array of the initial population densities of each species in the community
            - X[0] - the donor
            - X[1] - the recipient
            - X[2] - the transconjugant - generally 0 at start of experiment
            - X[3:] - the rest of the species communities
        conjugation_rate : float
            plasmid conjugation rate
        susceptibility : float
            susceptibility of the communities to the drug perturbation, from 0-1)
        drug_level : np.array
            the level of strength of the drug perturbation over time, the values in the array from 0-1
        plasmid_method : string, default='simple'
            method that plasmids are transferred
            - simple: only considers the number of potential recipients and donors
            - proportion: takes into account recipients and donors in total population and how other species might block contact
        prop_strength : float, default=1
            if using the 'proportion' plasmid_method, the strength of the effect of spatial inhibition, from 0-1
        t_dr : integer, default=0
            the time the donor and the recipient are introduced into the community, typically the start of the simulation
        t_drug : integer, default=0
            the time the drug should be introduced, typically the start of the simulation
        t_drug_recovery : integer, default=None
            the time the drug should be removed from the simulation, typically it is still acting on the community until the end of the simulation"""
        # Do not run simulation if there are problems with lengths of input lists - make errors clearer
        if (len(susceptibility) != self.n_species) & (len(x0) != self.n_species):
            print(f'Your susceptibility list is {len(susceptibility)} long, and your starting growth rate is {len(x0)}.')
            print(f'Please make the length of the susceptibility list and the starting growth rates {self.n_species} to be equal to the number of species.')
        elif (len(susceptibility) != self.n_species):
            print(f'Your susceptibility list is {len(susceptibility)} long.')
            print(f'Please make the length of the susceptibility list {self.n_species}, to be equal to the number of species.')
        elif (len(x0) != self.n_species):
            print(f'Your starting growth rate list is {len(x0)} long.')
            print(f'Please make the length of the starting growth rates {self.n_species}, to be equal to the number of species.')
        elif self.good_alpha != 1: 
            print('You have no valid interaction matrix to run this simulation. Please fix your interaction matrix before trying again.')
        # If interaction matrix works and no other errors, run ODEs to seem community 
        elif self.good_alpha == 1:
            # Determine drug strengths over time
            drug_time = np.zeros(t_end+1)  # initialise the drug strength over time
            drug_time[t_drug:t_drug_recovery] = drug_level  # add the strengths to the times, from introduction to removal
            # Run the model with equilibrium
            if equilibrium:
                self.equilibrium = solve_ivp(self.lotka_drug_plasmid,  # equilibriation step without any pathogens or perturbation from drugs
                                             [0, 1500], 
                                             [0, 0, 0] + x0[3:], 
                                             first_step=1, max_step=1, 
                                             args=[conjugation_rate, np.zeros(self.n_species), np.zeros(1501), 'equilibrium'])
                 # If donor and recipient are introduced from the beginning
                if t_dr == 0: 
                    self.results = solve_ivp(self.lotka_drug_plasmid,  # introduce pathogens + equilibriated community to perturbation with drugs and solve ode for growth rate
                                            [t_start, t_end], 
                                            x0[0:3] + self.equilibrium.y[3:,-1].tolist(), 
                                            first_step=1, max_step=1, 
                                            args=[conjugation_rate, susceptibility, drug_time, plasmid_method, prop_strength])
                # If donor and recipient are introduced at different times
                else:
                    self.results = solve_ivp(self.lotka_drug_plasmid,  # introduce equilibriated community to perturbation with drugs and solve ode for growth rate
                                             [t_start, t_dr], 
                                             [0,0,0] + self.equilibrium.y[3:,-1].tolist(), 
                                             first_step=1, max_step=1, 
                                             args=[conjugation_rate, susceptibility, drug_time, plasmid_method, prop_strength])
                    results = solve_ivp(self.lotka_drug_plasmid,  # introduce community with drugs to the DR and solve ode for growth rate
                                        [t_dr, t_end], 
                                        x0[0:3] + self.results.y[3:,-1].tolist(), 
                                        max_step=1, 
                                        args=[conjugation_rate, susceptibility, drug_time, plasmid_method, prop_strength])
                    # Add together the results of the before and after
                    self.results.nfev = self.results.nfev + results.nfev
                    self.results.t = np.concat([self.results.t, results.t[1:]], axis=0)
                    self.results.y = np.concat([self.results.y, results.y[:,1:]], axis=1)
            # Run the model without an equilibriation step
            else:
                 # If donor and recipient are introduced from the beginning
                if t_dr == 0: 
                    self.results = solve_ivp(self.lotka_drug_plasmid,  # introduce pathogens + community to perturbation with drugs and solve ode for growth rate
                                             [t_start, t_end], 
                                             x0, 
                                             first_step=1, max_step=1, 
                                             args=[conjugation_rate, susceptibility, drug_time, plasmid_method, prop_strength])
                # If donor and recipient are introduced at different times
                else:
                    self.results = solve_ivp(self.lotka_drug_plasmid,  # introduce equilibriated community to perturbation with drugs and solve ode for growth rate
                                             [t_start, t_dr], 
                                             [0,0,0] + x0[3:], 
                                             first_step=1, max_step=1, 
                                             args=[conjugation_rate, susceptibility, drug_time, plasmid_method, prop_strength])
                    results = solve_ivp(self.lotka_drug_plasmid,  # introduce community with drugs to the DR and solve ode for growth rate
                                        [t_dr, t_end], 
                                        x0[0:3] + self.results.y[3:,-1].tolist(), 
                                        max_step=1, 
                                        args=[conjugation_rate, susceptibility, drug_time, plasmid_method, prop_strength])
                    # Add together the results of the before and after
                    self.results.nfev = self.results.nfev + results.nfev
                    self.results.t = np.concat([self.results.t, results.t[1:]], axis=0)
                    self.results.y = np.concat([self.results.y, results.y[:,1:]], axis=1)
        
    def plot_results(self, t, y, title, figsize=(7,6), save_fig=False, filename='equilibrium', drt_colours=['magenta', 'b', 'black'], cmap='Spectral', max_colours=26):
        """Plots the graph of the community finding equilibrium, and potentially saves the figure.
        
        Parameters
        -----------
        t : np.array
            the array of timepoints
        y : np.array
            the array of the population sizes
        title : string
            the title to use for the graph
        figsize : tuple, default=(7,6)
            the size of the figure to be outputted
        save_fig : boolean, default=False
            determines if the figure should be saved to files
        filename : string, default='equilibrium'
            the filename if the figure is saved
        drt_colours : list, default=['magenta', 'b', 'black']
            the colours to use when plotting the donor, recipient, and transconjugants (should be distinct from the rest of the colour cycle)
        cmap : colourmap, default='Spectral'
            the continuous colourmap to use when plotting the species in the community in matplotlib
        max_colours : integer, default=26
            the maximum number of discrete colours to divide the cmap into"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_prop_cycle(self.set_colours(drt_colours, cmap, max_colours))
        plt.plot(t, y, label = ['Donor', 'Recipient', 'Transconjugants'] + [f'Species {i+1}' for i in range(self.n_species-3)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Population Density')
        plt.legend()
        plt.title(title, fontweight='bold')
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{filename}.png', dpi=500)
        plt.show()

    def next_color(self, i, cmap, max_colours):
        """Gets distinct colours from a colourmap.
        
        Parameters
        -----------
        i : integer
            the number of species that each need a distinct colour
        cmap : colourmap
            the continuous colourmap to use when plotting the species in the community in matplotlib
        max_colours : integer
            the maximum colours to divide the cmap into
        
        Returns
        --------
        cmap : colourmap
            the list of colours to use, with an entry for each species"""
        return cmap(((5 * i) % max_colours) / max_colours)

    def set_colours(self, drt_colours, cmap, max_colours):
        """Sets the colours for the graphs.
        
        Parameters
        -----------
        drt_colours : list, default=['magenta', 'b', 'black']
            the colours to use when plotting the donor, recipient, and transconjugants (should be distinct from the rest of the colour cycle)
        cmap : colourmap, default='Spectral'
            the continuous colourmap to use for the species in the community in matplotlib
        max_colours : integer, default=26
            the maximum number of discrete colours to divide the cmap into
        
        Returns
        --------
        cycler : cycler
            a list of all the colours to use for each of the species when plotting"""
        cmap = plt.get_cmap(cmap, lut=max_colours)
        colours = [self.next_color(i, cmap, max_colours) for i in range(self.n_species-3)]
        colours = [mcolors.to_hex(c) for c in colours]
        return cycler(color = drt_colours + colours)
    
    def determine_species_growth_rates(self, conjugation_rate, plasmid_method='simple', prop_strength=1):
        """Determines the difference between clonal expansion and actual conjugation within the transconjugant population. This method recalculates the ``plasmid_transfer`` term using the relevant method, and compares to the growth rates of species.
       
        Parameters
        ------------
        conjugation_rate : float
            plasmid conjugation rate for determining just plasmid transfer
        plasmid_method : string, default='simple'
            method that plasmids are transferred
            - simple: only considers the number of potential recipients and donors
            - proportion: takes into account recipients and donors in total population and how other species might block contact
        prop_strength : float, default=1
            if using the 'proportion' plasmid_method, the strength of the effect of spatial inhibition, from 0-1
        
        Returns
        --------
        time : np.array
            all the timesteps used in the simulation
        donor : np.array
            the change in donor population density at each timestep (gain or loss in the population)
        recipient : np.array
            the change in the recipient population density at each timestep (gain or loss in the population)
        trans_clonal : np.array
            the change in transconjugant population density from just 'clonal' growth at each timestep (gain or loss in the population)
        plasmid : np.array
            the population density of transconjugants who recieved the plasmid at each time step"""
        time = self.results.t
        # Donor
        donor = np.diff(self.results.y[0,:].T)
        donor = np.insert(donor, 0, 0)
        # Plasmid
        if plasmid_method == 'simple':
            plasmid = np.array([self.simple_plasmid(self.results.y[:,i], conjugation_rate) for i in range(len(time))][:-1])
        elif plasmid_method == 'proportion':
            plasmid = np.array([self.prop_plasmids(self.results.y[:,i], conjugation_rate, prop_strength) for i in range(len(time))][:-1])
        plasmid = np.insert(plasmid, 0, 0)
        # Recipient
        recipient = np.diff(self.results.y[1,:].T)
        recipient = np.insert(recipient, 0, 0)
        recipient = recipient + plasmid
        # Transconjugant
        trans_clonal = np.diff(self.results.y[2,:].T)
        trans_clonal = np.insert(trans_clonal, 0, 0)
        trans_clonal = trans_clonal - plasmid
        return time, donor, recipient, trans_clonal, plasmid

    def plot_growth_rates(self, conjugation_rate, plasmid_method='simple', prop_strength=1, show_transconjugant_growth=True, compare_dr=False, figsize=(7,6), save_fig=False, filename='growth_rates', tp_colours=['k', 'lime', 'magenta', 'b']):
        """Plots the graph of the transconjugant growth rate, vs how many are actually formed via conjugation (actual plasmid transfer term). May also include donor and recipient species.
        
        Parameters
        -----------
        conjugation_rate : float
            plasmid conjugation rate for determining just plasmid transfer
        plasmid_method : string, default='simple'
            method that plasmids are transferred
            - simple: only considers the number of potential recipients and donors
            - proportion: takes into account recipients and donors in total population and how other species might block contact
        prop_strength : float, default=1
            if using the 'proportion' plasmid_method, the strength of the effect of spatial inhibition, from 0-1
        show_transconjugant_growth : boolean, default=True
            whether to include the overal transconjugant population
        compare_dr : boolean, default=False
            whether the graph shows donor/recipient growth rates, or just transconjugant clonal growth and plasmid transfer rate
        save_fig : boolean, default=False
            determines if the figure should be saved to files
        filename : string, default='equilibrium'
            the filename if the figure is saved
        tp_colours : list
            the colours to use when plotting the transconjugants and the plasmid transfer"""
        time, donor, recipient, trans_clonal, plasmid = self.determine_species_growth_rates(conjugation_rate, plasmid_method, prop_strength)
        fig, ax = plt.subplots(figsize=figsize)
        if compare_dr:  
            ax.plot(time, donor, color=tp_colours[2], label='donor')
            ax.plot(time, recipient, color=tp_colours[3], label='recipient')
        if show_transconjugant_growth:
            secax = ax.twinx()
            secax.set_ylabel('Overall Transconjugant Population Density')
            if compare_dr:
                secax.plot(time, self.results.y[0], '--', color=tp_colours[2])
                secax.plot(time, self.results.y[1], '--', color=tp_colours[3])
            secax.plot(time, self.results.y[2], '--', color=tp_colours[0], label='transconjugant pop') 
            secax.legend(title='Population Density') 
        ax.plot(time, trans_clonal, color=tp_colours[0], label='transconjugant clonal')
        ax.plot(time, plasmid, color=tp_colours[1], label='plasmid transfer')
        ax.legend(title='Growth Rates')
        plt.title('Growth Rate Over Time', fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Growth Rate')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(f'{filename}.png', dpi=500)
        plt.show()