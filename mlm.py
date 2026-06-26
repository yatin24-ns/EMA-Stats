import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az

np.random.seed(42) #Random number generator of sorts 
#We siimulate the markov chains using this random seed for a total of 

n_subjects = 15
#prompts_per_day=3
prompts_per_subject = 60
#Total number of output data points simulated = 15*3*20= 900 data pts 

total_observations =n_subjects * prompts_per_subject
subject_ids = np.repeat(np.arange(1, n_subjects + 1), prompts_per_subject)
subject_ids_str = [f"Subject_{i}" for i in subject_ids] # Convert to strings for categorical clustering

# Generate a Level-1 Within-Person Predictor
momentary_stress = np.random.normal(loc=0, scale=1, size=total_observations)

#Define a novel function to make the priors for getting the bayesian estimate fo a specific cluster 



# Ground Truth Mathematical Parameters
grand_intercept = 4.5  # Overall population baseline anxiety
true_slope = 0.75      # For every 1 unit stress increase, anxiety increases by 0.75
sigma_u = 1.2          # True standard deviation between the 15 people's baselines
sigma_e = 0.6          # True Level-1 residual error 

# Generate Random Intercepts (u_i) for our 15 distinct people
random_intercepts = np.random.normal(loc=0, scale=sigma_u, size=n_subjects)
subject_effects = random_intercepts[subject_ids - 1]

residuals = np.random.normal(loc=0, scale=sigma_e, size=total_observations)

# Calculate Dependent Variable (Momentary Anxiety) using the composite equation
momentary_anxiety = grand_intercept + subject_effects + (true_slope * momentary_stress) + residuals

# Construct the final pandas DataFrame
df_ema = pd.DataFrame({
    "subject_id": subject_ids_str,
    "stress_pmc": momentary_stress,
    "anxiety": momentary_anxiety
})

print(f"Data Simulated Successfully. Matrix Shape: {df_ema.shape}")
print(df_ema.head(5))
print("-" * 60)

print ("We need to run this sim by regularising the priors and safeguarding the sample stuff")

# STEP 2: DEFINE REGULARIZING PRIORS TO SAFEGUARD THE N=15 SAMPLE
custom_priors = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=10),
    "stress_pmc": bmb.Prior("Normal", mu=0, sigma=5),
    "1|subject_id": bmb.Prior("HalfNormal", sigma=2.5), # Anchor for Level-2 SD
    "sigma": bmb.Prior("Exponential", lam=1)            # Anchor for Level-1 Residual SD
}
# STEP 3: BUILD AND FIT THE MODEL VIA PYMC / NUTS SAMPLER
print("Initializing Hamiltonian Monte Carlo (HMC/NUTS) Sampling...")

# Initialize the Bambi model structure
model = bmb.Model("anxiety ~ stress_pmc + (1 | subject_id)", data=df_ema)

# Run the Leapfrog integration algorithm across 4 independent chains
idata = model.fit(
    priors=custom_priors,
    draws=10000,      # Number of recorded posterior samples per chain
    tune=1000,       # Warmup/burn-in steps to map the log-posterior geometry
    chains=4,        # Independent paths to verify convergence
    random_seed=42,
    target_accept=0.95 # Increases step sensitivity to prevent divergent transitions
)

print("-" * 60)
print("Sampling Complete. Generating Mathematical Summaries...")
print("-" * 60)

#Posterior Inferences 
summary_stats = az.summary(idata, var_names=["Intercept", "stress_pmc", "1|subject_id_sigma", "sigma"])
print(summary_stats)


az.plot_trace(idata)