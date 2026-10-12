"""
The goal of this file is to fill NAs.
For the vast majority of features a NAs means actually zero. There are only three ecceptions:
1) Reddito Frabbricati: since the missing values and zeros of the Bolzano's territory are due to a different taxation.
2) Addetti: since the data is censured below 3%.
We are going to compute them using BYM2 and ADVI approximation.
"""

###############
### IMPORTS ###
###############
import pandas as pd
import numpy as np
import geopandas as gpd
import pymc
import arviz as az
import scipy.sparse as sp
import libpysal

######################
### FIND BEST CORR ###
######################

def corr_relation(data, variable, quantile = 0.8):

    data_numeric = data.select_dtypes(include=["float64", "int64"])

    totale_safe = data["TOTALE"].replace(0, np.nan)

    data_ratios = data_numeric.div(totale_safe, axis="index")

    corr_RATIOS = data_ratios.corr()

    var_corr = corr_RATIOS[variable]
    abs_threshold = var_corr.abs().quantile(quantile)
    top_RATIO_features = var_corr[var_corr.abs() >= abs_threshold]
    with pd.option_context('display.max_rows', None):
        top_features = top_RATIO_features.sort_values(ascending = False)
        #print(top_features)

    return top_features



##############
### MODELS ###
##############



def Bayesian_Model(Y_all, X_all, Offset_all, w, train_idx, predict_idx, distribution="Normal", int_prior = 0):

    # Scale using the full dataset
    X_mean, X_std = np.mean(X_all, axis=0), np.std(X_all, axis=0)
    X_scaled = (X_all - X_mean) / (X_std + 1e-8)

    Offset_safe = np.where(Offset_all == 0, 1e-8, Offset_all)
    log_offset = np.log(Offset_safe)

    N = len(Y_all)
    k = X_scaled.shape[1]

    # Convert sparse matrix to dense for PyMC
    W_dense = w.sparse.toarray()

    W_dense = np.maximum(W_dense, W_dense.T)

    with pymc.Model() as model:

        # 1. Setup Data Containers
        x_shared = pymc.Data("x", X_scaled)
        offset_shared = pymc.Data("offset", log_offset)

        # 2. Linear Predictor
        Intercept = pymc.Normal("Intercept", mu = int_prior, sigma=1) 
        beta = pymc.Normal("beta", mu=0, sigma=1, shape=k)

    # Spatial predictor
        v = pymc.Normal("v", mu=0, sigma=1, shape=N) 
        u = pymc.ICAR("u", W=W_dense) 

        # FIX 1: Enforce a sum-to-zero constraint. 
        # This anchors the spatial effect so it doesn't drift and conflict with the Intercept.
        u_centered = u - u.mean()

        phi = pymc.Beta("phi", alpha=1, beta=1)
        tau = pymc.HalfNormal("tau", sigma=1)

        spatial_effect = tau * (pymc.math.sqrt(phi) * u_centered + pymc.math.sqrt(1 - phi) * v)

        # FIX 2: Isolate the learned parameters (eta) from the fixed population offset
        eta = Intercept + pymc.math.dot(x_shared, beta) + spatial_effect

        # Clip eta to a numerically safe range. 
        # exp(20) is ~480 million, offering plenty of headroom while strictly preventing float overflow.
        eta_clipped = pymc.math.clip(eta, -20, 20)

        # Re-add the log_offset to form the final linear predictor
        mu = eta_clipped + offset_shared

        # 3. Likelihood & Prediction
        if distribution == "Normal":
            sigma_lik = pymc.HalfCauchy("sigma_lik", 1)

            mu_transformed = pymc.math.exp(mu)

            pymc.Normal("likelihood", mu=mu_transformed[train_idx], sigma=sigma_lik, observed=Y_all[train_idx])
            pymc.Deterministic("y_predict", mu_transformed[predict_idx])

        elif distribution == "Neg_Bin":
            alpha = pymc.HalfNormal("alpha", sigma=2)

            Y_train_int = np.round(Y_all[train_idx]).astype(int)

            pymc.NegativeBinomial("likelihood", mu=pymc.math.exp(mu[train_idx]), alpha=alpha, observed=Y_train_int)
            pymc.Deterministic("y_predict", pymc.math.exp(mu[predict_idx]))

        # FIX 3: Lower the learning rate. 
        # 0.01 is too aggressive for hierarchical models with exponential links.
        mean_field = pymc.fit(method='advi', n=50000, obj_optimizer=pymc.adam(learning_rate=0.001))
        idata = mean_field.sample(1500)

        Y_predict = idata.posterior["y_predict"].median(dim=["chain", "draw"]).values

    return pd.Series(Y_predict, index=predict_idx)



def Fill_NAs(data: gpd.GeoDataFrame, config: dict) -> gpd.GeoDataFrame:


    source = config["data"]["geopackage_path"]
    knn = config["imputation"]["knn_neighbors"]
    covariates = config["imputation"]["number_covariates"]
    int_prior = config["imputation"]["int_prior"]


    # Starting up:
    #So first we fill all with 0:
    data = data.fillna(0)

    ## Wieght matrix ##

    w = libpysal.weights.KNN.from_dataframe(data, k = knn)

    # 2. Ensure it is symmetric (If A is B's nearest neighbor, B must be A's neighbor)
    # This is a strict mathematical requirement for the BYM2 ICAR prior!
    w.symmetrize()

    # 3. Revert to unstandardized binary weights
    w.transform = 'O'

    # 4. Extract for PyMC
    w_sparse = w.sparse
    w_upper = sp.triu(w_sparse)
    node1, node2 = w_upper.nonzero()

    print(f"Total edges (KNN): {len(node1)}")


    print("Municipalities per component:")
    print(pd.Series(w.component_labels).value_counts())

    # 2. Find the ID of the largest component
    main_label = pd.Series(w.component_labels).mode()[0]

    # 3. Filter the GeoDataFrame to keep ONLY the mainland
    # .reset_index(drop=True) is crucial so libpysal doesn't get confused by missing indices
    data = data[w.component_labels == main_label].reset_index(drop=True)

    # Rebuild the matrix on the newly filtered dataset
    w = libpysal.weights.KNN.from_dataframe(data, k=8)
    w = w.symmetrize()
    w.transform = 'O'

        

    ## FABBRICATI_F ##

    # Make sure to compute train/predict indices as integer positions
    train_idx = np.where(data["prov_acr"] != "BZ")[0]
    predict_idx = np.where(data["prov_acr"] == "BZ")[0]
    Off_set_all = data["TOTALE"].values


    # All features for the ENTIRE dataset
    X_all = data[["R_PENSIONE_E", "R_LAVORO_DIPENDENTE_E", "DIVORZIATI_FEMMINE_F", "PROPRIETA", "R_PARTECIPAZIONE_F", "ALTRO_TITOLO", "30", "TERZO_GRADO_F", "CASE_ABITATE"]].values

    Y_all = data["R_FABBRICATI_F"].values

    # Usage
    Y_predict = Bayesian_Model(Y_all, X_all, Off_set_all, w, train_idx, predict_idx, distribution="Neg_Bin")

    Y_predict_int = Y_predict.round().astype(int)

    data.loc[data["prov_acr"]=="BZ", "R_FABBRICATI_F"] = Y_predict_int

    ## FABBRICATI_E ##

    # All features for the ENTIRE dataset
    X_all = data[["NUMERO_IMPRESE_F", "R_FABBRICATI_F", "R_PARTECIPAZIONE_F", "R_PARTECIPAZIONE_E", "MOTOCICLI_F", "AFFITTO", "PROPRIETA", "QUARTO_GRADO_F", 
                "NUMERO_UNITA_AGRICOLE", "ALTRO_TITOLO","TERZO_GRADO_F", "CASE_ABITATE", "REDDITI_UNITA_AGRICOLE"]].values

    Y_all = data["R_FABBRICATI_E"].values

    # Usage
    Y_predict = Bayesian_Model(Y_all, X_all, Off_set_all, w, train_idx, predict_idx, distribution="Normal")

    Y_predict = Y_predict.round().astype(int)

    data.loc[data["prov_acr"]=="BZ", "R_FABBRICATI_E"] = Y_predict


    ## ADDETTI ##


    Off_set_all = 0


    To_fill = data.com_istat_code

    To_fill = pd.merge(To_fill, data[["com_istat_code", "TOTALE", "DIPENDENTI_MICRO_IMPRESE_RATIO"]], on = "com_istat_code")

    for i in data.columns:
        stripped_name = i.strip("_")

        if "ADDETTI" in stripped_name:
            To_fill = pd.merge(To_fill, data[["com_istat_code", i]], on = "com_istat_code")


    for j in To_fill.columns:

        if j == "com_istat_code" or j == "TOTALE":
            print("Skip")

        else:

            top_features = corr_relation(data, j, quantile = 0.5)
            
            train_idx = np.where(data[j] != 0)[0]
            predict_idx = np.where(data[j] == 0)[0]

            top = top_features.nlargest(covariates)
            bottom = top_features.nsmallest(covariates)

            selected_cols = top.index.tolist() + bottom.index.tolist()

            # All features for the ENTIRE dataset
            X_all = data[selected_cols].values

            Y_all = data[j].values

            # Usage
            Y_predict = Bayesian_Model(Y_all, X_all, Off_set_all, w, train_idx, predict_idx, distribution="Normal", int_prior = int_prior) #We put a strong prior to force the output to be lower than 3%

            #print(f"Mean: {np.mean(Y_predict)}. \n Max: {np.max(Y_predict)}. \n Std: {np.std(Y_predict)}. \n Number of elements: {Y_predict.shape}")

            Y_predict = Y_predict.round().astype(float)

            data.loc[data[j]==0, j] = Y_predict




    return data